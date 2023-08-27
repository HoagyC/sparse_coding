import os
from functools import partial
from itertools import product
from typing import List, Tuple, Union, Any, Dict, Literal, Optional, Callable

from datasets import load_dataset
from einops import rearrange
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchtyping import TensorType

import tqdm

from transformer_lens import HookedTransformer

from autoencoders.learned_dict import LearnedDict
from autoencoders.pca import BatchedPCA

from activation_dataset import setup_data

import standard_metrics

import copy

from test_datasets.ioi import generate_ioi_dataset
from test_datasets.gender import generate_gender_dataset

from concept_erasure import LeaceEraser

from bottleneck_test import resample_ablation_hook

from datasets import Dataset, load_dataset

BASE_FOLDER = "~/sparse_coding_aidan"

def ablate_dirs_intervention(lens, loc, features_to_ablate):
    def ablation_intervention(tensor, hook=None):
        print(features_to_ablate)

        B, L, D = tensor.shape
        code = lens.encode(tensor.reshape(-1, D))

        ablation_code = torch.zeros_like(code)
        ablation_code[:, features_to_ablate] = -code[:, features_to_ablate]
        ablation = lens.decode(ablation_code).reshape(tensor.shape)

        output = tensor + ablation
        return output

    return ablation_intervention

def eraser_intervention(eraser):
    def hook_func(tensor, hook=None):
        B, L, D = tensor.shape
        return eraser(tensor.reshape(-1, D)).reshape(tensor.shape)
    
    return hook_func

def identity_hook(tensor, hook=None):
    return tensor

def kl_div(base_logits, new_logits):
    B, L, D = base_logits.shape
    base_probs = F.log_softmax(base_logits.reshape(B*L, D), dim=-1)
    new_probs = F.log_softmax(new_logits.reshape(B*L, D), dim=-1)

    return F.kl_div(new_probs, base_probs, log_target=True, reduction="batchmean").item()

def eval_hook(model, hook_func, token_dataset, location, base_logits, device, batch_size=100, max_batches=1):
    model.eval()

    sum_kl = 0

    with torch.no_grad():
        for i in  tqdm.tqdm(range(max_batches)):
            batch = token_dataset[i*batch_size:(i+1)*batch_size]
            logits = model.run_with_hooks(
                batch.to(device),
                fwd_hooks = [(
                    standard_metrics.get_model_tensor_name(location),
                    hook_func
                )],
                return_type="logits",
            )
            sum_kl += kl_div(base_logits[i].to(logits.device), logits)

    return sum_kl / (i + 1)

if __name__ == "__main__":
    torch.autograd.set_grad_enabled(False)

    device = torch.device("cuda:1")

    model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device=device)

    dataset_name = "NeelNanda/pile-10k"
    token_amount = 256
    token_dataset = load_dataset(dataset_name, split="train").map(
        lambda x: model.tokenizer(x['text']),
        batched=True,
    ).filter(
        lambda x: len(x['input_ids']) > token_amount
    ).map(
        lambda x: {'input_ids': x['input_ids'][:token_amount]}
    )

    batch_size = 100
    n_batches = 10

    token_tensors = torch.stack([torch.tensor(data["input_ids"], dtype=torch.long, device="cpu") for data in token_dataset], dim=0)
    token_tensors.pin_memory()

    del token_dataset

    base_logits = []
    for i in tqdm.tqdm(range(0, n_batches)):
        base_logits.append(model(
            token_tensors[i*batch_size:(i+1)*batch_size].to(device),
            return_type="logits",
        ).to("cpu"))
        base_logits[-1].pin_memory()

    scores = {}

    eraser = torch.load(os.path.join(BASE_FOLDER, "leace_eraser_layer_2.pt"))
    eraser_hook = eraser_intervention(eraser)
    scores["LEACE"] = eval_hook(model, eraser_hook, token_tensors, (2, "residual"), base_logits, max_batches=n_batches, batch_size=batch_size, device=device)

    print(f"KL LEACE: {scores['LEACE']}")

    dicts = torch.load(os.path.join(BASE_FOLDER, "erasure_dictionaries_layer_2.pt"))
    dict, _ = dicts["learned_2048"]

    idxs_sets = torch.load(os.path.join(BASE_FOLDER, "erasure_scores_layer_2.pt"))["learned_2048"]

    for i, (idxs, _, _) in enumerate(idxs_sets):
        hook, _ = resample_ablation_hook(
            lens=dict,
            features_to_ablate=idxs,
            ablation_rank="full",
        )

        scores[f"learned_2048_{i}"] = eval_hook(
            model,
            hook,
            token_tensors,
            (2, "residual"),
            base_logits,
            max_batches=n_batches,
            batch_size=batch_size,
            device=device,
        )

        print(f"KL Learned {i}: {scores[f'learned_2048_{i}']}")
    
    torch.save(scores, os.path.join(BASE_FOLDER, "kl_div_scores_layer_2.pt"))