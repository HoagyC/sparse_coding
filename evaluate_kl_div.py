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

from erasure import NullspaceProjector

from datasets import Dataset, load_dataset

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
    # measure KL-divergence between base_logits and logits with hook_func applied to the model
    # also measure model perplexity

    model.eval()

    sum_kl = 0
    sum_ce_per_word = 0

    n_words = 0

    with torch.no_grad():
        for i in  tqdm.tqdm(range(max_batches)):
            batch = token_dataset[i*batch_size:(i+1)*batch_size].to(device)
            logits = model.run_with_hooks(
                batch,
                fwd_hooks = [(
                    standard_metrics.get_model_tensor_name(location),
                    hook_func
                )],
                return_type="logits",
            )

            log_logits = F.log_softmax(logits, dim=-1)
            log_base = F.log_softmax(base_logits[i].to(logits.device), dim=-1)

            sum_kl += F.kl_div(
                log_logits.reshape(-1, log_logits.shape[-1]),
                log_base.reshape(-1, log_base.shape[-1]),
                log_target=True,
                reduction="batchmean",
            ).item()
            sum_ce_per_word += F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch.reshape(-1),
                reduction="mean",
            ).item()

    return sum_kl / (i + 1), np.exp(sum_ce_per_word / (i + 1))

def eval_kl_div(cfg):
    torch.autograd.set_grad_enabled(False)

    device = torch.device(cfg.device)

    model = HookedTransformer.from_pretrained(cfg.model_name, device=device)

    dataset_name = cfg.dataset_name
    token_amount = cfg.sequence_length
    token_dataset = load_dataset(dataset_name, split="train").map(
        lambda x: model.tokenizer(x['text']),
        batched=True,
    ).filter(
        lambda x: len(x['input_ids']) > token_amount
    ).map(
        lambda x: {'input_ids': x['input_ids'][:token_amount]}
    )

    batch_size = cfg.batch_size
    n_batches = cfg.n_batches

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

    eraser = torch.load(f"{cfg.output_dict}/leace_eraser_layer_{cfg.layer}.pt")
    
    def leace_hook(tensor, hook=None):
        B, L, D = tensor.shape
        return eraser(tensor.reshape(-1, D)).reshape(tensor.shape)

    scores["LEACE"] = eval_hook(
        model,
        leace_hook,
        token_tensors,
        (layer, "residual"),
        base_logits,
        max_batches=n_batches,
        batch_size=batch_size,
        device=device
    )

    print(f"Scores LEACE: {scores['LEACE']}")

    #dicts = torch.load("output_erasure.pt")
    #dict, _ = dicts["learned_2048"]

    best_dict = torch.load(f"{cfg.output_dict}/best_dict_layer_{cfg.layer}.pt")
    feature_scores = torch.load(f"{cfg.output_dict}/dict_feature_scores_layer_{cfg.layer}.pt")

    best_feature_idx = min(feature_scores, key=lambda x: x[1])[0]
    best_feature = best_dict.get_learned_dict()[best_feature_idx]
    best_feature_proj = NullspaceProjector(best_feature)

    def feature_hook(tensor, hook=None):
        B, L, D = tensor.shape
        return best_feature_proj.project(tensor.reshape(-1, D)).reshape(tensor.shape)

    scores["dict"] = eval_hook(
        model,
        feature_hook,
        token_tensors,
        (cfg.layer, "residual"),
        base_logits,
        max_batches=n_batches,
        batch_size=batch_size,
        device=device,
    )

    print(f"Scores Dict: {scores['dict']}")

    means_eraser = torch.load(f"{cfg.output_dict}/means_eraser_layer_{cfg.layer}.pt")

    def means_hook(tensor, hook=None):
        B, L, D = tensor.shape
        return means_eraser.project(tensor.reshape(-1, D)).reshape(tensor.shape)
    
    scores["means"] = eval_hook(
        model,
        means_hook,
        token_tensors,
        (cfg.layer, "residual"),
        base_logits,
        max_batches=n_batches,
        batch_size=batch_size,
        device=device,
    )

    print(f"Scores Means: {scores['means']}")

    torch.save(scores, f"{cfg.output_dict}/kl_div_scores_layer_{cfg.layer}.pt")

if __name__ == "__main__":
    from utils import dotdict
    import os

    cfg = dotdict({
        "device": "cuda:4",
        "model_name": "EleutherAI/pythia-70m-deduped",
        "dataset_name": "NeelNanda/pile-10k",
        "sequence_length": 256,
        "batch_size": 100,
        "n_batches": 50,
        "layer": None,
        "output_dict": "output_erasure",
    })

    os.makedirs(cfg.output_dict, exist_ok=True)

    for layer in range(0, 6):
        cfg.layer = layer
        print(f"Layer {layer}")
        eval_kl_div(cfg)