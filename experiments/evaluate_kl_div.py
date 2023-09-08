import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

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

def eval_kl_div(cfg, layers):
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

    for layer in layers:
        scores = {}

        eraser = torch.load(f"{cfg.output_dict}/leace_eraser_layer_{layer}.pt", map_location=cfg.device)
        
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

        best_dict = torch.load(f"{cfg.output_dict}/best_dict_layer_{layer}.pt")
        feature_scores = torch.load(f"{cfg.output_dict}/dict_feature_scores_layer_{layer}.pt")

        best_feature_idx = min(feature_scores, key=lambda x: x[1])
        print(f"Best feature: {best_feature_idx}")
        best_feature_idx = best_feature_idx[0]
        best_feature = best_dict.get_learned_dict()[best_feature_idx].to(cfg.device)
        best_feature_proj = NullspaceProjector(best_feature)

        def feature_hook(tensor, hook=None):
            B, L, D = tensor.shape
            return best_feature_proj.project(tensor.reshape(-1, D)).reshape(tensor.shape)

        scores["dict"] = eval_hook(
            model,
            feature_hook,
            token_tensors,
            (layer, "residual"),
            base_logits,
            max_batches=n_batches,
            batch_size=batch_size,
            device=device,
        )

        print(f"Scores Dict: {scores['dict']}")

        means_eraser = torch.load(f"{cfg.output_dict}/means_eraser_layer_{layer}.pt", map_location=cfg.device)

        def means_hook(tensor, hook=None):
            B, L, D = tensor.shape
            return means_eraser.project(tensor.reshape(-1, D)).reshape(tensor.shape)
        
        scores["means"] = eval_hook(
            model,
            means_hook,
            token_tensors,
            (layer, "residual"),
            base_logits,
            max_batches=n_batches,
            batch_size=batch_size,
            device=device,
        )

        print(f"Scores Means: {scores['means']}")

        torch.save(scores, f"{cfg.output_dict}/kl_div_scores_layer_{layer}.pt")

if __name__ == "__main__":
    from utils import dotdict
    import os

    cfg = dotdict({
        "device": "cuda:4",
        "model_name": "EleutherAI/pythia-410m-deduped",
        "dataset_name": "NeelNanda/pile-10k",
        "sequence_length": 256,
        "batch_size": 50,
        "n_batches": 100,
        "output_dict": "output_erasure_410m",
    })

    os.makedirs(cfg.output_dict, exist_ok=True)

    layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

    eval_kl_div(cfg, layers)

def normal_kl_div_test():
    torch.autograd.set_grad_enabled(False)

    device = torch.device("cuda:1")

    model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device=device)

    dataset_name = "NeelNanda/pile-10k"
    token_amount = 256
    token_dataset = (
        load_dataset(dataset_name, split="train")
        .map(
            lambda x: model.tokenizer(x["text"]),
            batched=True,
        )
        .filter(lambda x: len(x["input_ids"]) > token_amount)
        .map(lambda x: {"input_ids": x["input_ids"][:token_amount]})
    )

    batch_size = 100
    n_batches = 10

    token_tensors = torch.stack(
        [torch.tensor(data["input_ids"], dtype=torch.long, device="cpu") for data in token_dataset],
        dim=0,
    )
    token_tensors.pin_memory()

    del token_dataset

    base_logits = []
    for i in tqdm.tqdm(range(0, n_batches)):
        base_logits.append(
            model(
                token_tensors[i * batch_size : (i + 1) * batch_size].to(device),
                return_type="logits",
            ).to("cpu")
        )
        base_logits[-1].pin_memory()
    
    scores = {}

    dicts = {}
    dicts["Uncentered"] = torch.load("outputs_sphere/no_centering_12.pt")
    dicts["Centered"] = torch.load("outputs_sphere/mean_centered_12.pt")
    dicts["Learned Center"] = torch.load("outputs_sphere/learned_centered_mean_init_12.pt")
    dicts["Sphered"] = torch.load("outputs_sphere/sphered_12.pt")

    activation_dataset = torch.load("activation_data_sphere/0.pt", map_location="cpu")
    sample_idxs = np.random.choice(len(activation_dataset), 50000, replace=False)
    activation_dataset = activation_dataset[sample_idxs].to(device, dtype=torch.float32)

    for name, dictset in dicts.items():
        scores[name] = []
        for dict, hyperparams in dictset:
            dict.to_device(device)

            sparsity = standard_metrics.mean_nonzero_activations(dict, activation_dataset).sum().item()

            def reconstruction_hook(tensor, hook=None):
                B, L, D = tensor.shape
                data = dict.predict(tensor.reshape(-1, D))
                return data.reshape(tensor.shape)

            kl_div = eval_hook(
                model,
                reconstruction_hook,
                token_tensors,
                (2, "residual"),
                base_logits,
                max_batches=n_batches,
                batch_size=batch_size,
                device=device,
            )

            scores[name].append((kl_div, sparsity))
        
    #torch.save(scores, os.path.join(BASE_FOLDER, "kl_div_scores_layer_5.pt"))
    torch.save(scores, "kl_div_scores_layer_4.pt")