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

def logits_under_ablation(
    model: HookedTransformer,
    lens: LearnedDict,
    location: standard_metrics.Location,
    ablated_directions: List[int],
    tokens: TensorType["batch", "sequence"],
    calc_fvu: bool = False,
) -> Tuple[TensorType["batch", "sequence"], Optional[TensorType["batch", "sequence"]]]:
    
    fvu = None

    def intervention(tensor, hook=None):
        B, L, D = tensor.shape
        tensor = tensor.reshape(-1, D)
        codes = lens.encode(tensor)
        ablation = torch.einsum("be,ed->bd", codes[:, ablated_directions], lens.get_learned_dict()[ablated_directions])
        ablated = tensor - ablation

        if calc_fvu:
            nonlocal fvu
            fvu = (ablation ** 2).sum() / (tensor ** 2).sum()
        
        return ablated.reshape(B, L, D)
    
    logits = model.run_with_hooks(
        tokens,
        return_type="logits",
        fwd_hooks=[(
            standard_metrics.get_model_tensor_name(location),
            intervention,
        )]
    )

    return logits, fvu

def logits_under_reconstruction(
    model: HookedTransformer,
    lens: LearnedDict,
    location: standard_metrics.Location,
    ablated_directions: List[int],
    tokens: TensorType["batch", "sequence"],
    calc_fvu: bool = False,
    resample: Optional[TensorType["batch", "sequence", "n_dict_components"]] = None,
) -> Tuple[TensorType["batch", "sequence"], Optional[TensorType["batch", "sequence"]]]:
    fvu = None

    def intervention(tensor, hook=None):
        B, L, D = tensor.shape
        code = lens.encode(tensor.reshape(-1, D))
        if resample is not None:
            code[:, ablated_directions] = resample.reshape(-1, code.shape[-1])[:, ablated_directions]
        else:
            code[:, ablated_directions] = 0.0
        reconstruction = lens.decode(code).reshape(B, L, D)

        if calc_fvu:
            nonlocal fvu
            residuals = reconstruction - tensor
            fvu = (residuals ** 2).sum() / (tensor ** 2).sum()

        return reconstruction
    
    logits = model.run_with_hooks(
        tokens,
        return_type="logits",
        fwd_hooks=[(
            standard_metrics.get_model_tensor_name(location),
            intervention,
        )]
    )

    return logits, fvu

def bottleneck_test(
    model: HookedTransformer,
    lens: LearnedDict,
    location: standard_metrics.Location,
    tokens: TensorType["batch", "sequence"],
    logit_metric: Callable[[TensorType["batch", "sequence"]], TensorType["batch"]],
    calc_fvu: bool = False,
    ablation_type: Literal["ablation", "reconstruction"] = "ablation",
    feature_sample_size: Optional[int] = None,
) -> List[Tuple[int, Optional[float], float]]:
    # iteratively ablate away the least useful directions in the bottleneck

    remaining_directions = list(range(lens.n_dict_components()))

    results = []
    ablated_directions = []

    for i in tqdm.tqdm(range(lens.n_dict_components())):
        min_score = None
        min_direction = None
        min_fvu = None

        features_to_test = None

        if feature_sample_size is not None:
            if feature_sample_size < len(remaining_directions):
                features_to_test = np.random.choice(remaining_directions, size=feature_sample_size, replace=False)
            else:
                features_to_test = remaining_directions
        else:
            features_to_test = remaining_directions

        for direction in features_to_test:
            if ablation_type == "ablation":
                logits, fvu = logits_under_ablation(model, lens, location, ablated_directions + [direction], tokens, calc_fvu=calc_fvu)
            elif ablation_type == "reconstruction":
                logits, fvu = logits_under_reconstruction(model, lens, location, ablated_directions + [direction], tokens, calc_fvu=calc_fvu)
            else:
                raise ValueError(f"Unknown ablation type '{ablation_type}'")

            score = logit_metric(logits).item()
            fvu = fvu.item()

            if min_score is None or score < min_score:
                min_score = score
                min_direction = direction
                min_fvu = fvu

        results.append((min_direction, min_fvu, min_score))
        ablated_directions.append(min_direction)
        remaining_directions.remove(min_direction)
    
    return results

def resample_ablation(
    model: HookedTransformer,
    lens: LearnedDict,
    location: standard_metrics.Location,
    clean_tokens: TensorType["batch", "sequence"],
    corrupted_codes: TensorType["batch", "sequence", "n_dict_components"],
    features_to_ablate: List[int],
    ablation_type: Literal["ablation", "reconstruction"] = "ablation",
    handicap: Optional[TensorType["batch", "sequence", "d_activation"]] = None,
    **kwargs,
) -> Tuple[Any, TensorType["batch", "sequence", "d_activation"]]:
    corrupted_codes_ = corrupted_codes.reshape(-1, corrupted_codes.shape[-1])
    ablated_activation = None

    def reconstruction_intervention(tensor, hook=None):
        nonlocal ablated_activation
        B, L, D = tensor.shape
        code = lens.encode(tensor.reshape(-1, D))
        code[:, features_to_ablate] = corrupted_codes_[:, features_to_ablate]
        reconstr = lens.decode(code).reshape(tensor.shape)

        if handicap is not None:
            output = reconstr + handicap
        else:
            output = reconstr
        
        ablated_activation = output.clone()
        return output
    
    def ablation_intervention(tensor, hook=None):
        nonlocal ablated_activation
        B, L, D = tensor.shape
        code = lens.encode(tensor.reshape(-1, D))

        ablation_code = torch.zeros_like(code)
        ablation_code[:, features_to_ablate] = corrupted_codes_[:, features_to_ablate] - code[:, features_to_ablate]
        ablation = lens.decode(ablation_code).reshape(tensor.shape)

        if handicap is not None:
            output = tensor + ablation + handicap
        else:
            output = tensor + ablation
        
        ablated_activation = output.clone()
        return output

    logits = model.run_with_hooks(
        clean_tokens,
        fwd_hooks=[(
            standard_metrics.get_model_tensor_name(location),
            reconstruction_intervention if ablation_type == "reconstruction" else ablation_intervention,
        )],
        **kwargs,
    )

    return logits, ablated_activation

def activation_info(
    model: HookedTransformer,
    lens: LearnedDict,
    location: standard_metrics.Location,
    tokens: TensorType["batch", "sequence"],
    ablation_type: Literal["ablation", "reconstruction"] = "ablation",
    replacement_residuals: Optional[TensorType["batch", "sequence", "d_activation"]] = None,
) -> Tuple[TensorType["batch", "sequence", "d_activation"], TensorType["batch", "sequence", "n_dict_components"], TensorType["batch", "sequence", "d_activation"], TensorType["batch", "sequence", "vocab_size"]]:
    residuals = None
    codes = None
    activations = None
    logits = None

    def intervention(tensor, hook=None):
        nonlocal residuals, codes, activations
        B, L, D = tensor.shape
        activations = tensor.clone()
        code = lens.encode(tensor.reshape(-1, D))
        codes = code.reshape(B, L, -1).clone()
        output = lens.decode(code).reshape(tensor.shape)
        residuals = tensor - output

        if ablation_type == "reconstruction":
            return output
        else:
            if replacement_residuals is not None:
                return output + replacement_residuals
            else:
                return tensor
    
    logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(
            standard_metrics.get_model_tensor_name(location),
            intervention,
        )],
        return_type="logits",
    )

    return residuals, codes, activations, logits

def scaled_distance_to_clean(clean_activation, corrupted_activation, activation):
    total_dist = torch.norm(clean_activation - corrupted_activation, dim=(-1, -2))
    dist = torch.norm(clean_activation - activation, dim=(-1, -2))
    return dist / total_dist

def dot_difference_metric(clean_activation, corrupted_activation, activation):
    dataset_diff_vector = corrupted_activation - clean_activation
    diff_vector = activation - clean_activation
    return torch.einsum("bld,bld->b", diff_vector, dataset_diff_vector) / torch.norm(dataset_diff_vector, dim=(-1, -2)) ** 2

def acdc_test(
    model: HookedTransformer,
    lens: LearnedDict,
    location: standard_metrics.Location,
    clean_tokens: TensorType["batch", "sequence"],
    corrupted_tokens: TensorType["batch", "sequence"],
    logit_metric: Callable[[TensorType["batch", "sequence", "vocab_size"], TensorType["batch", "sequence", "vocab_size"]], float],
    threshold: float = 0.05,
    base_logits: Optional[TensorType["batch", "sequence", "vocab_size"]] = None,
    ablation_type: Literal["ablation", "reconstruction"] = "reconstruction",
    ablation_handicap: bool = False,
    distance_metric: Callable[[TensorType["batch", "sequence", "d_activation"], TensorType["batch", "sequence", "d_activation"], TensorType["batch", "sequence", "d_activation"]], TensorType["batch"]] = scaled_distance_to_clean,
) -> Tuple[List[int], float]:
    remaining_directions = list(range(lens.n_dict_components()))
    ablated_directions = []

    corrupted_residuals, corrupted_codes, corrupted_activation, _ = activation_info(
        model,
        lens,
        location,
        corrupted_tokens,
        ablation_type=ablation_type
    )

    clean_residuals, _, clean_activation, reconstruction_logits = activation_info(
        model,
        lens,
        location,
        clean_tokens,
        ablation_type=ablation_type,
        replacement_residuals=corrupted_residuals,
    )

    handicap = None
    if ablation_handicap:
        handicap = corrupted_residuals - clean_residuals

    if base_logits is None:
        base_logits = reconstruction_logits

    prev_divergence = logit_metric(reconstruction_logits, base_logits)

    idxs = np.arange(lens.n_dict_components())
    np.random.shuffle(idxs)

    for i in tqdm.tqdm(idxs):
        logits, activation = resample_ablation(
            model,
            lens,
            location,
            clean_tokens,
            corrupted_codes,
            features_to_ablate=ablated_directions + [i],
            return_type="logits",
            ablation_type=ablation_type,
            handicap=handicap,
        )

        divergence = logit_metric(logits, base_logits)

        if divergence - prev_divergence < threshold:
            prev_divergence = divergence
            ablated_directions.append(i)
            remaining_directions.remove(i)

    distance = distance_metric(clean_activation, corrupted_activation, activation)

    return remaining_directions, prev_divergence, distance.mean().item()

def diff_mean_activation_editing(
    model: HookedTransformer,
    location: standard_metrics.Location,
    clean_tokens: TensorType["batch", "sequence"],
    corrupted_tokens: TensorType["batch", "sequence"],
    logit_metric: Callable[[TensorType["batch", "sequence", "vocab_size"], TensorType["batch", "sequence", "vocab_size"]], float],
    scale_range: Tuple[float, float] = (0.0, 1.0),
    n_points: int = 10,
    distance_metric: Callable[[TensorType["batch", "sequence", "d_activation"], TensorType["batch", "sequence", "d_activation"], TensorType["batch", "sequence", "d_activation"]], TensorType["batch"]] = scaled_distance_to_clean,
) -> List[Tuple[float, float, float]]:
    clean_logits, activation_cache = model.run_with_cache(
        clean_tokens,
        #names_filter=[standard_metrics.get_model_tensor_name(location)],
        return_type="logits",
    )
    clean_activation = activation_cache[standard_metrics.get_model_tensor_name(location)]

    _, activation_cache = model.run_with_cache(
        corrupted_tokens,
        #names_filter=[standard_metrics.get_model_tensor_name(location)],
        return_type="logits",
    )
    corrupted_activation = activation_cache[standard_metrics.get_model_tensor_name(location)]

    diff_means_vector = clean_activation.mean(dim=0) - corrupted_activation.mean(dim=0)

    scales = torch.linspace(*scale_range, n_points)

    scores = []
    for scale in tqdm.tqdm(scales):
        activation = None

        def intervention(tensor, hook):
            nonlocal activation
            activation = tensor + scale * diff_means_vector
            return activation

        logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(
                standard_metrics.get_model_tensor_name(location),
                intervention,
            )],
            return_type="logits",
        )

        distance = distance_metric(clean_activation, corrupted_activation, activation).mean().item()
        logit_score = logit_metric(logits, clean_logits)
        scores.append((scale, distance, logit_score))
    
    return scores

if __name__ == "__main__":
    torch.autograd.set_grad_enabled(False)

    model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped")

    gpus = [f"cuda:{i}" for i in range(1)]

    device = "cuda:0"

    model.to(device)

    ioi_clean_full, ioi_corrupted_full = generate_ioi_dataset(model.tokenizer, 50, 50)
    ioi_clean = ioi_clean_full[:, :-1].to(device)
    ioi_corrupted = ioi_corrupted_full[:, :-1].to(device)
    ioi_correct = ioi_clean_full[:, -1].to(device)
    ioi_incorrect = ioi_corrupted_full[:, -1].to(device)

    base_logits = model(ioi_clean, return_type="logits")

    def divergence_metric(new_logits, base_logits):
        B, L, V = base_logits.shape
        new_logprobs = F.log_softmax(new_logits[:, -1], dim=-1)
        base_logprobs = F.log_softmax(base_logits[:, -1], dim=-1)
        return F.kl_div(new_logprobs, base_logprobs, log_target=True, reduction="none").sum(dim=-1).mean().item()

    def logit_diff(new_logits, base_logits):
        B, L, V = base_logits.shape
        correct = new_logits[:, -1, ioi_correct]
        incorrect = new_logits[:, -1, ioi_incorrect]
        return -(correct - incorrect).mean().item()

    layer = 2
    activation_dataset = torch.load(f"activation_data_layers/layer_{layer}/0.pt")
    activation_dataset = activation_dataset.to(device, dtype=torch.float32)

    #diff_mean_scores = diff_mean_activation_editing(
    #    model,
    #    (layer, "residual"),
    #    ioi_clean,
    #    ioi_corrupted,
    #    divergence_metric,
    #    n_points=100,
    #    scale_range=(-10.0, 100.0),
    #)

    pca = BatchedPCA(n_dims=activation_dataset.shape[-1], device=device)
    batch_size = 4096

    print("training pca")
    for i in tqdm.trange(0, activation_dataset.shape[0], batch_size):
        j = min(i + batch_size, activation_dataset.shape[0])
        pca.train_batch(activation_dataset[i:j])
    
    pca_dict = pca.to_rotation_dict(activation_dataset.shape[-1])

    pca_dict.to_device(device)

    max_fvu = 0.05
    best_dicts = {}
    ratios = [4]
    dict_sets = [(ratio, torch.load(f"/mnt/ssd-cluster/bigrun0308/tied_residual_l{layer}_r{ratio}/_9/learned_dicts.pt")) for ratio in ratios]

    print("evaluating dicts")
    for ratio, dicts in tqdm.tqdm(dict_sets):
        for dict, hyperparams in dicts:
            dict.to_device(device)
            sample_idxs = np.random.choice(activation_dataset.shape[0], size=50000, replace=False)
            fvu = standard_metrics.fraction_variance_unexplained(dict, activation_dataset[sample_idxs]).item()
            if fvu < max_fvu:
                if hyperparams["dict_size"] not in best_dicts:
                    best_dicts[hyperparams["dict_size"]] = (fvu, hyperparams, dict)
                else:
                    if fvu > best_dicts[hyperparams["dict_size"]][0]:
                        best_dicts[hyperparams["dict_size"]] = (fvu, hyperparams, dict)
    
    del activation_dataset

    dictionaries = {}
    dictionaries["pca"] = (pca_dict, {"pca": True})
    for dict_size, (_, hyperparams, dict) in best_dicts.items():
        dictionaries[f"learned_{dict_size}"] = (dict, hyperparams)

    tau_values = np.logspace(-6.5, -1.5, 10)

    scores = {}

    for name, (dict, _) in dictionaries.items():
        scores[name] = []
        print("evaluating", name)
        for i, tau in enumerate(tau_values):
            graph, div, corruption = acdc_test(
                model,
                dict, (layer, "residual"),
                ioi_clean,
                ioi_corrupted,
                divergence_metric,
                threshold=tau,
                ablation_type="ablation",
                base_logits=base_logits,
                ablation_handicap=True,
                distance_metric=scaled_distance_to_clean,
            )
            scores[name].append((tau, graph, div, corruption))
            print(f"tau: {tau:.3e} ({i+1}/{len(tau_values)}), graph size: {len(graph)}, div: {div:.3e}, corruption: {corruption:.2f}")

    torch.save(scores, f"dict_scores_layer_{layer}.pt")
    torch.save(dictionaries, f"dictionaries_layer_{layer}.pt")

    #torch.save(diff_mean_scores, f"diff_mean_scores_layer_{layer}.pt")