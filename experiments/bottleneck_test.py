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

_batch, _sequence, _n_dict_components, _d_activation, _vocab_size = None, None, None, None, None

BASE_FOLDER = "~/sparse_coding_aidan"

def logits_under_ablation(
    model: HookedTransformer,
    lens: LearnedDict,
    location: standard_metrics.Location,
    ablated_directions: List[int],
    tokens: TensorType["_batch", "_sequence"],
    calc_fvu: bool = False,
) -> Tuple[TensorType["_batch", "_sequence"], Optional[TensorType["_batch", "_sequence"]]]:
    
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
    tokens: TensorType["_batch", "_sequence"],
    calc_fvu: bool = False,
    resample: Optional[TensorType["_batch", "_sequence", "_n_dict_components"]] = None,
) -> Tuple[TensorType["_batch", "_sequence"], Optional[TensorType["_batch", "_sequence"]]]:
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
    tokens: TensorType["_batch", "_sequence"],
    logit_metric: Callable[[TensorType["_batch", "_sequence"]], TensorType["_batch"]],
    calc_fvu: bool = False,
    ablation_type: Literal["ablation", "reconstruction"] = "ablation",
    feature_sample_size: Optional[int] = None,
) -> List[Tuple[int, Optional[float], float]]:
    # iteratively ablate away the least useful directions in the bottleneck

    remaining_directions = list(range(lens.n_dict_components()))

    results = []
    ablated_directions: List[int] = []

    for i in tqdm.tqdm(range(lens.n_dict_components())):
        min_score = None
        min_direction = -1
        min_fvu = None

        features_to_test: List[int] = []

        if feature_sample_size is not None:
            if feature_sample_size < len(remaining_directions):
                features_to_test = list(np.random.choice(remaining_directions, size=feature_sample_size, replace=False))
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

            if calc_fvu:
                assert fvu is not None
                fvu_item: float = fvu.item()

            if min_score is None or score < min_score:
                min_score = score
                min_direction = direction
                min_fvu = fvu_item

        assert min_direction != -1
        assert min_score is not None
        results.append((min_direction, min_fvu, min_score))
        ablated_directions.append(min_direction)
        remaining_directions.remove(min_direction)
    
    return results

def resample_ablation_hook(
    lens: LearnedDict,
    features_to_ablate: List[int],
    corrupted_codes: Optional[TensorType["_batch", "_sequence", "_n_dict_components"]] = None,
    ablation_type: Literal["ablation", "reconstruction"] = "ablation",
    handicap: Optional[TensorType["_batch", "_sequence", "_d_activation"]] = None,
    ablation_rank: Literal["full", "partial"] = "partial",
    ablation_mask: Optional[TensorType["_batch", "_sequence"]] = None,
):
    if corrupted_codes is None:
        corrupted_codes_ = None
    else:
        corrupted_codes_ = corrupted_codes.reshape(-1, corrupted_codes.shape[-1])

    activation_dict = {"output": None}
    
    def reconstruction_intervention(tensor, hook=None):
        nonlocal activation_dict
        B, L, D = tensor.shape
        code = lens.encode(tensor.reshape(-1, D))

        if corrupted_codes_ is None:
            code[:, features_to_ablate] = 0.0
        else:
            code[:, features_to_ablate] = corrupted_codes_[:, features_to_ablate]
        
        reconstr = lens.decode(code).reshape(tensor.shape)

        if handicap is not None:
            output = reconstr + handicap
        else:
            output = reconstr

        if ablation_mask is not None:
            output[~ablation_mask] = tensor[~ablation_mask]

        activation_dict["output"] = output.clone()
        return output
    
    def partial_ablation_intervention(tensor, hook=None):
        nonlocal activation_dict
        B, L, D = tensor.shape
        code = lens.encode(tensor.reshape(-1, D))

        ablation_code = torch.zeros_like(code)

        if corrupted_codes_ is None:
            ablation_code[:, features_to_ablate] = -code[:, features_to_ablate]
        else:
            ablation_code[:, features_to_ablate] = corrupted_codes_[:, features_to_ablate] - code[:, features_to_ablate]
        
        ablation = lens.decode(ablation_code).reshape(tensor.shape)

        if handicap is not None:
            output = tensor + ablation + handicap
        else:
            output = tensor + ablation

        if ablation_mask is not None:
            output[~ablation_mask] = tensor[~ablation_mask]

        activation_dict["output"] = output.clone()
        return output

    def full_ablation_intervention(tensor, hook=None):
        nonlocal activation_dict
        B, L, D = tensor.shape
        code = torch.einsum("bd,nd->bn", tensor.reshape(-1,D), lens.get_learned_dict())

        ablation_code = torch.zeros_like(code)

        if corrupted_codes_ is None:
            ablation_code[:, features_to_ablate] = -code[:, features_to_ablate]
        else:
            ablation_code[:, features_to_ablate] = corrupted_codes_[:, features_to_ablate] - code[:, features_to_ablate]

        ablation = torch.einsum("bn,nd->bd", ablation_code, lens.get_learned_dict()).reshape(tensor.shape)
        output = tensor + ablation

        if ablation_mask is not None:
            output[~ablation_mask] = tensor[~ablation_mask]

        activation_dict["output"] = output.clone()
        return tensor + ablation

    ablation_func = None
    if ablation_type == "reconstruction":
        ablation_func = reconstruction_intervention
    elif ablation_type == "ablation" and ablation_rank == "partial":
        ablation_func = partial_ablation_intervention
    elif ablation_type == "ablation" and ablation_rank == "full":
        ablation_func = full_ablation_intervention
    else:
        raise ValueError(f"Unknown ablation type '{ablation_type}' with rank '{ablation_rank}'")
    
    return ablation_func, activation_dict

def resample_ablation(
    model: HookedTransformer,
    lens: LearnedDict,
    location: standard_metrics.Location,
    clean_tokens: TensorType["_batch", "_sequence"],
    features_to_ablate: List[int],
    corrupted_codes: Optional[TensorType["_batch", "_sequence", "_n_dict_components"]] = None,
    ablation_type: Literal["ablation", "reconstruction"] = "ablation",
    handicap: Optional[TensorType["_batch", "_sequence", "_d_activation"]] = None,
    ablation_rank: Literal["full", "partial"] = "partial",
    ablation_mask: Optional[TensorType["_batch", "_sequence"]] = None,
    **kwargs,
) -> Tuple[Any, TensorType["_batch", "_sequence", "_d_activation"]]:
    ablation_func, activation_dict = resample_ablation_hook(
        lens,
        features_to_ablate,
        corrupted_codes=corrupted_codes,
        ablation_type=ablation_type,
        handicap=handicap,
        ablation_rank=ablation_rank,
        ablation_mask=ablation_mask,
    )

    logits = model.run_with_hooks(
        clean_tokens,
        fwd_hooks=[(
            standard_metrics.get_model_tensor_name(location),
            ablation_func,
        )],
        **kwargs,
    )

    return logits, activation_dict["output"]

def activation_info(
    model: HookedTransformer,
    lens: LearnedDict,
    location: standard_metrics.Location,
    tokens: TensorType["_batch", "_sequence"],
    ablation_type: Literal["ablation", "reconstruction"] = "ablation",
    replacement_residuals: Optional[TensorType["_batch", "_sequence", "_d_activation"]] = None,
) -> Tuple[TensorType["_batch", "_sequence", "_d_activation"], TensorType["_batch", "_sequence", "_n_dict_components"], TensorType["_batch", "_sequence", "_d_activation"], TensorType["_batch", "_sequence", "_vocab_size"]]:
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
    clean_tokens: TensorType["_batch", "_sequence"],
    corrupted_tokens: TensorType["_batch", "_sequence"],
    logit_metric: Callable[[TensorType["_batch", "_sequence", "_vocab_size"], TensorType["_batch", "_sequence", "_vocab_size"]], float],
    threshold: float = 0.05,
    base_logits: Optional[TensorType["_batch", "_sequence", "_vocab_size"]] = None,
    ablation_type: Literal["ablation", "reconstruction"] = "reconstruction",
    ablation_handicap: bool = False,
    distance_metric: Callable[[TensorType["_batch", "_sequence", "_d_activation"], TensorType["_batch", "_sequence", "_d_activation"], TensorType["_batch", "_sequence", "_d_activation"]], TensorType["_batch"]] = scaled_distance_to_clean,
) -> Tuple[List[int], float, float]:
    remaining_directions = list(range(lens.n_dict_components()))
    ablated_directions: List[int] = []

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
            corrupted_codes=corrupted_codes,
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
    clean_tokens: TensorType["_batch", "_sequence"],
    corrupted_tokens: TensorType["_batch", "_sequence"],
    logit_metric: Callable[[TensorType["_batch", "_sequence", "_vocab_size"], TensorType["_batch", "_sequence", "_vocab_size"]], float],
    scale_range: Tuple[float, float] = (0.0, 1.0),
    n_points: int = 10,
    distance_metric: Callable[[TensorType["_batch", "_sequence", "_d_activation"], TensorType["_batch", "_sequence", "_d_activation"], TensorType["_batch", "_sequence", "_d_activation"]], TensorType["_batch"]] = scaled_distance_to_clean,
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

def ce_distance(clean_activation, activation):
    return torch.linalg.norm(clean_activation - activation, dim=(-1,-2))

def ablation_mask_from_seq_lengths(
    seq_lengths: TensorType["_batch"],
    max_length: int,
) -> TensorType["_batch", "_sequence"]:
    B = seq_lengths.shape[0]
    mask = torch.zeros((B, max_length), dtype=torch.bool)
    for i in range(B):
        mask[i, :seq_lengths[i]] = True
    return mask

def concept_ablation(
    model: HookedTransformer,
    lens: LearnedDict,
    location: standard_metrics.Location,
    dataset: TensorType["_batch", "_sequence"],
    scoring_function: Callable[[TensorType["_batch", "_sequence", "_vocab_size"]], float],
    max_features_removed: int = 10,
    distance_metric: Callable[[TensorType["_batch", "_sequence", "_d_activation"], TensorType["_batch", "_sequence", "_d_activation"]], TensorType["_batch"]] = ce_distance,
    ablation_type: Literal["ablation", "reconstruction"] = "ablation",
    ablation_rank: Literal["full", "partial"] = "partial",
    sequence_lengths: Optional[TensorType["_batch"]] = None,
    scale_by_magnitude: bool = False,
    min_perf_decrease: float = 1.0, # to stop scale_by_magnitude from removing unimportant features
) -> List[Tuple[List[int], float, float]]:
    """Try and add as much data back as possible while keeping a specific concept erased"""
    if sequence_lengths is not None:
        ablation_mask = ablation_mask_from_seq_lengths(sequence_lengths, dataset.shape[1])
    else:
        ablation_mask = None
    
    ablated_directions: List[int] = []
    remaining_directions = list(range(lens.n_dict_components()))

    _, activation_cache = model.run_with_cache(
        dataset,
        names_filter=lambda name: name == standard_metrics.get_model_tensor_name(location),
        return_type="logits",
    )
    clean_activation = activation_cache[standard_metrics.get_model_tensor_name(location)]

    logits, activation = resample_ablation(
        model,
        lens,
        location,
        dataset,
        corrupted_codes=None,
        features_to_ablate=ablated_directions,
        return_type="logits",
        ablation_type=ablation_type,
        ablation_rank=ablation_rank,
        ablation_mask=ablation_mask,
    )

    scores = []

    prev_score = float("inf")
    for iteration in range(max_features_removed):
        min_weighted_score = float("inf")
        min_score = float("inf")
        min_idx = None
        min_activation_dist = float("inf")

        for i in tqdm.tqdm(range(lens.n_dict_components())):
            logits, activation = resample_ablation(
                model,
                lens,
                location,
                dataset,
                corrupted_codes=None,
                features_to_ablate=ablated_directions + [i],
                return_type="logits",
                ablation_type=ablation_type,
                ablation_rank=ablation_rank,
                ablation_mask=ablation_mask,
            )

            score = scoring_function(logits)

            if scale_by_magnitude:
                weighted_score = score * distance_metric(clean_activation, activation).mean().item()
            else:
                weighted_score = score

            if weighted_score < min_weighted_score and score < prev_score * min_perf_decrease:
                min_weighted_score = weighted_score
                min_score = score
                min_idx = i
                min_activation_dist = distance_metric(clean_activation, activation).mean().item()
        
        if min_idx is None:
            print("Early stopped at iteration", iteration, "with score", prev_score)
            break

        ablated_directions.append(min_idx)
        remaining_directions.remove(min_idx)
        prev_score = min_score

        print(f"Removed {min_idx} with score {min_score} and activation distance {min_activation_dist}")

        scores.append((ablated_directions.copy(), min_score, min_activation_dist))

    return scores

def least_squares_erasure(
    model: HookedTransformer,
    location: standard_metrics.Location,
    dataset: TensorType["_batch", "_sequence"],
    classes: TensorType["_batch"],
    scoring_function: Callable[[TensorType["_batch", "_sequence", "_vocab_size"]], float],
    distance_metric: Callable[[TensorType["_batch", "_sequence", "_d_activation"], TensorType["_batch", "_sequence", "_d_activation"]], TensorType["_batch"]] = ce_distance,
    sequence_lengths: Optional[TensorType["_batch"]] = None,
) -> Tuple[float, float, Any]:
    if sequence_lengths is not None:
        ablation_mask = ablation_mask_from_seq_lengths(sequence_lengths, dataset.shape[1])
    else:
        ablation_mask = None

    _, activation_cache = model.run_with_cache(
        dataset,
        names_filter=lambda name: name == standard_metrics.get_model_tensor_name(location),
        return_type="logits"
    )

    if ablation_mask is None:
        ablation_mask = torch.ones_like(dataset, dtype=torch.bool)

    B, L, D = activation_cache[standard_metrics.get_model_tensor_name(location)].shape

    activations_flattened = activation_cache[standard_metrics.get_model_tensor_name(location)].reshape(B*L, D)
    classes_flattened = classes.repeat_interleave(L)
    mask_flattened = ablation_mask.reshape(B*L)

    activations = activations_flattened[mask_flattened]
    classes_ = classes_flattened[mask_flattened]

    print(activations.shape, classes_.shape)

    eraser = LeaceEraser.fit(activations, classes_) # type: ignore

    distance = None

    def erasure(tensor, hook):
        nonlocal distance
        erased = eraser(tensor.reshape(B*L, D)).reshape(B, L, D)

        if ablation_mask is not None:
            erased[~ablation_mask] = tensor[~ablation_mask]

        distance = distance_metric(tensor, erased)
        return erased
    
    logits = model.run_with_hooks(
        dataset,
        fwd_hooks=[(
            standard_metrics.get_model_tensor_name(location),
            erasure,
        )],
        return_type="logits",
    )

    score = scoring_function(logits)
    assert distance is not None

    return score, distance.mean().item(), eraser

def new_bottleneck_test():
    torch.autograd.set_grad_enabled(False)

    model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped")

    device = "cuda:7"

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

    layer = 3
    activation_dataset = torch.load(os.path.join(BASE_FOLDER, f"activation_data/layer_3/0.pt"))
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

    max_fvus = [0.2, 0.1, 0.05]
    best_dicts = {}
    ratios = [4]
    dict_sets = [(ratio, "learned_{max_fvu:.2f}", torch.load(f"/mnt/ssd-cluster/bigrun0308/tied_residual_l{layer}_r{ratio}/_9/learned_dicts.pt")) for ratio in ratios]
    dict_sets += [(4, "zero_l1_baseline", torch.load(os.path.join(BASE_FOLDER, "output_zero_b_4/_7/learned_dicts.pt")))]

    print("evaluating dicts")
    for max_fvu in max_fvus:
        for ratio, label, dicts in tqdm.tqdm(dict_sets):
            for dict, hyperparams in dicts:
                dict.to_device(device)
                sample_idxs = np.random.choice(activation_dataset.shape[0], size=50000, replace=False)
                fvu = standard_metrics.fraction_variance_unexplained(dict, activation_dataset[sample_idxs]).item()
                if fvu < max_fvu:
                    name = label.format(max_fvu=max_fvu, dict_size=hyperparams["dict_size"])

                    if name not in best_dicts:
                        best_dicts[name] = (fvu, hyperparams, dict)
                    else:
                        if fvu > best_dicts[name][0]:
                            best_dicts[name] = (fvu, hyperparams, dict)
    
    print("found satisfying dicts:", list(best_dicts.keys()))
    
    del activation_dataset

    dictionaries = {}
    dictionaries["pca"] = (pca_dict, {"pca": True})
    for name, (_, hyperparams, dict) in best_dicts.items():
        dictionaries[name] = (dict, hyperparams)

    tau_values = np.logspace(-6.5, -1.5, 10)

    scores: Dict[str, List] = {}

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

    torch.save(scores, os.path.join(BASE_FOLDER, f"dict_scores_layer_{layer}.pt"))
    torch.save(dictionaries, os.path.join(BASE_FOLDER, f"dictionaries_layer_{layer}.pt"))

    #torch.save(diff_mean_scores, os.path.join(BASE_FOLDER, f"diff_mean_scores_layer_{layer}.pt"))

def erasure_test():
    torch.autograd.set_grad_enabled(False)

    model_name = "EleutherAI/pythia-70m-deduped"

    model = HookedTransformer.from_pretrained(model_name)

    device = "cuda:1"

    model.to(device)

    prompts, classes, class_tokens, sequence_lengths = generate_gender_dataset(model_name, 100, 100, model.tokenizer.pad_token_id)
    prompts = prompts.to(device)
    classes = classes.to(device)
    sequence_lengths = sequence_lengths.to(device)
    print(sequence_lengths)
    class_one_hot = F.one_hot(classes, num_classes=2).float()

    def gender_erasure_metric(predictions):
        predictions = predictions[torch.arange(sequence_lengths.shape[0]), sequence_lengths-1]
        predictions = predictions[:, [class_tokens[0], class_tokens[1]]]
        probs = F.softmax(predictions, dim=-1)

        pred = torch.einsum("bc,bc->b", probs, class_one_hot).mean()
        return torch.abs(pred - 0.5).item() + 0.5
    
    layer = 2
    activation_dataset = torch.load(os.path.join(BASE_FOLDER, f"activation_data/layer_{layer}/0.pt"))
    activation_dataset = activation_dataset.to(device, dtype=torch.float32)

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
    for dict_size, (_, hyperparams, dict) in best_dicts.items():
        dictionaries[f"learned_{dict_size}"] = (dict, hyperparams)

    leace_score, leace_edit, leace_eraser = least_squares_erasure(
        model,
        (layer, "residual"),
        prompts,
        classes,
        scoring_function=gender_erasure_metric,
        distance_metric=ce_distance,
        sequence_lengths=sequence_lengths,
    )

    print(f"LEACE score: {leace_score:.3e}, LEACE edit: {leace_edit:.2f}")

    base_logits = model(prompts, return_type="logits")
    base_score = gender_erasure_metric(base_logits)

    print(f"base score: {base_score:.3e}")

    torch.save((leace_score, leace_edit, base_score), os.path.join(BASE_FOLDER, f"leace_scores_layer_{layer}.pt"))
    torch.save(leace_eraser, os.path.join(BASE_FOLDER, f"leace_eraser_layer_{layer}.pt"))

    scores = {}
    tau_values = np.logspace(-4, 0, 10)
    for name, (dict, _) in dictionaries.items():
        scores[name] = concept_ablation(
            model,
            dict, (layer, "residual"),
            prompts,
            scoring_function=gender_erasure_metric,
            scale_by_magnitude=False,
            sequence_lengths=sequence_lengths,
            ablation_rank="full",
        )

    torch.save(scores, os.path.join(BASE_FOLDER, f"erasure_scores_layer_{layer}.pt"))
    torch.save(dictionaries, os.path.join(BASE_FOLDER, f"erasure_dictionaries_layer_{layer}.pt"))

if __name__ == "__main__":
    erasure_test()