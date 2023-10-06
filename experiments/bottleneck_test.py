import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import copy
from functools import partial
from itertools import product
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from concept_erasure import LeaceEraser
from datasets import load_dataset
from einops import rearrange
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchtyping import TensorType
from transformer_lens import HookedTransformer

import standard_metrics
from activation_dataset import setup_data
from autoencoders.learned_dict import LearnedDict
from autoencoders.pca import BatchedPCA
from test_datasets.gender import generate_gender_dataset
from test_datasets.ioi import generate_ioi_dataset

_batch, _sequence, _n_dict_components, _d_activation, _vocab_size = (
    None,
    None,
    None,
    None,
    None,
)

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
        ablation = torch.einsum(
            "be,ed->bd",
            codes[:, ablated_directions],
            lens.get_learned_dict()[ablated_directions],
        )
        ablated = tensor - ablation

        if calc_fvu:
            nonlocal fvu
            fvu = (ablation**2).sum() / (tensor**2).sum()

        return ablated.reshape(B, L, D)

    logits = model.run_with_hooks(
        tokens,
        return_type="logits",
        fwd_hooks=[
            (
                standard_metrics.get_model_tensor_name(location),
                intervention,
            )
        ],
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
            fvu = (residuals**2).sum() / (tensor**2).sum()

        return reconstruction

    logits = model.run_with_hooks(
        tokens,
        return_type="logits",
        fwd_hooks=[
            (
                standard_metrics.get_model_tensor_name(location),
                intervention,
            )
        ],
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
                logits, fvu = logits_under_ablation(
                    model,
                    lens,
                    location,
                    ablated_directions + [direction],
                    tokens,
                    calc_fvu=calc_fvu,
                )
            elif ablation_type == "reconstruction":
                logits, fvu = logits_under_reconstruction(
                    model,
                    lens,
                    location,
                    ablated_directions + [direction],
                    tokens,
                    calc_fvu=calc_fvu,
                )
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
        code = torch.einsum("bd,nd->bn", tensor.reshape(-1, D), lens.get_learned_dict())

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
        fwd_hooks=[
            (
                standard_metrics.get_model_tensor_name(location),
                ablation_func,
            )
        ],
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
) -> Tuple[
    TensorType["_batch", "_sequence", "_d_activation"],
    TensorType["_batch", "_sequence", "_n_dict_components"],
    TensorType["_batch", "_sequence", "_d_activation"],
    TensorType["_batch", "_sequence", "_vocab_size"],
]:
    residuals = None
    codes = None
    activations = None
    logits = None

    def intervention(tensor, hook=None):
        nonlocal residuals, codes, activations
        B, L, D = tensor.shape
        activations = tensor.clone()
        code = lens.encode(lens.center(tensor.reshape(-1, D)))
        codes = code.reshape(B, L, -1).clone()
        output = lens.uncenter(lens.decode(code)).reshape(tensor.shape)
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
        fwd_hooks=[
            (
                standard_metrics.get_model_tensor_name(location),
                intervention,
            )
        ],
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

def acdc_intervention(
    model: HookedTransformer,
    lens: LearnedDict,
    location: standard_metrics.Location,
    clean_codes: TensorType["_batch", "_sequence", "_n_dict_components"],
    corrupted_tokens: TensorType["_batch", "_sequence"],
    ablated_directions: List[int],
):
    activation = None
    def intervention(tensor, hook=None):
        nonlocal activation
        B, L, D = tensor.shape
        _, _, N = clean_codes.shape

        centered_tensor = lens.center(tensor.reshape(-1, D))

        corrupted_codes = lens.encode(centered_tensor).reshape(B, L, -1)

        corrupted_codes_to_ablate = torch.zeros_like(corrupted_codes)
        corrupted_codes_to_ablate[:, :, ablated_directions] = corrupted_codes[:, :, ablated_directions]

        corrupted_difference = lens.decode(corrupted_codes_to_ablate.reshape(-1, N))

        clean_codes_to_ablate = torch.zeros_like(clean_codes)
        clean_codes_to_ablate[:, :, ablated_directions] = clean_codes[:, :, ablated_directions]

        clean_difference = lens.decode(clean_codes_to_ablate.reshape(-1, N))

        edited_centered_tensor = centered_tensor - corrupted_difference + clean_difference
        activation = lens.uncenter(edited_centered_tensor).reshape(B, L, D)
        return activation.clone()

    logits = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[
            (
                standard_metrics.get_model_tensor_name(location),
                intervention,
            )
        ],
        return_type="logits",
    )

    return logits, activation

def acdc_test(
    model: HookedTransformer,
    lens: LearnedDict,
    location: standard_metrics.Location,
    clean_tokens: TensorType["_batch", "_sequence"],
    corrupted_tokens: TensorType["_batch", "_sequence"],
    logit_metric: Callable[
        [
            TensorType["_batch", "_sequence", "_vocab_size"],
            TensorType["_batch", "_sequence", "_vocab_size"],
        ],
        float,
    ],
    thresholds: List[float] = [0.05],
    base_logits: Optional[TensorType["_batch", "_sequence", "_vocab_size"]] = None,
    ablation_handicap: bool = False,
    distance_metric: Callable[
        [
            TensorType["_batch", "_sequence", "_d_activation"],
            TensorType["_batch", "_sequence", "_d_activation"],
            TensorType["_batch", "_sequence", "_d_activation"],
        ],
        TensorType["_batch"],
    ] = scaled_distance_to_clean,
    initial_directions: Optional[List[int]] = None,
) -> List[Tuple[List[int], float, float]]:
    ablation_type = "ablation"

    if initial_directions is None:
        initial_directions = list(range(lens.n_dict_components()))

    ablated_directions = [x for x in range(lens.n_dict_components()) if x not in initial_directions]
    remaining_directions = list(initial_directions)

    _, corrupted_codes, corrupted_activation, _ = activation_info(
        model,
        lens,
        location,
        corrupted_tokens,
        ablation_type=ablation_type
    )

    _, clean_codes, clean_activation, _ = activation_info(
        model,
        lens,
        location,
        clean_tokens,
        ablation_type=ablation_type,
    )
    
    base_logits = model(
        clean_tokens,
        return_type="logits",
    )

    scores = []

    zero_logits, zero_activation = acdc_intervention(
        model,
        lens,
        location,
        clean_codes,
        corrupted_tokens,
        remaining_directions,
    )

    zero_divergence = logit_metric(zero_logits, base_logits)
    zero_distance = distance_metric(clean_activation, corrupted_activation, zero_activation)

    scores.append(([], zero_divergence, zero_distance.mean().item()))

    prev_divergence = zero_divergence

    #print(ablated_directions, remaining_directions)

    for tau in sorted(thresholds):
        if len(remaining_directions) > 0:
            activation = None

            assert len(ablated_directions) + len(remaining_directions) == lens.n_dict_components()

            for i in tqdm.tqdm(remaining_directions.copy()):
                #logits, activation = resample_ablation(
                #    model,
                #    lens,
                #    location,
                #    clean_tokens,
                #    corrupted_codes=corrupted_codes,
                #    features_to_ablate=ablated_directions + [i],
                #    return_type="logits",
                #    ablation_type=ablation_type,
                #    handicap=handicap,
                #)
                logits, activation = acdc_intervention(
                    model,
                    lens,
                    location,
                    clean_codes,
                    corrupted_tokens,
                    [x for x in remaining_directions if x != i],
                )

                divergence = logit_metric(logits, base_logits)

                if divergence - prev_divergence < tau:
                    prev_divergence = divergence
                    ablated_directions.append(i)
                    remaining_directions.remove(i)

            distance = distance_metric(clean_activation, corrupted_activation, activation)
            scores.append((remaining_directions.copy(), prev_divergence, distance.mean().item()))

            print(f"graph size: {len(remaining_directions)} div: {prev_divergence} edit: {distance.mean().item()}")

    full_logits, full_activation = acdc_intervention(
        model,
        lens,
        location,
        clean_codes,
        corrupted_tokens,
        [],
    )

    full_divergence = logit_metric(full_logits, base_logits)
    full_distance = distance_metric(clean_activation, corrupted_activation, full_activation)

    scores.append((list(range(lens.n_dict_components())), full_divergence, full_distance.mean().item()))

    return scores

def ce_distance(clean_activation, activation):
    return torch.linalg.norm(clean_activation - activation, dim=(-1, -2))


def ablation_mask_from_seq_lengths(
    seq_lengths: TensorType["_batch"],
    max_length: int,
) -> TensorType["_batch", "_sequence"]:
    B = seq_lengths.shape[0]
    mask = torch.zeros((B, max_length), dtype=torch.bool)
    for i in range(B):
        mask[i, : seq_lengths[i]] = True
    return mask

def clean_logits_and_activations(
    model: HookedTransformer,
    location: standard_metrics.Location,
    dataset: TensorType["_batch", "_sequence"],
):
    base_logits, activation_cache = model.run_with_cache(
        dataset,
        names_filter=lambda name: name == standard_metrics.get_model_tensor_name(location),
        return_type="logits",
    )
    return base_logits, activation_cache[standard_metrics.get_model_tensor_name(location)]

def new_bottleneck_test(cfg, layer, device, done_flag):
    torch.autograd.set_grad_enabled(False)

    # Train PCA

    activation_dataset = torch.load(f"activation_data/layer_{layer}/0.pt")
    activation_dataset = activation_dataset.to(device, dtype=torch.float32)

    pca = BatchedPCA(n_dims=activation_dataset.shape[-1], device=device)
    batch_size = 2048

    print("training pca")
    for i in tqdm.trange(0, activation_dataset.shape[0], batch_size):
        j = min(i + batch_size, activation_dataset.shape[0])
        pca.train_batch(activation_dataset[i:j])

    pca_dict = pca.to_rotation_dict(activation_dataset.shape[-1])
    #pca_dict_nz = 

    pca_dict.to_device(device)

    del activation_dataset

    # Load model

    model = HookedTransformer.from_pretrained(cfg.model_name)

    model.to(device)

    ioi_clean_full, ioi_corrupted_full = generate_ioi_dataset(model.tokenizer, cfg.dataset_size, cfg.dataset_size)
    ioi_clean = ioi_clean_full[:, :-1].to(device)
    ioi_corrupted = ioi_corrupted_full[:, :-1].to(device)
    ioi_correct = ioi_clean_full[:, -1].to(device)
    ioi_incorrect = ioi_corrupted_full[:, -1].to(device)

    def divergence_metric(new_logits, base_logits):
        B, L, V = base_logits.shape
        new_logprobs = F.log_softmax(new_logits[:, -1], dim=-1)
        base_logprobs = F.log_softmax(base_logits[:, -1], dim=-1)
        return F.kl_div(new_logprobs, base_logprobs, log_target=True, reduction="batchmean").item()

    def logit_diff(new_logits, base_logits):
        B, L, V = base_logits.shape
        correct = new_logits[:, -1, ioi_correct]
        incorrect = new_logits[:, -1, ioi_incorrect]
        return -(correct - incorrect).mean().item()

    l1_alphas = [1e-3, 3e-4, 1e-4]
    name_fmt = "learned_r{ratio}_{l1_alpha:.0e}"
    best_dicts = {}
    ratios = [4]
    dict_sets = [
        (
            ratio,
            #torch.load(f"/mnt/ssd-cluster/pythia410/tied_residual_l{layer}_r{ratio}/_79/learned_dicts.pt"),
            torch.load(f"/mnt/ssd-cluster/bigrun0308/tied_residual_l{layer}_r{ratio}/_9/learned_dicts.pt")
        )
        for ratio in ratios
    ]

    print("evaluating dicts")
    for l1_alpha in l1_alphas:
        for ratio, dicts in tqdm.tqdm(dict_sets):
            best_approx_dist = float("inf")
            best_dict = None
            for dict, hyperparams in dicts:
                dist = abs(hyperparams["l1_alpha"] - l1_alpha)
                if dist < best_approx_dist:
                    best_approx_dist = dist
                    best_dict = (dict, hyperparams)
            
            best_dicts[name_fmt.format(ratio=ratio, l1_alpha=l1_alpha)] = best_dict

    print("found satisfying dicts:", list(best_dicts.keys()))

    dictionaries = {}
    dictionaries["pca"] = (pca_dict, {"pca": True})
    for name, (dict, hyperparams) in best_dicts.items():
        dictionaries[name] = (dict, hyperparams)

    tau_values = list(np.linspace(0, np.exp(cfg.tau_lin_max), cfg.tau_n_lin)[1:]) + list(np.logspace(cfg.tau_lin_max, cfg.tau_log_max, cfg.tau_n_log)[1:])

    scores: Dict[str, List] = {}

    for name, (dict, _) in dictionaries.items():
        dict.to_device(device)
        print("evaluating", name)

        #active_components = filter_active_components(base_activations, dict, threshold=cfg.activity_threshold)
        scores[name] = acdc_test(
            model,
            dict,
            (layer, "residual"),
            ioi_clean,
            ioi_corrupted,
            logit_metric=divergence_metric,
            thresholds=tau_values,
            distance_metric=scaled_distance_to_clean,
        #    initial_directions=active_components,
        )

    torch.save(scores, f"{cfg.output_dir}/dict_scores_layer_{layer}.pt")
    torch.save(dictionaries, f"{cfg.output_dir}/dictionaries_layer_{layer}.pt")

    done_flag.value = 1

    # torch.save(diff_mean_scores, os.path.join(BASE_FOLDER, f"diff_mean_scores_layer_{layer}.pt"))

def bottleneck_everything_multigpu(cfg):
    layers = [3]
    free_gpus = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"]

    import torch.multiprocessing as mp
    import time
    mp.set_start_method("spawn")

    processes = []

    # while some gpus are still free
    while True:
        new_processes = []
        for process, gpu, done_flag in processes:
            if done_flag.value == 1:
                process.join()
                free_gpus.append(gpu)
                print(f"finished layer {process} on gpu {gpu}")
            else:
                new_processes.append((process, gpu, done_flag))
        
        processes = new_processes

        if len(processes) == 0 and len(layers) == 0:
            break

        if len(free_gpus) == 0:
            time.sleep(0.1)
            continue

        if len(layers) == 0:
            time.sleep(0.1)
            continue
        
        layer = layers.pop(0)
        gpu = free_gpus.pop(0)

        print(f"starting layer {layer} on gpu {gpu}")

        done_flag = mp.Value("i", 0)

        process = mp.Process(
            target=new_bottleneck_test,
            args=(cfg, layer, gpu, done_flag),
        )

        process.start()

        processes.append((process, gpu, done_flag))

if __name__ == "__main__":
    from utils import dotdict

    cfg = dotdict({
        "model_name": "EleutherAI/pythia-70m-deduped",
        "dataset_size": 25,
        "tau_lin_max": -3.5,
        "tau_log_max": -2.5,
        "tau_n_lin": 2,
        "tau_n_log": 10,
        "output_dir": "bottleneck_70m"
    })

    os.makedirs(cfg.output_dir, exist_ok=True)

    bottleneck_everything_multigpu(cfg)