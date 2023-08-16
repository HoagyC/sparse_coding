import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.utils.data as data

from utils import check_transformerlens_model, get_activation_size

from cluster_runs import dispatch_job_on_chunk
from activation_dataset import setup_data
from sc_datasets.random_dataset import SparseMixDataset

import numpy as np
from itertools import product, chain

from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer

import tqdm

import wandb
import datetime
import pickle
import json
import os
import sys

import standard_metrics
from autoencoders.learned_dict import LearnedDict, UntiedSAE, TiedSAE

def get_model(cfg):
    if check_transformerlens_model(cfg.model_name):
        model = HookedTransformer.from_pretrained(cfg.model_name, device=cfg.device)
    else:
        raise ValueError("Model name not recognised")

    if hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer
    else:
        print("Using default tokenizer from gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    return model, tokenizer

def calc_expected_interference(dictionary, batch):
    # dictionary: [n_features, d_activation]
    # batch: [batch_size, n_features]
    norms = torch.norm(dictionary, 2, dim=-1)
    normed_weights = dictionary / torch.clamp(norms, 1e-8)[:, None]

    cosines = torch.einsum("ij,kj->ik", normed_weights, normed_weights)
    totals = torch.einsum("ij,bj->bi", cosines ** 2, batch)

    capacities = batch / torch.clamp(totals, min=1e-8)
    # nonzero_count: [n_features]
    nonzero_count = batch.count_nonzero(dim=0).float()
    # nonzero_capacity: [n_features]
    nonzero_capacity = capacities.sum(dim=0) / torch.clamp(nonzero_count, min=1.0)
    return nonzero_capacity


def filter_learned_dicts(learned_dicts, hyperparam_filters):
    from math import isclose

    filtered_learned_dicts = []
    for learned_dict, hyperparams in learned_dicts:
        if all([isclose(hyperparams[hp], val, rel_tol=1e-3) if isinstance(val, float) else hyperparams[hp] == val for hp, val in hyperparam_filters.items()]):
            filtered_learned_dicts.append((learned_dict, hyperparams))
    return filtered_learned_dicts

def format_hyperparam_val(val):
    if isinstance(val, float):
        return f"{val:.2E}".replace("+", "")
    else:
        return str(val)

def make_hyperparam_name(setting):
    return "_".join([f"{k}_{format_hyperparam_val(v)}" for k, v in setting.items()])

def log_standard_metrics(learned_dicts, chunk, chunk_num, hyperparam_ranges, cfg):
    n_samples = 2000
    sample_indexes = np.random.choice(len(chunk), size=n_samples, replace=False)
    sample = chunk[sample_indexes]


    grid_hyperparams = [k for k in hyperparam_ranges.keys() if k not in ["l1_alpha", "dict_size"]]
    mmcs_plot_settings = []
    for setting in product(*[hyperparam_ranges[hp] for hp in grid_hyperparams]):
        mmcs_plot_settings.append({hp: val for hp, val in zip(grid_hyperparams, setting)})
    
    l1_values = hyperparam_ranges["l1_alpha"]
    dict_sizes = hyperparam_ranges["dict_size"]

    if len(dict_sizes) > 1:
        small_dict_size = dict_sizes[0]

        mmcs_grid_plots = {}

        for setting in mmcs_plot_settings:
            mmcs_scores = np.zeros((len(l1_values), len(dict_sizes)))

            for i, l1_value in enumerate(l1_values):
                small_dict_setting_ = setting.copy()
                small_dict_setting_["l1_alpha"] = l1_value
                small_dict_setting_["dict_size"] = small_dict_size

                small_dict = filter_learned_dicts(learned_dicts, small_dict_setting_)[0][0]

                for j, dict_size in enumerate(dict_sizes[1:]):
                    setting_ = setting.copy()
                    setting_["l1_alpha"] = l1_value
                    setting_["dict_size"] = dict_size

                    larger_dict = filter_learned_dicts(learned_dicts, setting_)[0][0]
                    mmcs_scores[i, j] = standard_metrics.mcs_duplicates(small_dict, larger_dict).mean().item()
            
            mmcs_grid_plots[make_hyperparam_name(setting)] = standard_metrics.plot_grid(
                mmcs_scores,
                l1_values, dict_sizes[1:],
                "l1_alpha", "dict_size",
                cmap="viridis"
            )
    
    sparsity_hists = {}

    for learned_dict, setting in learned_dicts:
        sparsity_hists[make_hyperparam_name(setting)] = standard_metrics.plot_hist(
            standard_metrics.mean_nonzero_activations(learned_dict, sample),
            "Mean nonzero activations",
            "Frequency",
            bins=20
        )
    
    if cfg.use_wandb:
        if len(dict_sizes) > 1:
            for k, plot in mmcs_grid_plots.items():
                cfg.wandb_instance.log({f"mmcs_grid_{chunk_num}/{k}": wandb.Image(plot)}, commit=False)
        
        for k, plot in sparsity_hists.items():
            cfg.wandb_instance.log({f"sparsity_hist_{chunk_num}/{k}": wandb.Image(plot)})

def ensemble_train_loop(ensemble, cfg, args, ensemble_name, sampler, dataset, progress_counter):
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    np.random.seed(0)

    if cfg.use_wandb:
        run = cfg.wandb_instance

    for i, batch_idxs in enumerate(sampler):
        batch = dataset[batch_idxs].to(args["device"])
        losses, aux_buffer = ensemble.step_batch(batch)

        num_nonzero = aux_buffer["c"].count_nonzero(dim=-1).float().mean(dim=-1)

        if cfg.use_wandb:
            log = {}
            for m in range(ensemble.n_models):
                hyperparam_values = {}

                for ep in cfg.ensemble_hyperparams:
                    if ep in args:
                        hyperparam_values[ep] = args[ep]
                    else:
                        raise ValueError(f"Hyperparameter {ep} not found in args")
                    
                for bp in cfg.buffer_hyperparams:
                    if bp in ensemble.buffers:
                        hyperparam_values[bp] = ensemble.buffers[bp][m].item()
                    else:
                        raise ValueError(f"Hyperparameter {bp} not found in buffers")

                name = make_hyperparam_name(hyperparam_values)

                for k in losses.keys():
                    log[f"{ensemble_name}_{name}_{k}"] = losses[k][m].item()
                
                log[f"{ensemble_name}_{name}_num_nonzero"] = num_nonzero[m].item()

            run.log(log, commit=True)

        progress_counter.value = i

def unstacked_to_learned_dicts(ensemble, args, ensemble_hyperparams, buffer_hyperparams):
    unstacked = ensemble.unstack(device="cpu")
    learned_dicts = []
    for model in unstacked:
        hyperparam_values = {}

        params, buffers = model

        for ep in ensemble_hyperparams:
            if ep in args:
                hyperparam_values[ep] = args[ep]
            else:
                raise ValueError(f"Hyperparameter {ep} not found in args")
            
        for bp in buffer_hyperparams:
            if bp in buffers:
                hyperparam_values[bp] = buffers[bp].item()
            else:
                raise ValueError(f"Hyperparameter {bp} not found in buffers")

        learned_dict = ensemble.sig.to_learned_dict(params, buffers)
        
        learned_dicts.append((learned_dict, hyperparam_values))
    return learned_dicts

def generate_synthetic_dataset(cfg, generator, chunk_size, n_chunks):
    batch_size = generator.batch_size
    n_samples = chunk_size // batch_size

    for i in range(n_chunks):
        print(f"Generating chunk {i+1}/{n_chunks}")
        chunk = torch.zeros((chunk_size, cfg.activation_width), dtype=torch.float32, device="cpu")
        for j in tqdm.tqdm(range(n_samples)):
            chunk[j*batch_size:(j+1)*batch_size] = generator.send(None).cpu()
        torch.save(chunk, os.path.join(cfg.dataset_folder, f"{i}.pt"))

def init_model_dataset(cfg):
    cfg.activation_width = get_activation_size(cfg.model_name, cfg.layer_loc)

    if len(os.listdir(cfg.dataset_folder)) == 0:
        print(f"Activations in {cfg.dataset_folder} do not exist, creating them")
        transformer, tokenizer = get_model(cfg)
        setup_data(
            tokenizer,
            transformer,
            dataset_name=cfg.dataset_name,
            dataset_folder=cfg.dataset_folder,
            layer=cfg.layer,
            layer_loc=cfg.layer_loc,
            n_chunks=cfg.n_chunks,
            device=cfg.device
        )
        del transformer, tokenizer
    else:
        print(f"Activations in {cfg.dataset_folder} already exist, loading them")

def init_synthetic_dataset(cfg):
    if len(os.listdir(cfg.dataset_folder)) == 0:
        print(f"Activations in {cfg.dataset_folder} do not exist, creating them")
        generator = SparseMixDataset(
            cfg.activation_width,
            cfg.n_ground_truth_components,
            cfg.gen_batch_size,
            cfg.feature_num_nonzero,
            cfg.feature_prob_decay,
            cfg.noise_magnitude_scale,
            "cuda:0",
            sparse_component_covariance = None if cfg.correlated_components else torch.eye(cfg.n_ground_truth_components, device="cuda:0"),
            t_type=torch.float16
        )

        print("generated dataset")

        chunk_size = cfg.chunk_size_gb * 1024**3
        chunk_activations = chunk_size // (cfg.activation_width * 2)
        generate_synthetic_dataset(cfg, generator, chunk_activations, cfg.n_chunks)

        # save the generator for later
        torch.save(generator, os.path.join(cfg.output_folder, "generator.pt"))
    else:
        print(f"Activations in {cfg.dataset_folder} already exist, loading them")

def sweep(ensemble_init_func, cfg):
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter('ignore')
    
    torch.set_grad_enabled(False)
    with torch.no_grad():
        torch.cuda.empty_cache()
    mp.set_start_method("spawn", force=True)
    torch.manual_seed(0)
    np.random.seed(0)

    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(cfg.dataset_folder, exist_ok=True)
    os.makedirs(cfg.output_folder, exist_ok=True)

    if cfg.use_wandb:
        secrets = json.load(open("secrets.json"))
        wandb.login(key=secrets["wandb_key"])
        wandb_run_name = f"ensemble_{cfg.model_name}_{start_time[4:]}"  # trim year
        cfg.wandb_instance = wandb.init(project="sparse coding", config=dict(cfg), name=wandb_run_name, entity="sparse_coding")

    if cfg.use_synthetic_dataset:
        init_synthetic_dataset(cfg)
    else:
        init_model_dataset(cfg)

    print("Initialising ensembles...", end=" ")

    # the ensemble initialization function returns
    # a list of (ensemble, args, name) tuples
    # and a dict of hyperparam ranges
    ensembles, ensemble_hyperparams, buffer_hyperparams, hyperparam_ranges = ensemble_init_func(cfg)

    # ensemble_hyperparams are constant across all models in a given ensemble
    # they are stored in the ensemble's args
    # buffer_hyperparams can vary between models in an ensemble
    # they are stored in each model's buffer and have to be torch tensors
    cfg.ensemble_hyperparams = ensemble_hyperparams
    cfg.buffer_hyperparams = buffer_hyperparams

    print("Ensembles initialised.")

    n_chunks = len(os.listdir(cfg.dataset_folder))

    chunk_order = np.random.permutation(n_chunks)

    if cfg.n_repetitions is not None:
        chunk_order = np.tile(chunk_order, cfg.n_repetitions)

    for i, chunk_idx in enumerate(chunk_order):
        print(f"Chunk {i+1}/{n_chunks}")

        cfg.iter_folder = os.path.join(cfg.output_folder, f"_{i}")
        os.makedirs(cfg.iter_folder, exist_ok=True)

        chunk_loc = os.path.join(cfg.dataset_folder, f"{chunk_idx}.pt")
        chunk = torch.load(chunk_loc).to(device="cpu", dtype=torch.float32)

        dispatch_job_on_chunk(
            ensembles, cfg, chunk, ensemble_train_loop
        )

        learned_dicts = []
        for ensemble, arg, _ in ensembles:
            learned_dicts.extend(unstacked_to_learned_dicts(ensemble, arg, cfg.ensemble_hyperparams, cfg.buffer_hyperparams))

        if cfg.wandb_images:
            log_standard_metrics(learned_dicts, chunk, i, hyperparam_ranges, cfg)

        del chunk
        if i == n_chunks - 1 or i % cfg.save_every == 0:
            torch.save(learned_dicts, os.path.join(cfg.iter_folder, "learned_dicts.pt"))

        print("\n")