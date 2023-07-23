import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.utils.data as data

import torchopt

from cluster_runs import dispatch_job_on_chunk

from autoencoders.ensemble import FunctionalEnsemble
from autoencoders.sae_ensemble import FunctionalSAE, FunctionalTiedSAE
from autoencoders.semilinear_autoencoder import SemiLinearSAE

from activation_dataset import setup_data
from utils import dotdict, make_tensor_name
from argparser import parse_args

import numpy as np
from itertools import product, chain

from transformer_lens import HookedTransformer, GPT2Tokenizer

import wandb
import datetime
import pickle
import json
import os

import standard_metrics
from autoencoders.learned_dict import LearnedDict, UntiedSAE, TiedSAE

def get_model(cfg):
    if cfg.model_name in ["gpt2", "EleutherAI/pythia-70m-deduped"]:
        model = HookedTransformer.from_pretrained(cfg.model_name, device=cfg.device)
    else:
        raise ValueError("Model name not recognised")

    if hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer
    else:
        print("Using default tokenizer from gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    return model, tokenizer

def init_semilinear_grid(cfg):
    l1_values = list(np.logspace(-7, 0, 16))
    dict_ratios = [2, 4, 8]

    ensembles = []
    ensemble_args = []
    ensemble_tags = []
    devices = [f"cuda:{i}" for i in range(8)]

    for i in range(4):
        cfgs = l1_values[i*4:(i+1)*4]
        models = [
            SemiLinearSAE.init(cfg.activation_width, cfg.activation_width * 8, l1_alpha, dtype=cfg.dtype)
            for l1_alpha in cfgs
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, SemiLinearSAE.loss,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        ensembles.append(ensemble)
        ensemble_args.append({"batch_size": cfg.batch_size, "device": device})
        ensemble_tags.append(f"dict_ratio_8_group_{i}")
    
    for i in range(2):
        cfgs = l1_values[i*8:(i+1)*8]
        models = [
            SemiLinearSAE.init(cfg.activation_width, cfg.activation_width * 4, l1_alpha, dtype=cfg.dtype)
            for l1_alpha in cfgs
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, SemiLinearSAE.loss,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        ensembles.append(ensemble)
        ensemble_args.append({"batch_size": cfg.batch_size, "device": device})
        ensemble_tags.append(f"dict_ratio_4_group_{i}")
    
    for i in range(1):
        cfgs = l1_values
        models = [
            SemiLinearSAE.init(cfg.activation_width, cfg.activation_width * 2, l1_alpha, dtype=cfg.dtype)
            for l1_alpha in cfgs
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, SemiLinearSAE.loss,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        ensembles.append(ensemble)
        ensemble_args.append({"batch_size": cfg.batch_size, "device": device})
        ensemble_tags.append(f"dict_ratio_2_group_{i}")
    
    return ensembles, ensemble_args, ensemble_tags

def init_ensembles_for_mcs_testing(cfg):
    l1_value = 1e-2
    bias_decay = 0.0

    ensembles = []
    ensemble_args = []
    ensemble_tags = []
    devices = [f"cuda:{i}" for i in range(8)]

    for i in range(4):
        models = [
            FunctionalSAE.init(cfg.activation_width, cfg.activation_width * 8, l1_value, bias_decay=bias_decay, dtype=cfg.dtype)
            for _ in range(4)
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalSAE.loss,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        ensembles.append(ensemble)
        ensemble_args.append({"batch_size": cfg.batch_size, "device": device, "tied": False})
        ensemble_tags.append(f"dict_ratio_8_group_{i}")

    for i in range(2):
        models = [
            FunctionalSAE.init(cfg.activation_width, cfg.activation_width * 4, l1_value, bias_decay=bias_decay, dtype=cfg.dtype)
            for _ in range(8)
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalSAE.loss,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        ensembles.append(ensemble)
        ensemble_args.append({"batch_size": cfg.batch_size, "device": device, "tied": False})
        ensemble_tags.append(f"dict_ratio_4_group_{i}")
    
    for i in range(2):
        models = [
            FunctionalSAE.init(cfg.activation_width, cfg.activation_width * 2, l1_value, bias_decay=bias_decay, dtype=cfg.dtype)
            for _ in range(8)
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalSAE.loss,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        ensembles.append(ensemble)
        ensemble_args.append({"batch_size": cfg.batch_size, "device": device, "tied": False})
        ensemble_tags.append(f"dict_ratio_2_group_{i}")

    return ensembles, ensemble_args, ensemble_tags

def init_ensembles_inc_tied(cfg):
    l1_values = list(np.logspace(-3.5, -2, 4))

    print(f"Using l1 values: {l1_values}")

    bias_decays = [0.0, 0.05, 0.1]
    dict_ratios = [2, 4, 8]

    dict_sizes = [cfg.activation_width * ratio for ratio in dict_ratios]

    ensembles = []
    ensemble_args = []
    ensemble_tags = []
    devices = [f"cuda:{i}" for i in range(8)]

    for i in range(2):
        cfgs = product(l1_values[i*2:(i+1)*2], bias_decays)
        models = [
            FunctionalSAE.init(cfg.activation_width, cfg.activation_width * 8, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
            for l1_alpha, bias_decay in cfgs
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalSAE.loss,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        ensembles.append(ensemble)
        ensemble_args.append({"batch_size": cfg.batch_size, "device": device, "tied": False})
        ensemble_tags.append(f"dict_ratio_8_group_{i}")
    
    for i in range(2):
        cfgs = product(l1_values[i*2:(i+1)*2], bias_decays)
        models = [
            FunctionalTiedSAE.init(cfg.activation_width, cfg.activation_width * 8, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
            for l1_alpha, bias_decay in cfgs
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalTiedSAE.loss,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        ensembles.append(ensemble)
        ensemble_args.append({"batch_size": cfg.batch_size, "device": device, "tied": True})
        ensemble_tags.append(f"dict_ratio_8_group_{i}_tied")
    
    for _ in range(1):
        cfgs = product(l1_values, bias_decays)
        models = [
            FunctionalSAE.init(cfg.activation_width, cfg.activation_width * 4, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
            for l1_alpha, bias_decay in cfgs
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalSAE.loss,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        ensembles.append(ensemble)
        ensemble_args.append({"batch_size": cfg.batch_size, "device": device, "tied": False})
        ensemble_tags.append(f"dict_ratio_4")
    
    for _ in range(1):
        cfgs = product(l1_values, bias_decays)
        models = [
            FunctionalTiedSAE.init(cfg.activation_width, cfg.activation_width * 4, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
            for l1_alpha, bias_decay in cfgs
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalTiedSAE.loss,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        ensembles.append(ensemble)
        ensemble_args.append({"batch_size": cfg.batch_size, "device": device, "tied": True})
        ensemble_tags.append(f"dict_ratio_4_tied")
    
    for _ in range(1):
        cfgs = product(l1_values, bias_decays)
        models = [
            FunctionalSAE.init(cfg.activation_width, cfg.activation_width * 2, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
            for l1_alpha, bias_decay in cfgs
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalSAE.loss,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        ensembles.append(ensemble)
        ensemble_args.append({"batch_size": cfg.batch_size, "device": device, "tied": False})
        ensemble_tags.append(f"dict_ratio_2")
    
    for _ in range(1):
        cfgs = product(l1_values, bias_decays)
        models = [
            FunctionalTiedSAE.init(cfg.activation_width, cfg.activation_width * 2, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
            for l1_alpha, bias_decay in cfgs
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalTiedSAE.loss,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        ensembles.append(ensemble)
        ensemble_args.append({"batch_size": cfg.batch_size, "device": device, "tied": True})
        ensemble_tags.append(f"dict_ratio_2_tied")
    
    return ensembles, ensemble_args, ensemble_tags, (l1_values, bias_decays, dict_sizes)

def calc_expected_interference(dictionary, batch):
    # dictionary: [n_dict_components, d_activation]
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

# weird asymmetric kurtosis/skew with center at 0
def calc_feature_skew(batch):
    # batch: [batch_size, n_features]
    variance = torch.var(batch, dim=0)
    asymm_skew = torch.mean(batch**3, dim=0) / torch.clamp(variance**1.5, min=1e-8)

    return asymm_skew

def calc_feature_kurtosis(batch):
    # batch: [batch_size, n_features]
    variance = torch.var(batch, dim=0)
    asymm_kurtosis = torch.mean(batch**4, dim=0) / torch.clamp(variance**2, min=1e-8)

    return asymm_kurtosis

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

def log_standard_metrics(learned_dicts, chunk, hyperparam_settings, l1_values, dict_sizes, cfg):
    n_samples = 2000
    sample_indexes = np.random.choice(len(chunk), size=n_samples, replace=False)
    sample = chunk[sample_indexes]

    small_dict_size = dict_sizes[0]

    mmcs_grid_plots = {}

    for setting in hyperparam_settings:
        mmcs_scores = np.zeros((len(l1_values), len(dict_sizes)))

        for i, l1_value in enumerate(l1_values):
            small_dict_setting_ = setting.copy()
            small_dict_setting_["l1_alpha"] = l1_value
            small_dict_setting_["n_dict_components"] = small_dict_size

            small_dict = filter_learned_dicts(learned_dicts, small_dict_setting_)[0][0]

            for j, dict_size in enumerate(dict_sizes[1:]):
                setting_ = setting.copy()
                setting_["l1_alpha"] = l1_value
                setting_["n_dict_components"] = dict_size

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
    
    for k, plot in mmcs_grid_plots.items():
        cfg.wandb_instance.log({f"mmcs_grid/{k}": wandb.Image(plot)}, commit=False)
    
    for k, plot in sparsity_hists.items():
        cfg.wandb_instance.log({f"sparsity_hist/{k}": wandb.Image(plot)})

def ensemble_train_loop(ensemble, cfg, args, name, sampler, dataset, progress_counter):
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    np.random.seed(0)

    if cfg.use_wandb:
        run = cfg.wandb_instance

    if args["tied"] == True:
        dict_name = "encoder"
    else:
        dict_name = "decoder"

    for i, batch_idxs in enumerate(sampler):
        batch = dataset[batch_idxs].to(args["device"])
        losses, aux_buffer = ensemble.step_batch(batch)

        expected_interferences = torch.vmap(calc_expected_interference)(ensemble.params[dict_name], aux_buffer["c"])
        mean_expected_interference = expected_interferences.mean(dim=-1)

        feature_skews = torch.vmap(calc_feature_skew)(aux_buffer["c"])
        mean_feature_skew = feature_skews.mean(dim=-1)

        feature_kurtosis = torch.vmap(calc_feature_kurtosis)(aux_buffer["c"])
        mean_feature_kurtosis = feature_kurtosis.mean(dim=-1)

        num_nonzero = aux_buffer["c"].count_nonzero(dim=-1).float().mean(dim=-1)

        if cfg.use_wandb:
            log = {}
            for m in range(ensemble.n_models):
                name = make_hyperparam_name({
                    "tied": args["tied"],
                    "l1_alpha": ensemble.buffers["l1_alpha"][m].item(),
                    "bias_decay": ensemble.buffers["bias_decay"][m].item(),
                    "dict_size": ensemble.params[dict_name][m].shape[0]
                })
                log[f"{name}_loss"] = losses["loss"][m].item()
                log[f"{name}_l_l1"] = losses["l_l1"][m].item()
                log[f"{name}_l_reconstruction"] = losses["l_reconstruction"][m].item()
                log[f"{name}_l_bias_decay"] = losses["l_bias_decay"][m].item()
                log[f"{name}_sparsity"] = num_nonzero[m].item()
                log[f"{name}_mean_expected_interference"] = mean_expected_interference[m].item()
                #log[f"{name}_mean_feature_skew"] = mean_feature_skew[m].item()
                log[f"{name}_mean_feature_kurtosis"] = mean_feature_kurtosis[m].item()

            run.log(log, commit=True)

        progress_counter.value = i

def unstacked_to_learned_dicts(ensemble, args, hyperparams, tag_with_n_feats=True):
    unstacked = ensemble.unstack(device="cpu")
    learned_dicts = []
    for model in unstacked:
        hyperparam_values = {}

        params, buffers = model

        for hp in hyperparams:
            if hp in args:
                hyperparam_values[hp] = args[hp]
            elif hp in buffers:
                hyperparam_values[hp] = buffers[hp].item()
            else:
                raise ValueError(f"Hyperparameter {hp} not found in args or model buffers")

        decoder_name = "decoder" if args["tied"] == False else "encoder"

        if tag_with_n_feats:
            hyperparam_values["n_dict_components"] = params[decoder_name].shape[0]

        if args["tied"] == True:
            learned_dict = TiedSAE(params["encoder"], params["encoder_bias"])
        else:
            learned_dict = UntiedSAE(params["encoder"], params["decoder"], params["encoder_bias"])
        
        learned_dicts.append((learned_dict, hyperparam_values))
    return learned_dicts

def main():
    torch.set_grad_enabled(False)
    mp.set_start_method("spawn", force=True)
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = parse_args()

    cfg.model_name = "EleutherAI/pythia-70m-deduped"
    cfg.dataset_folder = "activation_data"
    cfg.output_folder = "output"

    cfg.batch_size = 1024
    cfg.lr = 3e-4

    cfg.use_wandb = True

    cfg.dtype = torch.float32

    cfg.layer = 2
    cfg.use_residual = True

    if cfg.use_residual:
        cfg.activation_width = 512
    else:
        cfg.activation_width = 2048 # mlp_width is 4x the residual width

    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if cfg.use_wandb:
        secrets = json.load(open("secrets.json"))
        wandb.login(key=secrets["wandb_key"])
        wandb_run_name = f"ensemble_{cfg.model_name}_{start_time[4:]}"  # trim year
        cfg.wandb_instance = wandb.init(project="sparse coding", config=dict(cfg), name=wandb_run_name, entity="sparse_coding")

    os.makedirs(cfg.dataset_folder, exist_ok=True)

    if len(os.listdir(cfg.dataset_folder)) == 0:
        print(f"Activations in {cfg.dataset_folder} do not exist, creating them")
        transformer, tokenizer = get_model(cfg)
        setup_data(cfg, tokenizer, transformer)
        del transformer, tokenizer
    else:
        print(f"Activations in {cfg.dataset_folder} already exist, loading them")

    dataset = torch.load(os.path.join(cfg.dataset_folder, "0.pt"))
    n_lines = cfg.max_lines
    del dataset

    print("Initialising ensembles...", end=" ")

    ensembles, args, tags, hyperparam_ranges = init_ensembles_inc_tied(cfg)
    l1_values, bias_decays, dict_sizes = hyperparam_ranges

    print("Ensembles initialised.")

    n_chunks = len(os.listdir(cfg.dataset_folder))
    chunk_order = np.random.permutation(n_chunks)

    for i, chunk_idx in enumerate(chunk_order):
        print(f"Chunk {i+1}/{n_chunks}")

        cfg.iter_folder = os.path.join(cfg.output_folder, f"_{i}")
        os.makedirs(cfg.iter_folder, exist_ok=True)

        chunk_loc = os.path.join(cfg.dataset_folder, f"{chunk_idx}.pt")
        chunk = torch.load(chunk_loc).to(device="cpu", dtype=torch.float32)

        dispatch_job_on_chunk(
            ensembles, cfg, args, tags, chunk, ensemble_train_loop
        )

        hyperparams = ["l1_alpha", "bias_decay", "tied"]
        
        learned_dicts = []
        for ensemble, arg in zip(ensembles, args):
            learned_dicts.extend(unstacked_to_learned_dicts(ensemble, arg, hyperparams, tag_with_n_feats=True))
        
        hyperparam_settings = []
        for tied_val in [True, False]:
            for bias_decay in bias_decays:
                hyperparam_settings.append({"tied": tied_val, "bias_decay": bias_decay})

        log_standard_metrics(learned_dicts, chunk, hyperparam_settings, l1_values, dict_sizes, cfg)

        del chunk

        torch.save(learned_dicts, os.path.join(cfg.iter_folder, "learned_dicts.pt"))

if __name__ == "__main__":
    main()