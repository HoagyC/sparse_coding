import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.utils.data as data

import torchopt

from cluster_runs import dispatch_on_chunk, dispatch_job_on_chunk

from autoencoders.ensemble import FunctionalEnsemble
from autoencoders.sae_ensemble import FunctionalSAE, FunctionalTiedSAE
from autoencoders.semilinear_autoencoder import SemiLinearSAE

from activation_dataset import setup_data
from utils import dotdict, make_tensor_name
from argparser import parse_args

import numpy as np
from itertools import product, chain

from transformer_lens import HookedTransformer

import wandb
import datetime
import pickle
import json
import os

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
            SemiLinearSAE.init(cfg.mlp_width, cfg.mlp_width * 8, l1_alpha, dtype=cfg.dtype)
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
            SemiLinearSAE.init(cfg.mlp_width, cfg.mlp_width * 4, l1_alpha, dtype=cfg.dtype)
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
            SemiLinearSAE.init(cfg.mlp_width, cfg.mlp_width * 2, l1_alpha, dtype=cfg.dtype)
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

def init_ensembles_inc_tied(cfg):
    l1_values = list(np.logspace(-3.5, -2, 4))

    print(f"Using l1 values: {l1_values}")

    #l1_values = [0.001, 0.01, 0.1]
    bias_decays = [0.0, 0.05, 0.1]
    dict_ratios = [2, 4, 8]

    ensembles = []
    ensemble_args = []
    ensemble_tags = []
    devices = [f"cuda:{i}" for i in range(8)]

    for i in range(2):
        cfgs = product(l1_values[i*2:(i+1)*2], bias_decays)
        models = [
            FunctionalSAE.init(cfg.mlp_width, cfg.mlp_width * 8, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
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
        ensemble_args.append({"batch_size": cfg.batch_size, "device": device})
        ensemble_tags.append(f"dict_ratio_8_group_{i}")
    
    for i in range(2):
        cfgs = product(l1_values[i*2:(i+1)*2], bias_decays)
        models = [
            FunctionalTiedSAE.init(cfg.mlp_width, cfg.mlp_width * 8, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
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
        ensemble_args.append({"batch_size": cfg.batch_size, "device": device})
        ensemble_tags.append(f"dict_ratio_8_group_{i}_tied")
    
    for _ in range(1):
        cfgs = product(l1_values, bias_decays)
        models = [
            FunctionalSAE.init(cfg.mlp_width, cfg.mlp_width * 4, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
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
        ensemble_args.append({"batch_size": cfg.batch_size, "device": device})
        ensemble_tags.append(f"dict_ratio_4")
    
    for _ in range(1):
        cfgs = product(l1_values, bias_decays)
        models = [
            FunctionalTiedSAE.init(cfg.mlp_width, cfg.mlp_width * 4, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
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
        ensemble_args.append({"batch_size": cfg.batch_size, "device": device})
        ensemble_tags.append(f"dict_ratio_4_tied")
    
    for _ in range(1):
        cfgs = product(l1_values, bias_decays)
        models = [
            FunctionalSAE.init(cfg.mlp_width, cfg.mlp_width * 2, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
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
        ensemble_args.append({"batch_size": cfg.batch_size, "device": device})
        ensemble_tags.append(f"dict_ratio_2")
    
    for _ in range(1):
        cfgs = product(l1_values, bias_decays)
        models = [
            FunctionalTiedSAE.init(cfg.mlp_width, cfg.mlp_width * 2, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
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
        ensemble_args.append({"batch_size": cfg.batch_size, "device": device})
        ensemble_tags.append(f"dict_ratio_2_tied")
    
    return ensembles, ensemble_args, ensemble_tags

def init_ensembles_grid(cfg):
    l1_values = list(np.logspace(-5.5, -2, 8))
    bias_decays = [0.0]
    dict_ratios = [2, 4, 8, 16]

    # total 8x3x4 = 96 runs
    # split across 8 GPUs as follows:
    # 16 split over 4 GPUs
    # 8 split over 2 GPUs
    # 2 and 4 split over 1 GPU each

    ensembles = []
    ensemble_args = []
    ensemble_tags = []
    devices = [f"cuda:{i}" for i in range(8)]

    for i in range(4):
        cfgs = product(l1_values[i*2:(i+1)*2], bias_decays)
        models = [
            FunctionalSAE.init(cfg.mlp_width, cfg.mlp_width * 16, l1_alpha, bias_decay=bias_decay)
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
        ensemble_args.append({"batch_size": cfg.batch_size, "device": device})
        ensemble_tags.append(f"16_{i}")
    
    for i in range(2):
        cfgs = product(l1_values[i*4:(i+1)*4], bias_decays)
        models = [
            FunctionalSAE.init(cfg.mlp_width, cfg.mlp_width * 8, l1_alpha, bias_decay=bias_decay)
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
        ensemble_args.append({"batch_size": cfg.batch_size, "device": device})
        ensemble_tags.append(f"8_{i}")

    cfgs = product(l1_values, bias_decays)
    models = [
        FunctionalSAE.init(cfg.mlp_width, cfg.mlp_width * 4, l1_alpha, bias_decay=bias_decay)
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
    ensemble_args.append({"batch_size": cfg.batch_size, "device": device})
    ensemble_tags.append("4")

    cfgs = product(l1_values, bias_decays)
    models = [
        FunctionalSAE.init(cfg.mlp_width, cfg.mlp_width * 2, l1_alpha, bias_decay=bias_decay)
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
    ensemble_args.append({"batch_size": cfg.batch_size, "device": device})
    ensemble_tags.append("2")

    return ensembles, ensemble_args, ensemble_tags

def dead_features_logger(ensemble, n_batch, logger_data, losses, aux_buffer):
    # c: np.array(n_models, n_logs, batch_size, n_features)
    c = np.stack([aux["c"] for aux in aux_buffer], axis=1)
    # count_nonzero: np.array(n_models, n_features)
    count_nonzero = (c != 0).sum(axis=(1, 2))
    # mean: np.array(n_models, n_features)
    mean = c.mean(axis=(1, 2))
    # mean_nonzero: np.array(n_models, n_features)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_nonzero = c.sum(axis=(1, 2)) / count_nonzero
        mean_nonzero[np.isnan(mean_nonzero)] = 0
    return {"mean": mean, "mean_nonzero": mean_nonzero, "count_nonzero": count_nonzero}

def ensemble_train_loop(ensemble, cfg, args, name, sampler, dataset, progress_counter):
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
            log = dict(chain(*[[
                (f"{name}_{m}_loss", losses["loss"][m].item()),
                (f"{name}_{m}_l_l1", losses["l_l1"][m].item()),
                (f"{name}_{m}_l_reconstruction", losses["l_reconstruction"][m].item()),
                (f"{name}_{m}_l_bias_decay", losses["l_bias_decay"][m].item()),
                (f"{name}_{m}_sparsity", num_nonzero[m].item()),
            ] for m in range(ensemble.n_models)]))
            
            run.log(log, commit=True)

        progress_counter.value = i

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
    #print(dataset.dtype) # torch.float16
    cfg.mlp_width = dataset.shape[-1]
    n_lines = cfg.max_lines
    del dataset

    print("Initialising ensembles...", end=" ")

    ensembles, args, tags = init_ensembles_inc_tied(cfg)

    print("Ensembles initialised.")

    n_chunks = len(os.listdir(cfg.dataset_folder))
    chunk_order = np.random.permutation(n_chunks)

    for i, chunk_idx in enumerate(chunk_order):
        print(f"Chunk {i+1}/{n_chunks}")

        cfg.iter_folder = os.path.join(cfg.output_folder, f"_{i}")
        os.makedirs(cfg.iter_folder, exist_ok=True)

        chunk_loc = os.path.join(cfg.dataset_folder, f"{chunk_idx}.pt")
        chunk = torch.load(chunk_loc).to(device="cpu", dtype=torch.float32) #.to(dtype=torch.float32)

        #outputs = dispatch_on_chunk(ensembles, args, tags, chunk, logger=dead_features_logger, interval=9)
        dispatch_job_on_chunk(
            ensembles, cfg, args, tags, chunk, ensemble_train_loop
        )

        del chunk

        hyperparams = ["l1_alpha", "bias_decay"]
        hyperparam_map = {}

        for ensemble, tag in zip(ensembles, tags):
            for idx, model in enumerate(ensemble.unstack(device="cpu")):
                name = f"{tag}_{idx}.pt"
                _, buffers = model
                hyperparam_map[name] = {
                    k: buffers[k].item() for k in hyperparams
                }
                print("Saving", name)
                torch.save(model, os.path.join(cfg.iter_folder, name))
        
        with open(os.path.join(cfg.iter_folder, "hyperparams.json"), "w") as f:
            json.dump(hyperparam_map, f)

if __name__ == "__main__":
    main()