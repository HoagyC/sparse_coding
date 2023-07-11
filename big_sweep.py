import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.utils.data as data

import torchopt

from cluster_runs import dispatch_on_chunk

from autoencoders.ensemble import FunctionalEnsemble
from autoencoders.sae_ensemble import FunctionalSAE

from activation_dataset import setup_data
from utils import dotdict, make_tensor_name
from argparser import parse_args

import numpy as np
from itertools import product, chain

from transformer_lens import HookedTransformer

import pickle

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

def init_ensembles(cfg):
    l1_values = list(np.logspace(-4.5, -1, 8))
    bias_decays = [0.0, 0.01, 0.1]
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

def main():
    torch.set_grad_enabled(False)
    mp.set_start_method("spawn", force=True)
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = parse_args()

    cfg.model_name = "EleutherAI/pythia-70m-deduped"
    cfg.dataset_folder = "activation_data"

    cfg.batch_size = 1024
    cfg.lr = 3e-4

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

    ensembles, args, tags = init_ensembles(cfg)

    print("Ensembles initialised.")

    n_chunks = len(os.listdir(cfg.dataset_folder))
    chunk_order = np.random.permutation(n_chunks)

    for i, chunk_idx in enumerate(chunk_order):
        print(f"Chunk {i+1}/{n_chunks}")

        chunk_loc = os.path.join(cfg.dataset_folder, f"{chunk_idx}.pt")
        chunk = torch.load(chunk_loc).to(dtype=torch.float32)

        outputs = dispatch_on_chunk(ensembles, args, tags, chunk)

        sum_losses = [0 for _ in range(10)]
        for j in range(len(ensembles)):
            for k in range(10):
                sum_losses[k] += outputs[j][k]["l_l1"].sum().item()

        print("Losses over time:", sum_losses)
        print(f"Chunk {i+1}/{n_chunks} done, saving")

        iter_folder = os.path.join(cfg.output_folder, f"{i}")
        os.makedirs(iter_folder, exist_ok=True)
        
        for i in range(len(ensembles)):
            torch.save(ensembles[i].state_dict()["params"], os.path.join(iter_folder, f"ensemble_{ensemble_tags[i]}.pt"))
        
        torch.save(logger_data, os.path.join(iter_folder, "logger_data.pt"))

if __name__ == "__main__":
    main()