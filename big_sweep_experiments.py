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

from transformer_lens import HookedTransformer

import wandb
import datetime
import pickle
import json
import os

import standard_metrics
from autoencoders.learned_dict import LearnedDict, UntiedSAE, TiedSAE

from big_sweep import sweep

def tied_vs_not_experiment(cfg):
    l1_values = list(np.logspace(-3.5, -2, 4))

    bias_decays = [0.0, 0.05, 0.1]
    dict_ratios = [2, 4, 8]

    dict_sizes = [cfg.mlp_width * ratio for ratio in dict_ratios]

    ensembles = []
    devices = [f"cuda:{i}" for i in range(8)]

    for i in range(2):
        cfgs = product(l1_values[i*2:(i+1)*2], bias_decays)
        models = [
            FunctionalSAE.init(cfg.mlp_width, cfg.mlp_width * 8, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
            for l1_alpha, bias_decay in cfgs
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalSAE,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        args = {"batch_size": cfg.batch_size, "device": device, "tied": False, "dict_size": cfg.mlp_width * 8}
        name = f"dict_ratio_8_group_{i}"
        
        ensembles.append((ensemble, args, name))
    
    for i in range(2):
        cfgs = product(l1_values[i*2:(i+1)*2], bias_decays)
        models = [
            FunctionalTiedSAE.init(cfg.mlp_width, cfg.mlp_width * 8, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
            for l1_alpha, bias_decay in cfgs
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalTiedSAE,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        args = {"batch_size": cfg.batch_size, "device": device, "tied": True, "dict_size": cfg.mlp_width * 8}
        name = f"dict_ratio_8_group_{i}_tied"

        ensembles.append((ensemble, args, name))

    for _ in range(1):
        cfgs = product(l1_values, bias_decays)
        models = [
            FunctionalSAE.init(cfg.mlp_width, cfg.mlp_width * 4, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
            for l1_alpha, bias_decay in cfgs
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalSAE,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        args = {"batch_size": cfg.batch_size, "device": device, "tied": False, "dict_size": cfg.mlp_width * 4}
        name = f"dict_ratio_4"

        ensembles.append((ensemble, args, name))
    
    for _ in range(1):
        cfgs = product(l1_values, bias_decays)
        models = [
            FunctionalTiedSAE.init(cfg.mlp_width, cfg.mlp_width * 4, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
            for l1_alpha, bias_decay in cfgs
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalTiedSAE,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        args = {"batch_size": cfg.batch_size, "device": device, "tied": True, "dict_size": cfg.mlp_width * 4}
        name = f"dict_ratio_4_tied"

        ensembles.append((ensemble, args, name))

    for _ in range(1):
        cfgs = product(l1_values, bias_decays)
        models = [
            FunctionalSAE.init(cfg.mlp_width, cfg.mlp_width * 2, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
            for l1_alpha, bias_decay in cfgs
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalSAE,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        args = {"batch_size": cfg.batch_size, "device": device, "tied": False, "dict_size": cfg.mlp_width * 2}
        name = f"dict_ratio_2"

        ensembles.append((ensemble, args, name))

    for _ in range(1):
        cfgs = product(l1_values, bias_decays)
        models = [
            FunctionalTiedSAE.init(cfg.mlp_width, cfg.mlp_width * 2, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
            for l1_alpha, bias_decay in cfgs
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalTiedSAE,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        args = {"batch_size": cfg.batch_size, "device": device, "tied": True, "dict_size": cfg.mlp_width * 2}
        name = f"dict_ratio_2_tied"

        ensembles.append((ensemble, args, name))
    
    return (ensembles,
        ["tied", "dict_size"], ["l1_alpha", "bias_decay"],
        {"tied": [True, False], "dict_size": dict_sizes, "l1_alpha": l1_values, "bias_decay": bias_decays})

if __name__ == "__main__":
    cfg = parse_args()
    sweep(tied_vs_not_experiment, cfg)