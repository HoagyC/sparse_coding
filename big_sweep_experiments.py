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

# an example function that builds a list of ensembles to run
# you could this as a template for other experiments

# it returns:
# - a list of (ensemble, args, name) tuples,
# - a list of hyperparameters that vary between ensembles
# - a list of hyperparameters that vary between models in the same ensemble
# - a dict of hyperparameter ranges
def tied_vs_not_experiment(cfg):
    l1_values = list(np.logspace(-3.5, -2, 4))

    bias_decays = [0.0, 0.05, 0.1]
    dict_ratios = [2, 4, 8]

    dict_sizes = [cfg.activation_width * ratio for ratio in dict_ratios]

    ensembles = []
    devices = [f"cuda:{i}" for i in range(8)]

    for i in range(2):
        cfgs = product(l1_values[i*2:(i+1)*2], bias_decays)
        models = [
            # this function returns a tuple of (params, buffers)
            # where both are dicts of tensors
            # in this format so they can be ensembled/stacked
            FunctionalSAE.init(cfg.activation_width, cfg.activation_width * 8, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
            for l1_alpha, bias_decay in cfgs
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            # passing the class here as it is used
            # to figure out how to run the model
            # and convert it into a LearnedDict
            models, FunctionalSAE,
            torchopt.adam, {
                "lr": cfg.lr
            },
            # specify the device to run on
            device=device
        )
        # be sure to specify batch_size, device and dict_size as these are all used regardless of configuration
        args = {"batch_size": cfg.batch_size, "device": device, "tied": False, "dict_size": cfg.activation_width * 8}
        name = f"dict_ratio_8_group_{i}"
        
        ensembles.append((ensemble, args, name))
    
    for i in range(2):
        cfgs = product(l1_values[i*2:(i+1)*2], bias_decays)
        models = [
            FunctionalTiedSAE.init(cfg.activation_width, cfg.activation_width * 8, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
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
        args = {"batch_size": cfg.batch_size, "device": device, "tied": True, "dict_size": cfg.activation_width * 8}
        name = f"dict_ratio_8_group_{i}_tied"

        ensembles.append((ensemble, args, name))

    for _ in range(1):
        cfgs = product(l1_values, bias_decays)
        models = [
            FunctionalSAE.init(cfg.activation_width, cfg.activation_width * 4, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
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
        args = {"batch_size": cfg.batch_size, "device": device, "tied": False, "dict_size": cfg.activation_width * 4}
        name = f"dict_ratio_4"

        ensembles.append((ensemble, args, name))
    
    for _ in range(1):
        cfgs = product(l1_values, bias_decays)
        models = [
            FunctionalTiedSAE.init(cfg.activation_width, cfg.activation_width * 4, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
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
        args = {"batch_size": cfg.batch_size, "device": device, "tied": True, "dict_size": cfg.activation_width * 4}
        name = f"dict_ratio_4_tied"

        ensembles.append((ensemble, args, name))

    for _ in range(1):
        cfgs = product(l1_values, bias_decays)
        models = [
            FunctionalSAE.init(cfg.activation_width, cfg.activation_width * 2, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
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
        args = {"batch_size": cfg.batch_size, "device": device, "tied": False, "dict_size": cfg.activation_width * 2}
        name = f"dict_ratio_2"

        ensembles.append((ensemble, args, name))

    for _ in range(1):
        cfgs = product(l1_values, bias_decays)
        models = [
            FunctionalTiedSAE.init(cfg.activation_width, cfg.activation_width * 2, l1_alpha, bias_decay=bias_decay, dtype=cfg.dtype)
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
        args = {"batch_size": cfg.batch_size, "device": device, "tied": True, "dict_size": cfg.activation_width * 2}
        name = f"dict_ratio_2_tied"

        ensembles.append((ensemble, args, name))
    
    # each ensemble is a tuple of (ensemble, args, name)
    # where the name is used to identify the ensemble in the progress bar
    return (ensembles,
        # two slightly different sets of hyperparameters,
        # # that are used in slightly different ways
        # and so you need to specify them separately
        
        # the first list is hyperparameters that vary between ensembles,
        # but are the same for all models in an ensemble
        # the values of these are in the args dict
        ["tied", "dict_size"],
        
        # the second list is hyperparameters that vary between models in the same ensemble
        # the values of these are in the buffers dict, and must be stackable (i.e. 0-dimensional tensors)
        ["l1_alpha", "bias_decay"],
        
        # all the different ranges for each hyperparameter
        {"tied": [True, False], "dict_size": dict_sizes, "l1_alpha": l1_values, "bias_decay": bias_decays})

if __name__ == "__main__":
    cfg = parse_args()
    sweep(tied_vs_not_experiment, cfg)