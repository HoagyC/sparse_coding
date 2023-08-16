import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.utils.data as data

import torchopt

from cluster_runs import dispatch_job_on_chunk

from autoencoders.ensemble import FunctionalEnsemble
from autoencoders.sae_ensemble import FunctionalSAE, FunctionalTiedSAE, FunctionalMaskedTiedSAE, FunctionalThresholdingSAE
from autoencoders.semilinear_autoencoder import SemiLinearSAE
from autoencoders.residual_denoising_autoencoder import FunctionalLISTADenoisingSAE, FunctionalResidualDenoisingSAE
from autoencoders.direct_coef_search import DirectCoefOptimizer
from autoencoders.topk_encoder import TopKEncoder

from argparser import parse_args

import numpy as np

import shutil
from itertools import product

from big_sweep import sweep

import shutil
import os

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
        
        # all the different ranges for each hyperparameter, these don't need to be in order, it's used to generate a cartesian product to then filter outputs before plotting
        {"tied": [True, False], "dict_size": dict_sizes, "l1_alpha": l1_values, "bias_decay": bias_decays})

DICT_RATIO = None

import tqdm

def topk_experiment(cfg):
    sparsity_levels = np.arange(1, 161, 10)
    dict_ratios = [0.5, 1, 2, 4, 0.5, 1, 2, 4]
    dict_sizes = [int(cfg.activation_width * ratio) for ratio in dict_ratios]
    devices = [f"cuda:{i}" for i in range(8)]

    ensembles = []
    for i in tqdm.tqdm(range(8)):
        dict_ratio = dict_ratios[i]
        dict_size = int(cfg.activation_width * dict_ratio)
        cfgs = sparsity_levels
        models = [
            TopKEncoder.init(cfg.activation_width, dict_size, sparsity, dtype=cfg.dtype)
            for sparsity in cfgs
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, TopKEncoder,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device,
            no_stacking=True
        )
        args = {"batch_size": cfg.batch_size, "device": device, "dict_size": dict_size}
        name = f"topk_{i}"
        ensembles.append((ensemble, args, name))
    
    return (ensembles, ["dict_size"], ["sparsity"], {"dict_size": dict_sizes, "sparsity": sparsity_levels})

def synthetic_linear_range(cfg):
    l1_vals = np.logspace(-4, -2, 32)
    dict_ratios = [0.5, 1, 2, 4]
    dict_sizes = [int(cfg.activation_width * ratio) for ratio in dict_ratios]
    settings = list(product([l1_vals[:16], l1_vals[16:]], dict_ratios))

    devices = [f"cuda:{i}" for i in range(8)]

    ensembles = []
    for i in tqdm.tqdm(range(8)):
        dict_ratio = settings[i][1]
        dict_size = int(cfg.activation_width * dict_ratio)
        l1_range = settings[i][0]
        print(settings[i])
        models = [
            FunctionalTiedSAE.init(cfg.activation_width, dict_size, l1_alpha, dtype=cfg.dtype)
            for l1_alpha in l1_range
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalTiedSAE,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        args = {"batch_size": cfg.batch_size, "device": device, "dict_size": dict_size}
        name = f"topk_{i}"
        ensembles.append((ensemble, args, name))
    
    return (ensembles, ["dict_size"], ["l1_alpha"], {"dict_size": dict_sizes, "l1_alpha": l1_vals})

def dense_l1_range_experiment(cfg):
    l1_values = np.logspace(-4, -2, 16)
    devices = [f"cuda:{i}" for i in range(8)]

    ensembles = []
    for i in range(8):
        cfgs = l1_values[i*2:(i+1)*2]
        dict_size = int(cfg.activation_width * cfg.learned_dict_ratio)
        if cfg.tied_ae:
            models = [
                FunctionalTiedSAE.init(cfg.activation_width, dict_size, l1_alpha, bias_decay=0.0, dtype=cfg.dtype)
                for l1_alpha in cfgs
            ]
        else:
            models = [
                FunctionalSAE.init(cfg.activation_width, dict_size, l1_alpha, bias_decay=0.0, dtype=cfg.dtype)
                for l1_alpha in cfgs
            ]
    

        device = devices.pop()
        if cfg.tied_ae:
            ensemble = FunctionalEnsemble(
                models, FunctionalTiedSAE,
                torchopt.adam, {
                    "lr": cfg.lr
                },
                device=device
            )
        else:
            ensemble = FunctionalEnsemble(
                models, FunctionalSAE,
                torchopt.adam, {
                    "lr": cfg.lr
                },
                device=device
            )
        args = {"batch_size": cfg.batch_size, "device": device, "dict_size": dict_size}
        name = f"l1_range_8_{i}"
        ensembles.append((ensemble, args, name))

    return (ensembles, ["dict_size"], ["l1_alpha"], {"dict_size": [dict_size], "l1_alpha": l1_values})


def residual_denoising_experiment(cfg):
    l1_values = np.logspace(-5, -3, 16)
    devices = [f"cuda:{i}" for i in range(8)]

    ensembles = []
    for i in range(4):
        #print(f"cuda:{i}", torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i))
        cfgs = l1_values[i*4:(i+1)*4]
        dict_size = int(cfg.activation_width * DICT_RATIO)
        models = [
            FunctionalLISTADenoisingSAE.init(cfg.activation_width, dict_size, 3, l1_alpha, dtype=cfg.dtype)
            for l1_alpha in cfgs
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalLISTADenoisingSAE,
            #adam_grouped.Adam, {
            torchopt.adam, {
                "lr": cfg.lr
            #    "lr_groups": FunctionalResidualDenoisingSAE.init_lr(3, lr=1e-4, lr_encoder=1e-3),
            #    "betas": (0.9, 0.999),
            #    "eps": 1e-8
            },
            device=device
        )
        args = {"batch_size": cfg.batch_size, "device": device, "dict_size": dict_size}
        name = f"residual_denoising_8_{i}"
        ensembles.append((ensemble, args, name))

    return (ensembles, ["dict_size"], ["l1_alpha"], {"dict_size": [dict_size], "l1_alpha": l1_values})

def residual_denoising_comparison(cfg):
    l1_values = np.logspace(-4, -2, 16)
    devices = [f"cuda:{i}" for i in range(4)]

    ensembles = []
    for i in range(4):
        #print(f"cuda:{i}", torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i))
        cfgs = l1_values[i*4:(i+1)*4]
        dict_size = int(cfg.activation_width * DICT_RATIO)
        models = [
            FunctionalTiedSAE.init(cfg.activation_width, dict_size, l1_alpha, dtype=cfg.dtype)
            for l1_alpha in cfgs
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalTiedSAE,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        args = {"batch_size": cfg.batch_size, "device": device, "dict_size": dict_size}
        name = f"residual_denoising_8_{i}"
        ensembles.append((ensemble, args, name))

    return (ensembles, ["dict_size"], ["l1_alpha"], {"dict_size": [dict_size], "l1_alpha": l1_values})

def thresholding_experiment(cfg):
    l1_values = np.logspace(-4, -2, 16)
    devices = [f"cuda:{i}" for i in range(4)]

    dict_ratio = 4

    ensembles = []
    for i in range(4):
        #print(f"cuda:{i}", torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i))
        cfgs = l1_values[i*4:(i+1)*4]
        dict_size = int(cfg.activation_width * dict_ratio)
        models = [
            FunctionalThresholdingSAE.init(cfg.activation_width, dict_size, l1_alpha, dtype=cfg.dtype)
            for l1_alpha in cfgs
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalThresholdingSAE,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        args = {"batch_size": cfg.batch_size, "device": device, "dict_size": dict_size}
        name = f"thresholding_8_{i}"
        ensembles.append((ensemble, args, name))

    return (ensembles, ["dict_size"], ["l1_alpha"], {"dict_size": [dict_size], "l1_alpha": l1_values})

def run_thresholding():
    cfg = parse_args()

    cfg.model_name = "EleutherAI/pythia-70m-deduped"
    cfg.layer = 2
    cfg.layer_loc = "residual"

    cfg.use_synthetic_dataset = False
    cfg.dataset_folder = "activation_data"
    cfg.output_folder = "output_thresholding"
    cfg.n_chunks = 10

    cfg.batch_size = 1024
    cfg.gen_batch_size = 4096
    cfg.n_ground_truth_components = 1024
    cfg.activation_width = 512
    cfg.noise_magnitude_scale = 0.001
    cfg.feature_prob_decay = 0.99
    cfg.feature_num_nonzero = 10

    cfg.lr = 1e-3
    cfg.use_wandb = True
    cfg.wandb_images = False

    cfg.dtype = torch.float32

    sweep(thresholding_experiment, cfg)

def run_resid_denoise():
    cfg = parse_args()

    cfg.model_name = "EleutherAI/pythia-70m-deduped"
    cfg.layer = 2
    cfg.layer_loc = "residual"

    cfg.use_synthetic_dataset = False
    cfg.dataset_folder = "activation_data"
    cfg.output_folder = "output_aidan"
    cfg.n_chunks = 10

    cfg.batch_size = 1024
    cfg.gen_batch_size = 4096
    cfg.n_ground_truth_components = 1024
    cfg.activation_width = 512
    cfg.noise_magnitude_scale = 0.001
    cfg.feature_prob_decay = 0.99
    cfg.feature_num_nonzero = 10

    cfg.lr = 1e-3
    cfg.use_wandb = False
    cfg.wandb_images = False
    cfg.dtype = torch.float32

    for dict_ratio in [4]:
        global DICT_RATIO
        DICT_RATIO = dict_ratio
        cfg.output_folder = f"output_{dict_ratio}_lista_neg"
        sweep(residual_denoising_experiment, cfg)

    
def zero_l1_baseline(cfg):
    l1_values = np.array([0.0])
    devices = ["cuda:1"]

    ensembles = []
    cfgs = l1_values
    dict_size = int(cfg.activation_width * 4)
    if cfg.tied_ae:
        models = [
            FunctionalTiedSAE.init(cfg.activation_width, dict_size, l1_alpha, bias_decay=0.0, dtype=cfg.dtype)
            for l1_alpha in cfgs
        ]
    else:
        models = [
            FunctionalSAE.init(cfg.activation_width, dict_size, l1_alpha, bias_decay=0.0, dtype=cfg.dtype)
            for l1_alpha in cfgs
        ]

    device = devices.pop()
    if cfg.tied_ae:
        ensemble = FunctionalEnsemble(
            models, FunctionalTiedSAE,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
    else:
        ensemble = FunctionalEnsemble(
            models, FunctionalSAE,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
    args = {"batch_size": cfg.batch_size, "device": device, "dict_size": dict_size}
    name = f"l1_range_zero_b"
    ensembles.append((ensemble, args, name))

    return (ensembles, ["dict_size"], ["l1_alpha"], {"dict_size": [dict_size], "l1_alpha": l1_values})

def dict_ratio_experiment(cfg):
    #l1_values = np.logspace(-4, -2, 12)
    dict_sizes = [int(512 * x) for x in np.linspace(1, 5, 8)]
    max_size = max(dict_sizes)

    l1_value = 1e-3

    devices = [f"cuda:{i}" for i in [1, 2, 3, 4, 6, 7]]

    ensembles = []
    for i in range(6):
        #l1_range = l1_values[i*2:(i+1)*2]
        models = [
            FunctionalMaskedTiedSAE.init(cfg.activation_width, dict_size, max_size, l1_value, dtype=cfg.dtype)
            for _ in range(12) for dict_size in dict_sizes
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalMaskedTiedSAE,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        args = {"batch_size": cfg.batch_size, "device": device, "dict_size": max_size}
        name = f"l1_{i}"
        ensembles.append((ensemble, args, name))
    
    return (ensembles, [], ["l1_alpha", "dict_size"], {"dict_size": dict_sizes, "l1_alpha": [l1_value]})

def run_dict_ratio():
    cfg = parse_args()

    cfg.model_name = "EleutherAI/pythia-70m-deduped"
    cfg.dataset_name = "NeelNanda/pile-10k"

    cfg.layer = 4
    cfg.layer_loc = "residual"

    cfg.use_synthetic_dataset = True

    cfg.feature_num_nonzero = 100
    cfg.gen_batch_size = 4096
    cfg.activation_width = 512
    cfg.noise_magnitude_scale = 0.0
    cfg.n_ground_truth_components = 2048
    cfg.feature_prob_decay = 0.996
    cfg.lr = 1e-3
    cfg.n_chunks = 10
    cfg.correlated_components = False
    cfg.chunk_size_gb = 2

    cfg.batch_size = 1024

    cfg.lr = 1e-3
    cfg.use_wandb = False
    cfg.wandb_images = False
    cfg.dtype = torch.float32

    cfg.n_repetitions = 1

    cfg.dataset_folder = "activation_data"
    cfg.output_folder = "output_dict_ratio"

    #shutil.rmtree(cfg.dataset_folder, ignore_errors=True)
    #os.makedirs(cfg.dataset_folder, exist_ok=True)

    sweep(dict_ratio_experiment, cfg)

def run_dense_l1_range():
    cfg = parse_args()
    cfg.model_name = "EleutherAI/pythia-70m-deduped"
    cfg.dataset_name = "EleutherAI/pile"

    cfg.batch_size = 2048
    cfg.layer_loc = "attn"
    cfg.activation_width = 512

    cfg.output_folder = f"layerloctest_{'_tied' if cfg.tied_ae else ''}_{cfg.layer_loc}_l{cfg.layer}_r{int(cfg.learned_dict_ratio)}"
    cfg.dataset_folder = f"pilechunks_l{cfg.layer}_{cfg.layer_loc}"
    cfg.use_synthetic_dataset = False
    cfg.dtype = torch.float32
    cfg.lr = 3e-4
    cfg.n_chunks=30

    sweep(dense_l1_range_experiment, cfg)

def run_across_layers():
    cfg = parse_args()
    cfg.model_name = "EleutherAI/pythia-70m-deduped"
    cfg.dataset_name = "EleutherAI/pile"

    cfg.batch_size = 1024
    cfg.use_wandb = False
    cfg.activation_width = 512
    cfg.save_every = 5
    cfg.n_chunks=10
    cfg.tied_ae=True
    for layer in [0, 1, 2, 3, 4, 5]:
        for layer_loc in ["residual", "mlp"]:
            for dict_ratio in [0.5, 1, 2, 4, 8, 16, 32]:
                cfg.layer = layer
                cfg.layer_loc = layer_loc
                cfg.learned_dict_ratio = dict_ratio

                print(f"Running layer {layer}, layer location {layer_loc}, dict_ratio {dict_ratio}")

                cfg.output_folder = f"output_hoagy_dense_sweep{'_tied' if cfg.tied_ae else ''}_{layer_loc}_l{cfg.layer}_r{int(cfg.learned_dict_ratio)}"
                cfg.dataset_folder = f"pilechunks_l{cfg.layer}_{layer_loc}"
                
                print(f"Output folder: {cfg.output_folder}, dataset folder: {cfg.dataset_folder}")
                
                cfg.use_synthetic_dataset = False
                cfg.dtype = torch.float32
                cfg.lr = 1e-3

                sweep(dense_l1_range_experiment, cfg)

            # delete the dataset to save space
            shutil.rmtree(cfg.dataset_folder)

def run_across_layers_attn():
    cfg = parse_args()
    cfg.model_name = "EleutherAI/pythia-70m-deduped"
    cfg.dataset_name = "EleutherAI/pile"

    cfg.batch_size = 2048
    cfg.use_wandb = False
    cfg.save_every = 2
    cfg.tied_ae=True
    for layer in [0, 1, 2, 3, 4, 5]:
        layer_loc = "attn"
        for dict_ratio in [1, 2, 4, 8]:
            cfg.layer = layer
            cfg.layer_loc = layer_loc
            cfg.learned_dict_ratio = dict_ratio

            cfg.output_folder = f"output_attn_sweep{'_tied' if cfg.tied_ae else ''}_{cfg.layer_loc}_l{cfg.layer}_r{int(cfg.learned_dict_ratio)}"
            cfg.dataset_folder = f"pilechunks_l{cfg.layer}_{cfg.layer_loc}"
            cfg.use_synthetic_dataset = False
            cfg.dtype = torch.float32
            cfg.lr = 3e-4
            cfg.n_chunks=10

            sweep(dense_l1_range_experiment, cfg)

        # delete the dataset
        shutil.rmtree(cfg.dataset_folder)

def run_across_layers_mlp_out():
    cfg = parse_args()
    cfg.model_name = "EleutherAI/pythia-70m-deduped"
    cfg.dataset_name = "EleutherAI/pile"

    cfg.batch_size = 2048
    cfg.use_wandb = False
    cfg.save_every = 2
    cfg.tied_ae=True
    for layer in [0, 1, 3, 4, 5]:
        layer_loc = "mlpout"
        for dict_ratio in [1, 2, 4, 8]:
            cfg.layer = layer
            cfg.layer_loc = layer_loc
            cfg.learned_dict_ratio = dict_ratio

            cfg.output_folder = f"output_sweep{'_tied' if cfg.tied_ae else ''}_{cfg.layer_loc}_l{cfg.layer}_r{int(cfg.learned_dict_ratio)}"
            cfg.dataset_folder = f"pilechunks_l{cfg.layer}_{cfg.layer_loc}"
            cfg.use_synthetic_dataset = False
            cfg.dtype = torch.float32
            cfg.lr = 3e-4
            cfg.n_chunks=10

            sweep(dense_l1_range_experiment, cfg)

        # delete the dataset
        shutil.rmtree(cfg.dataset_folder)

def run_across_layers_mlp_untied():
    cfg = parse_args()
    cfg.model_name = "EleutherAI/pythia-70m-deduped"
    cfg.dataset_name = "EleutherAI/pile"

    cfg.batch_size = 2048
    cfg.use_wandb = False
    cfg.save_every = 2
    cfg.tied_ae=False
    for layer in [0, 1, 2, 3, 4, 5]:
        layer_loc = "mlp"
        for dict_ratio in [1, 2, 4, 8]:
            cfg.layer = layer
            cfg.layer_loc = layer_loc
            cfg.learned_dict_ratio = dict_ratio

            cfg.output_folder = f"output_sweep{'_tied' if cfg.tied_ae else ''}_{cfg.layer_loc}_l{cfg.layer}_r{int(cfg.learned_dict_ratio)}"
            cfg.dataset_folder = f"pilechunks_l{cfg.layer}_{cfg.layer_loc}"
            cfg.use_synthetic_dataset = False
            cfg.dtype = torch.float32
            cfg.lr = 3e-4
            cfg.n_chunks=10

            sweep(dense_l1_range_experiment, cfg)

        # delete the dataset
        shutil.rmtree(cfg.dataset_folder)

def run_zero_l1_baseline():
    cfg = parse_args()
    cfg.model_name = "EleutherAI/pythia-70m-deduped"
    cfg.dataset_name = "NeelNanda/pile-10k"
    cfg.layer = 3
    cfg.layer_loc="residual"
    cfg.tied_ae = True
    cfg.dict_ratio=4

    cfg.use_wandb=False

    cfg.batch_size = 2048
    cfg.activation_width = 512

    cfg.output_folder = f"output_zero_b_{cfg.dict_ratio}"
    cfg.dataset_folder = f"activation_data/layer_3"
    cfg.use_synthetic_dataset = False
    cfg.dtype = torch.float32
    cfg.lr = 3e-4
    cfg.n_chunks=38

    sweep(zero_l1_baseline, cfg)

def topk():
    cfg = parse_args()
    cfg.model_name = "EleutherAI/pythia-70m-deduped"
    cfg.dataset_name = "EleutherAI/pile"

    cfg.batch_size = 1024
    cfg.activation_width = 512

    cfg.use_residual = True

    cfg.wandb_images = False

    cfg.output_folder = f"output_topk"
    cfg.dataset_folder = f"activation_data"
    cfg.use_synthetic_dataset = False
    cfg.dtype = torch.float32
    cfg.lr = 1e-3
    cfg.n_chunks=10
    cfg.n_repetitions = 5

    sweep(topk_experiment, cfg)

def synthetic_test():
    import shutil
    import os

    cfg = parse_args()

    cfg.use_synthetic_dataset = True

    cfg.dataset_folder = f"activation_data_synthetic"

    cfg.batch_size = 1024
    cfg.gen_batch_size = 4096
    cfg.activation_width = 512
    #cfg.noise_magnitude_scale = 0.0
    cfg.feature_prob_decay = 1.0
    cfg.lr = 1e-3
    cfg.n_chunks = 10
    cfg.correlated_components = False

    cfg.wandb_images = False
    cfg.use_wandb = False

    cfg.dtype = torch.float32

    os.makedirs(cfg.dataset_folder, exist_ok=True)

    n_ground_truth_components = [1024, 2048]
    feature_num_nonzero = [10, 50, 100]
    noise_magnitude = [0.1]
    for (noise_mag, num_nz, n_ground) in product(noise_magnitude, feature_num_nonzero, n_ground_truth_components):
        shutil.rmtree(cfg.dataset_folder)

        cfg.noise_magnitude_scale = noise_mag
        cfg.n_ground_truth_components = n_ground
        cfg.feature_num_nonzero = num_nz
        cfg.output_folder = f"output_synthetic_{noise_mag:.2E}_{n_ground}_{num_nz}"

        sweep(synthetic_linear_range, cfg)

def pythia_1_4_b_dict(cfg):
    dict_ratio = 6
    l1_values = np.logspace(-4, -2, 5)
    dict_size = int(cfg.activation_width * dict_ratio)
    devices = ["cuda:1"]

    ensembles = []
    for i in range(1):
        #l1_range = l1_values[i*2:(i+1)*2]
        models = [
            FunctionalTiedSAE.init(cfg.activation_width, dict_size, l1_value, dtype=cfg.dtype)
            for l1_value in l1_values
        ]
        device = devices.pop()
        ensemble = FunctionalEnsemble(
            models, FunctionalTiedSAE,
            torchopt.adam, {
                "lr": cfg.lr
            },
            device=device
        )
        args = {"batch_size": cfg.batch_size, "device": device, "dict_size": max_size}
        name = f"l1_{i}"
        ensembles.append((ensemble, args, name))
    
    return (ensembles, [], ["l1_alpha", "dict_size"], {"dict_size": dict_sizes, "l1_alpha": [l1_value]})

def run_pythia_1_4_b_sweep():
    cfg = parse_args()

    cfg.model_name = "EleutherAI/pythia-1.4B-deduped"
    cfg.dataset_name = "EleutherAI/pile"

    cfg.batch_size = 1024
    cfg.lr = 1e-3

    cfg.use_wandb = False
    cfg.wandb_images = False

    cfg.activation_width = 512
    cfg.n_chunks = 30

    cfg.dataset_folder = "activation_data_1_4_b"

    sweep(pythia_1_4_b_dict, cfg)

if __name__ == "__main__":
    run_zero_l1_baseline()