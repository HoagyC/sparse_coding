# Testing the ability to run the whole pipeline from start to finish
from datetime import datetime
import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torchopt
import wandb

from config import TrainArgs
from autoencoders.sae_ensemble import FunctionalSAE, FunctionalTiedSAE
from autoencoders.ensemble import FunctionalEnsemble
from big_sweep import sweep

def single_setoff(cfg):
    l1_values = np.array([1e-3])
    l1_values = np.concatenate([[0], l1_values])
    ensembles = []

    dict_size = int(cfg.activation_width * cfg.learned_dict_ratio)
    device = cfg.device
    if cfg.tied_ae:
        models = [
            FunctionalTiedSAE.init(
                cfg.activation_width,
                dict_size,
                l1_alpha,
                dtype=cfg.dtype,
            )
            for l1_alpha in l1_values
        ]
    else:
        models = [
            FunctionalSAE.init(
                cfg.activation_width,
                dict_size,
                l1_alpha,
                bias_decay=0.0,
                dtype=cfg.dtype,
            )
            for l1_alpha in l1_values
        ]
        
        
    if cfg.tied_ae:
        ensemble = FunctionalEnsemble(models, FunctionalTiedSAE, torchopt.adam, {"lr": cfg.lr}, device=device)
    else:
        ensemble = FunctionalEnsemble(models, FunctionalSAE, torchopt.adam, {"lr": cfg.lr}, device=device)
    args = {"batch_size": cfg.batch_size, "device": device, "dict_size": dict_size}
    name = f"simple_{cfg.device}"
    ensembles.append((ensemble, args, name))

    print(len(ensembles), "ensembles")
    return (
        ensembles,
        ["dict_size"],
        ["l1_alpha"],
        {"dict_size": [dict_size], "l1_alpha": l1_values},
    )    

# make unittest
class TestEndToEnd(unittest.TestCase):
    def test_end_to_end(self):
        cfg = TrainArgs()
        cfg.model_name = "pythia-70m-deduped"
        cfg.dataset_name = "NeelNanda/pile-10k"

        cfg.batch_size = 500
        cfg.use_wandb = True
        cfg.wandb_images = True
        cfg.save_every = 10
        cfg.use_synthetic_dataset = False
        cfg.dtype = torch.float32
        cfg.lr = 1e-3
        cfg.n_chunks = 1
        cfg.n_repetitions = 1
        cfg.activation_width = 2048
        cfg.layer = 2
        cfg.chunk_size_gb = 0.1

        cfg.layer_loc = "residual"

        cfg.tied_ae = False
        cfg.learned_dict_ratio = 0.5

        cfg.output_folder = (
            f"integration_test"
        )
        cfg.dataset_folder = f"pile10k_test"
        print(cfg.device)
        sweep(single_setoff, cfg)
        wandb.finish()
        torch.cuda.empty_cache()
        

if __name__ == "__main__":
    unittest.main()
