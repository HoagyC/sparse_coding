from cluster_runs import dispatch_lite, statusbar_lite, update_statusbar_lite, collect_lite
from big_sweep import init_model_dataset, unstacked_to_learned_dicts

from autoencoders.pca import calc_pca
from autoencoders.sae_ensemble import FunctionalTiedSAE, FunctionalTiedCenteredSAE
from autoencoders.ensemble import FunctionalEnsemble

from utils import dotdict

import torch
import numpy as np
import os

from contextlib import nullcontext

import torchopt

import time

def get_jobs(cfg, pca):
    translation, rotation, scaling = pca.get_centering_transform()

    ratio = 4
    dict_size = cfg.activation_width * ratio
    l1_vals = np.logspace(-4, -2, 8)

    jobs = []
    devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"]

    with nullcontext():
        device = devices.pop()
        models = [
            FunctionalTiedSAE.init(
                cfg.activation_width,
                dict_size,
                l1_val,
            )
            for l1_val in l1_vals
        ]
        ensemble = FunctionalEnsemble(
            models, FunctionalTiedSAE,
            torchopt.adam, {
                "lr": 1e-3,
            },
            device=device,
        )
        jobs.append((ensemble, {"batch_size": 1024, "device": device, "dict_size": dict_size}, f"no_centering"))

    with nullcontext():
        device = devices.pop()
        models = [
            FunctionalTiedSAE.init(
                cfg.activation_width,
                dict_size,
                l1_val,
                translation=translation,
            )
            for l1_val in l1_vals
        ]
        ensemble = FunctionalEnsemble(
            models, FunctionalTiedSAE,
            torchopt.adam, {
                "lr": 1e-3,
            },
            device=device,
        )
        jobs.append((ensemble, {"batch_size": 1024, "device": device, "dict_size": dict_size}, f"mean_centered"))
    
    """
    with nullcontext():
        device = devices.pop()
        models = [
            FunctionalTiedCenteredSAE.init(
                cfg.activation_width,
                dict_size,
                l1_val,
            )
            for l1_val in l1_vals
        ]
        ensemble = FunctionalEnsemble(
            models, FunctionalTiedCenteredSAE,
            torchopt.adam, {
                "lr": 1e-3,
            },
            device=device,
        )
        jobs.append((ensemble, {"batch_size": 1024, "device": device, "dict_size": dict_size}, f"learned_centered"))
    
    with nullcontext():
        device = devices.pop()
        models = [
            FunctionalTiedCenteredSAE.init(
                cfg.activation_width,
                dict_size,
                l1_val,
                center=translation,
            )
            for l1_val in l1_vals
        ]
        ensemble = FunctionalEnsemble(
            models, FunctionalTiedCenteredSAE,
            torchopt.adam, {
                "lr": 1e-3,
            },
            device=device,
        )
        jobs.append((ensemble, {"batch_size": 1024, "device": device, "dict_size": dict_size}, f"learned_centered_mean_init"))

    with nullcontext():
        device = devices.pop()
        models = [
            FunctionalTiedSAE.init(
                cfg.activation_width,
                dict_size,
                l1_val,
                translation=translation,
                rotation=rotation,
                scaling=scaling,
            )
            for l1_val in np.logspace(-4, -2, 8)
        ]
        ensemble = FunctionalEnsemble(
            models, FunctionalTiedSAE,
            torchopt.adam, {
                "lr": 1e-3,
            },
            device=device,
        )
        jobs.append((ensemble, {"batch_size": 1024, "device": device, "dict_size": dict_size}, f"sphered"))
    """

    return jobs

def sphered_job(ensemble, cfg, args, name, progress_counter):
    torch.set_grad_enabled(False)
    np.random.seed(0)

    device = args["device"]
    batch_size = args["batch_size"]

    total_samples = cfg.n_samples
    current_samples = 0

    for dataset_chunk_idx, dataset_chunk_id in enumerate(np.random.permutation(cfg.n_chunks)):
        dataset = torch.load(os.path.join(cfg.dataset_folder, f"{dataset_chunk_id}.pt"))
        
        for i in range(0, dataset.shape[0], batch_size):
            j = min(i + batch_size, dataset.shape[0])
            batch = dataset[i:j].to(device)

            with torch.no_grad():
                losses, _ = ensemble.step_batch(batch)
            
            current_samples += batch.shape[0]
            progress_counter.value = current_samples / total_samples
            #print(progress_counter.value)
        
        learned_dicts = unstacked_to_learned_dicts(ensemble, args, ["dict_size"], ["l1_alpha"])
        torch.save(learned_dicts, os.path.join(cfg.outputs_folder, f"{name}_{dataset_chunk_idx}.pt"))

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    cfg = dotdict({
        "model_name": "EleutherAI/pythia-160m-deduped",
        "dataset_name": "NeelNanda/pile-10k",
        "layer": 3,
        "layer_loc": "residual",
        "outputs_folder": "outputs_sphere",
        "dataset_folder": "activation_data_sphere",
        "n_chunks": 30,
        "chunk_size_gb": 2.,
        "device": "cuda:0",
        "n_samples": None,
        "center_dataset": False,
        "max_lines": 10000,
    })

    os.makedirs(cfg.outputs_folder, exist_ok=True)
    os.makedirs(cfg.dataset_folder, exist_ok=True)

    cfg.n_samples = init_model_dataset(cfg)
    cfg.n_chunks = len(os.listdir(cfg.dataset_folder))

    dataset = torch.load(os.path.join(cfg.dataset_folder, "0.pt"))
    pca = calc_pca(dataset, batch_size=512, device=cfg.device)
    jobs = get_jobs(cfg, pca)
    processes = [(dispatch_lite(cfg, ensemble, args, name, sphered_job), args, name) for ensemble, args, name in jobs]

    statusbar = statusbar_lite(processes, n_points=1000)

    while not collect_lite(processes):
        update_statusbar_lite(statusbar, processes, n_points=1000)
        time.sleep(0.1)