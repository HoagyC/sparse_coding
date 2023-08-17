#import run
from activation_dataset import get_activation_size
from utils import *

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

import numpy as np

from transformers import GPT2Tokenizer
from transformer_lens import HookedTransformer
import pickle

class BatchedPCA():
    def __init__(self, n_dims, device):
        super().__init__()
        self.n_dims = n_dims
        self.device = device

        self.cov = torch.zeros((n_dims, n_dims), device=device)
        self.mean = torch.zeros((n_dims,), device=device)
        self.n_samples = 0
    
    def train_iter(self, activations):
        # activations: (batch_size, n_dims)
        batch_size = activations.shape[0]
        corrected = activations - self.mean.unsqueeze(0)
        new_mean = self.mean + torch.mean(corrected, dim=0) * batch_size / (self.n_samples + batch_size)
        cov_update = torch.einsum("bi,bj->bij", corrected, activations - new_mean.unsqueeze(0)).mean(dim=0)
        self.cov = self.cov * (self.n_samples / (self.n_samples + batch_size)) + cov_update * batch_size / (self.n_samples + batch_size)
        self.mean = new_mean
        self.n_samples += batch_size
    
    def get_dictionary(self):
        eigvals, eigvecs = torch.linalg.eigh(self.cov)
        return eigvals.detach().cpu().numpy(), eigvecs.detach().cpu().numpy()

def run_pca_on_activation_dataset(cfg: dotdict, outputs_folder):
    # prelim dataset setup

    if len(os.listdir(cfg.dataset_folder)) == 0:
        print("activation dataset not found, throwing error")
        raise Exception("activation dataset not found")

    with open(os.path.join(cfg.dataset_folder, "0.pkl"), "rb") as f:
        dataset = pickle.load(f)
    activation_dim = get_activation_size(cfg.model_name, cfg.layer_loc)
    n_lines = cfg.max_lines
    del dataset

    # actual pca

    pca_model = BatchedPCA(activation_dim, cfg.device)

    n_chunks_in_folder = len(os.listdir(cfg.dataset_folder))

    for chunk_id in range(n_chunks_in_folder):
        chunk_loc = os.path.join(cfg.dataset_folder, str(chunk_id) + ".pkl")
        # realistically can just load the whole thing into memory (only 2GB/chunk)
        print(chunk_loc)
        with open(chunk_loc, "rb") as f:
            chunk = pickle.load(f)
        dataset = DataLoader(chunk, batch_size=cfg.batch_size, shuffle=False)
        for batch in dataset:
            activations = batch[0].to(cfg.device)
            pca_model.train_iter(activations)

    pca_components, pca_directions = pca_model.get_dictionary()

    pca_directions_loc = os.path.join(outputs_folder, "pca_results.pkl")
    with open(pca_directions_loc, "wb") as f:
        pickle.dump((pca_directions, pca_components), f)

    return pca_directions, pca_components

def main():
    from argparser import parse_args

    cfg = parse_args()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    from datetime import datetime

    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    outputs_folder = os.path.join(cfg.outputs_folder, start_time)
    os.makedirs(outputs_folder, exist_ok=True)

    from run import run_real_data_model, AutoEncoder
    from activation_dataset import setup_data

    cfg.model_name = "EleutherAI/pythia-70m-deduped"
    cfg.use_wandb = False
    cfg.dict_ratio_exp_high = 5
    data_split = "train"

    cfg.l1_exp_low = -16
    cfg.l1_exp_high = -14

    model = None 

    if cfg.model_name in ["gpt2", "EleutherAI/pythia-70m-deduped"]:
        model = HookedTransformer.from_pretrained(cfg.model_name, device=cfg.device)
        use_baukit = False
    
    if hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer
    else:
        print("Using default tokenizer from gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    dataset_name = cfg.dataset_name.split("/")[-1] + "-" + cfg.model_name + "-" + str(cfg.layer)
    cfg.dataset_folder = os.path.join(cfg.datasets_folder, dataset_name)
    os.makedirs(cfg.dataset_folder, exist_ok=True)

    if len(os.listdir(cfg.dataset_folder)) == 0:
        print(f"Activations in {cfg.dataset_folder} do not exist, creating them")
        n_lines = setup_data(
            tokenizer, 
            model,
            dataset_name=cfg.dataset_name,
            dataset_folder=cfg.dataset_folder,
            layer=cfg.layer,
            layer_loc=cfg.layer_loc,
            n_chunks=cfg.n_chunks,
            device=cfg.device
        )
    
    # do pca on activations
    pca_directions, pca_components = run_pca_on_activation_dataset(cfg, outputs_folder=outputs_folder)

    run_real_data_model(cfg)

if __name__ == "__main__":
    main()