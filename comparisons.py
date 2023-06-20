#import run
from utils import *

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

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

def run_pca_on_activation_dataset(cfg: dotdict):
    # prelim dataset setup

    dataset_name = cfg.dataset_name.split("/")[-1] + "-" + cfg.model_name + "-" + str(cfg.layer)
    cfg.dataset_folder = os.path.join(cfg.datasets_folder, dataset_name)
    os.makedirs(cfg.dataset_folder, exist_ok=True)

    if len(os.listdir(cfg.dataset_folder)) == 0:
        print("activation dataset not found, throwing error")
        raise Exception("activation dataset not found")

    with open(os.path.join(cfg.dataset_folder, "0.pkl"), "rb") as f:
        dataset = pickle.load(f)
    cfg.mlp_width = dataset.tensors[0][0].shape[-1]
    n_lines = cfg.max_lines
    del dataset

    # actual pca

    pca_model = BatchedPCA(cfg.mlp_width, cfg.device)

    n_chunks_in_folder = len(os.listdir(cfg.dataset_folder))

    for chunk_id in range(n_chunks_in_folder):
        chunk_loc = os.path.join(cfg.dataset_folder, str(chunk_id) + ".pkl")
        # realistically can just load the whole thing into memory (only 2GB/chunk)
        dataset = DataLoader(torch.load(chunk_loc), batch_size=cfg.batch_size, shuffle=False)
        for batch in dataset:
            activations = batch.to(cfg.device)
            pca_model.train_iter(activations)

    pca_components, pca_directions = pca_model.get_dictionary()

    outputs_folder = os.path.join(cfg.outputs_folder, start_time)

    pca_directions_loc = os.path.join(outputs_folder, "pca_results.pkl")
    with open(pca_directions_loc, "wb") as f:
        pickle.dump((pca_directions, pca_components), f)

    return pca_directions, pca_components

def main():
    from argparser import parse_args

    cfg = parse_args()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    from run import setup_data, mean_max_cosine_similarity

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
        n_lines = setup_data(cfg, tokenizer, model, use_baukit=use_baukit)
    else:
        print(f"Activations in {cfg.dataset_folder} already exist, loading them")
        # get mlp_width from first file
        with open(os.path.join(cfg.dataset_folder, "0.pkl"), "rb") as f:
            dataset = pickle.load(f)
        cfg.mlp_width = dataset.tensors[0][0].shape[-1]
        n_lines = cfg.max_lines
        del dataset
    
    # do pca on activations
    pca_directions, pca_components = run_pca_on_activation_dataset(cfg)

    # train sparse autoencoders on activations

    l1_range = [cfg.l1_exp_base**exp for exp in range(cfg.l1_exp_low, cfg.l1_exp_high)]
    dict_ratios = [cfg.dict_ratio_exp_base**exp for exp in range(cfg.dict_ratio_exp_low, cfg.dict_ratio_exp_high)]
    dict_sizes = [int(cfg.mlp_width * ratio) for ratio in dict_ratios]

    print("Range of l1 values being used: ", l1_range)
    print("Range of dict_sizes being used:", dict_sizes)
    dead_neurons_matrix = np.zeros((len(l1_range), len(dict_sizes)))
    recon_loss_matrix = np.zeros((len(l1_range), len(dict_sizes)))
    l1_loss_matrix = np.zeros((len(l1_range), len(dict_sizes)))

    # 2D array of learned dictionaries, indexed by l1_alpha and learned_dict_ratio, start with Nones
    auto_encoders = [[AutoEncoder(cfg.mlp_width, n_feats, l1_coef=l1_ndx).to(cfg.device) for n_feats in dict_sizes] for l1_ndx in l1_range]

    learned_dicts: List[List[Optional[torch.Tensor]]] = [[None for _ in range(len(dict_sizes))] for _ in range(len(l1_range))]

    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    outputs_folder = os.path.join(cfg.outputs_folder, start_time)
    os.makedirs(outputs_folder, exist_ok=True)

    step_n = 0
    for l1_ndx, dict_size_ndx in list(itertools.product(range(len(l1_range)), range(len(dict_sizes)))):
        l1_loss = l1_range[l1_ndx]
        dict_size = dict_sizes[dict_size_ndx]

        cfg.l1_alpha = l1_loss
        cfg.n_components_dictionary = dict_size
        auto_encoder = auto_encoders[l1_ndx][dict_size_ndx]

        auto_encoder, n_dead_neurons, reconstruction_loss, l1_loss, completed_batches = run_with_real_data(cfg, auto_encoder, completed_batches=step_n)
        if l1_ndx == (len(l1_range) - 1) and dict_size_ndx == (len(dict_sizes) - 1):
            step_n = completed_batches

        dead_neurons_matrix[l1_ndx, dict_size_ndx] = n_dead_neurons
        recon_loss_matrix[l1_ndx, dict_size_ndx] = reconstruction_loss
        l1_loss_matrix[l1_ndx, dict_size_ndx] = l1_loss

    learned_dicts = [[auto_e.decoder.weight.detach().cpu().data.t() for auto_e in l1] for l1 in auto_encoders]

    # save learned_dicts
    learned_dicts_loc = os.path.join(outputs_folder, "learned_dicts.pkl")
    with open(learned_dicts_loc, "wb") as f:
        pickle.dump(learned_dicts, f)