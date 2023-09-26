from dataclasses import dataclass
import json
import os
from typing import List
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import tqdm
import wandb

from autoencoders.learned_dict import LearnedDict
from activation_dataset import get_activation_size, setup_data
from big_sweep import get_model
from config import BaseArgs


class SAE(nn.Module, LearnedDict):
    def __init__(self, input_size, latent_size, l1_alpha):
        """
        Class for a sparse autoencoder with tied weights, designed for training a single dictionary with large data volumes.
        """
        super(SAE, self).__init__()
        dict = torch.randn(latent_size, input_size)
        dict = dict / torch.norm(dict, 2, dim=-1, keepdim=True)

        self.dict = nn.Parameter(dict)
        self.encoder = nn.Parameter(dict.clone().T)
        self.threshold = nn.Parameter(torch.zeros(latent_size))
        self.centering = nn.Parameter(torch.zeros(input_size))
        self.l1_alpha = l1_alpha
    
    def get_learned_dict(self):
        normed_dict = self.dict / torch.norm(self.dict, 2, dim=-1, keepdim=True)
        return normed_dict

    def encode(self, x):
        x_centered = x - self.centering
        c = F.relu(torch.einsum("dn,bd->bn", self.encoder, x_centered) + self.threshold)

        return c

    def forward(self, x):
        normed_dict = self.get_learned_dict()

        c = self.encode(x)
        x_hat = torch.einsum("nd,bn->bd", normed_dict, c) + self.centering

        mse_losses = (x - x_hat).pow(2).mean(dim=-1) # mean per batch element
        mse_loss = mse_losses.mean()
        sparsity_loss = self.l1_alpha * torch.norm(c, 1, dim=-1).mean()
        return mse_loss + sparsity_loss, mse_loss, sparsity_loss, c, mse_losses

    def to_device(self, device):
        self.to(device)
        
class UntiedSAE(nn.Module, LearnedDict):
    def __init__(self, input_size, latent_size, l1_alpha):
        """
        Class for a sparse autoencoder with tied weights, designed for training a single dictionary with large data volumes.
        """
        super(UntiedSAE, self).__init__()    

        self.decoder = torch.randn(latent_size, input_size)
        self.decoder /= torch.norm(self.decoder, 2, dim=-1, keepdim=True)
        self.decoder = nn.Parameter(self.decoder)
        self.encoder = nn.Parameter(torch.randn(input_size, latent_size))
        self.threshold = nn.Parameter(torch.zeros(latent_size))
        self.centering = nn.Parameter(torch.zeros(input_size))
        self.l1_alpha = l1_alpha
    
    def get_learned_dict(self):
        normed_dict = self.decoder / torch.norm(self.decoder, 2, dim=-1, keepdim=True)
        return normed_dict

    def encode(self, x):
        x = x - self.centering
        c = F.relu(torch.einsum("dn,bd->bn", self.encoder, x) + self.threshold)
        return c

    def forward(self, x):
        normed_dict = self.get_learned_dict()

        c = self.encode(x)
        x_hat = torch.einsum("nd,bn->bd", normed_dict, c)
        # x_hat = x_hat + self.centering

        mse_losses = (x - x_hat).pow(2).mean(dim=-1) # mean per batch element
        mse_loss = mse_losses.mean()
        sparsity_loss = self.l1_alpha * torch.norm(c, 1, dim=-1).mean()
        return mse_loss + sparsity_loss, mse_loss, sparsity_loss, c, mse_losses

    def to_device(self, device):
        self.to(device)

        
class DatasetWithIndex(torch.utils.data.Dataset):
    """
    Takes in a matrix with the first dimension being the index and the second dimension being the data vectors.
    Returns a Dataset which returns the index along with the data vector.
    """
    def __init__(self, data):
        self.data = data
        self.shape = data.shape
    
    def __getitem__(self, index):
        return index, self.data[index]
    
    def __len__(self):
        return self.data.shape[0]
    
    
class WorstIndices:
    """
    Maintains a list of the worst indices seen so far, along with their losses.
    """
    def __init__(self, n_ndxs: int) -> None:
        self.n_ndxs = n_ndxs
        self.ndxs: List[int] = []
        self.losses: List[float] = []
    
    def update(self, ndx, loss):
        if len(self.ndxs) < self.n_ndxs:
            self.ndxs.append(ndx)
            self.losses.append(loss)
        else:
            min_loss = min(self.losses)
            min_ndx = self.losses.index(min_loss)
            if loss > min_loss:
                self.ndxs[min_ndx] = ndx
                self.losses[min_ndx] = loss
    
    def get_worst(self, n_worst):
        """
        Return the n_worst worst indices seen so far.
        """
        # sort by loss
        sorted_ndxs = [ndx for _, ndx in sorted(zip(self.losses, self.ndxs), reverse=True)]
        return sorted_ndxs[:n_worst]
        
    
    
def process_reinit(rank, cfg, world_size):
    setup(rank, world_size)

    if rank == 0:
        run = wandb.init(
            reinit=True,
            project="sparse coding",
            config=dict(cfg),
            name=cfg.wandb_run_name,
            entity="sparse_coding",
        )

    # Create model and move it to GPU.
    n_features = 4096
    input_size = 2048
    model = UntiedSAE(input_size, n_features, 1e-3)
    model.to(rank)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

    torch.manual_seed(cfg.seed)

    n_samples = 0

    for chunk_idx in cfg.chunk_order:
        # load data
        dataset = torch.load(f"{cfg.dataset_folder}/{chunk_idx}.pt", map_location="cpu")
        ndx_dataset = DatasetWithIndex(dataset)

        loader = torch.utils.data.DataLoader(
            ndx_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
        )

        bar = tqdm.tqdm(total=dataset.shape[0])
        
        c_totals = torch.zeros(model.decoder.shape[0])
        worst_indices = WorstIndices(n_features)
    
        for batch_idx, (ndxs, x) in enumerate(loader):
            optimizer.zero_grad()
            x = x.to(rank, dtype=torch.float32)
            loss, mse, _, c, mse_losses = model(x)
            loss.backward()

            n_nonzero = (c > 0).sum(dim=-1).float().mean()

            n_samples += x.shape[0] * world_size

            optimizer.step()
            scheduler.step()

            # add the feature activations to the total
            c_totals += c.detach().cpu().sum(dim=0)
            
            # add losses and indices to the list
            for loss, ndx in zip(mse_losses, ndxs):
                worst_indices.update(ndx.item(), loss.item())
                            
            if rank == 0:
                bar.update(x.shape[0] * world_size)
                wandb.log({
                    "loss": loss.item(),
                    "mse": mse.item(),
                    "n_samples": n_samples,
                    "center_norm": torch.norm(model.centering).item(),
                    "n_nonzero": n_nonzero.item(),
                    
                })
                
   
        if cfg.reinit and chunk_idx % 10 == 0:
            with torch.no_grad():
                n_replace = (c_totals == 0).sum().item()
                replace_ndxs = torch.where(c_totals == 0)[0]
                print(f"Replacing {n_replace} dictionary elements")
                
                # Get the dataset indices of the points which were modelled the worst
                worst_example_indices = worst_indices.get_worst(n_replace)
                worst_vectors = dataset[worst_example_indices].to(torch.float32).to(rank)
                if worst_vectors.dim() == 1: # occurs when n_replace = 1
                    worst_vectors = worst_vectors.unsqueeze(0)
                    
                # average encoder norms
                av_encoder_norms = torch.norm(model.encoder, dim=0).mean()
                encoder_norm_ratio = 0.2
                
                model.encoder[:, replace_ndxs] = worst_vectors.T * encoder_norm_ratio / av_encoder_norms
                
                # reset the adam state for the new parameters
                optimizer.state[model.decoder]['exp_avg'][replace_ndxs] = 0
                optimizer.state[model.decoder]['exp_avg_sq'][replace_ndxs] = 0
                
                optimizer.state[model.encoder]['exp_avg'][:, replace_ndxs] = 0
                optimizer.state[model.encoder]['exp_avg_sq'][:, replace_ndxs] = 0
                
                optimizer.state[model.threshold]['exp_avg'][replace_ndxs] = 0
                optimizer.state[model.threshold]['exp_avg_sq'][replace_ndxs] = 0
                
            wandb.log({
                "n_dead_feats": n_replace,
            })          

    cleanup()
        

def process_main(rank, cfg, world_size):
    setup(rank, world_size)

    if rank == 0:
        run = wandb.init(
            reinit=True,
            project="sparse coding",
            config=dict(cfg),
            name=cfg.wandb_run_name,
            entity="sparse_coding",
        )

    # Create model and move it to GPU.
    model = SAE(1024, 16384, 1e-3)
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(ddp_model.parameters(), lr=cfg.lr)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

    torch.manual_seed(cfg.seed)

    n_samples = 0

    for chunk_idx in cfg.chunk_order:
        print("starting chunk")
        # load data
        dataset = torch.load(f"{cfg.dataset_folder}/{chunk_idx}.pt", map_location="cpu")

        if rank == 0:
            print(dataset.shape[0] // (world_size * cfg.batch_size))

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True
        )

        bar = tqdm.tqdm(total=dataset.shape[0])
    
        for batch_idx, x in enumerate(loader):
            optimizer.zero_grad()
            x = x.to(rank, dtype=torch.float32)
            loss, mse, _, c, _ = ddp_model(x)
            loss.backward()

            n_nonzero = (c > 0).sum(dim=-1).float().mean()

            n_samples += x.shape[0] * world_size

            if rank == 0:
                bar.update(x.shape[0] * world_size)
                wandb.log({
                    "loss": loss.item(),
                    "mse": mse.item(),
                    "n_samples": n_samples,
                    "center_norm": torch.norm(model.centering).item(),
                    "n_nonzero": n_nonzero.item(),
                })

            optimizer.step()
            scheduler.step()
        
        if rank == 0:
            torch.save(model.state_dict(), f"{cfg.output_dir}/sae_{chunk_idx}.pt")

    cleanup()

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
@dataclass
class HugeBatchArgs(BaseArgs):
    dataset_folder: str = "activation_data/layer_12"
    output_dir: str = "huge_batch_size"
    batch_Size: int = 256
    num_workers: int = 4
    chunk_order: List[int] = [0]
    seed: int = 0
    lr: float = 1e-3
    reinit: bool = False

def main():
    cfg = HugeBatchArgs()
    os.makedirs(cfg.output_dir, exist_ok=True)

    world_size = torch.cuda.device_count()
    mp.spawn(process_main, args=(cfg, world_size,), nprocs=world_size, join=True)

def single_gpu_main():
    cfg = HugeBatchArgs()
    os.makedirs(cfg.output_dir, exist_ok=True)

    secrets = json.load(open("secrets.json", "r"))
    wandb.login(key=secrets["wandb_key"])

    lr_values = [1e-4]

    for lr_value in lr_values:
        cfg.lr = lr_value
        cfg.wandb_run_name = f"lr_{lr_value:.2e}_decay"
        process_main(0, cfg, 1)

@dataclass
class HugeReinitArgs(BaseArgs):
    dataset_folder: str = "owtchunks_pythia70m_l3_mlp"
    output_dir: str = "huge_batch_size"
    batch_size: int = 2048
    num_workers: int = 4
    seed: int = 0
    lr: float = 1e-3
    reinit: bool = True
    layer_loc: str = "mlp"
    layer: int = 3
    model_name: str = "pythia-70m-deduped"
    dataset_name: str = "openwebtext"
    n_chunks: int = 100
    device: str = "cuda:0"
    chunk_size_gb: int = 2
    center_dataset: bool = False

def single_gpu_reinit():
    cfg = HugeReinitArgs()
    os.makedirs(cfg.output_dir, exist_ok=True)

    make_dataset(cfg)
    cfg.chunk_order = list(range(cfg.n_chunks))

    secrets = json.load(open("secrets.json", "r"))
    wandb.login(key=secrets["wandb_key"])

    lr_values = [1e-3]

    for lr_value in lr_values:
        cfg.lr = lr_value
        cfg.wandb_run_name = f"lr_{lr_value:.2e}_decay_reinit"
        process_reinit(0, cfg, 1)

def make_dataset(cfg):
    cfg.activation_width = get_activation_size(cfg.model_name, cfg.layer_loc)

    if not os.path.exists(cfg.dataset_folder) or len(os.listdir(cfg.dataset_folder)) < cfg.n_chunks:
        print(f"Activations in {cfg.dataset_folder} do not exist, creating them")
        transformer, tokenizer = get_model(cfg)
        n_datapoints = setup_data(
            tokenizer,
            transformer,
            dataset_name=cfg.dataset_name,
            dataset_folder=cfg.dataset_folder,
            layer=cfg.layer,
            layer_loc=cfg.layer_loc,
            n_chunks=cfg.n_chunks,
            device=cfg.device,
            chunk_size_gb=cfg.chunk_size_gb,
            center_dataset=cfg.center_dataset,
        )


if __name__ == "__main__":
    single_gpu_reinit()