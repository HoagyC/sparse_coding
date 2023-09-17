import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from autoencoders.learned_dict import LearnedDict

import wandb
import json
import tqdm

class SAE(nn.Module, LearnedDict):
    def __init__(self, input_size, latent_size, l1_alpha):
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
        c = F.relu(torch.einsum("dn,bd->bn", self.encoder, x) + self.threshold)

        return c

    def forward(self, x):
        normed_dict = self.get_learned_dict()

        c = self.encode(x)
        x_hat = torch.einsum("nd,bn->bd", normed_dict, c) + self.centering

        mse_loss = (x - x_hat).pow(2).mean()
        sparsity_loss = self.l1_alpha * torch.norm(c, 1, dim=-1).mean()
        return mse_loss + sparsity_loss, mse_loss, sparsity_loss, c

    def to_device(self, device):
        self.to(device)

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
            loss, mse, _, c = ddp_model(x)
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

def main():
    import argparse
    import os
    import math

    args = {
        "dataset_folder": "activation_data/layer_12",
        "output_dir": "huge_batch_size",
        "batch_size": 256,
        "num_workers": 4,
        "chunk_order": [0],
        "seed": 0,
        "lr": 1e-3,
    }

    os.makedirs(args["output_dir"], exist_ok=True)

    from utils import dotdict

    cfg = dotdict(args)

    world_size = torch.cuda.device_count()
    mp.spawn(process_main, args=(cfg, world_size,), nprocs=world_size, join=True)

def single_gpu_main():
    import argparse
    import os
    import math
    import numpy as np

    args = {
        "dataset_folder": "activation_data/layer_12",
        "output_dir": "huge_batch_size",
        "batch_size": 256,
        "num_workers": 4,
        "chunk_order": [0],
        "seed": 0,
        "lr": 1e-3,
    }

    os.makedirs(args["output_dir"], exist_ok=True)

    from utils import dotdict

    cfg = dotdict(args)

    secrets = json.load(open("secrets.json", "r"))
    wandb.login(key=secrets["wandb_key"])

    lr_values = [1e-4]

    for lr_value in lr_values:
        cfg.lr = lr_value
        cfg.wandb_run_name = f"lr_{lr_value:.2e}_decay"
        process_main(0, cfg, 1)

if __name__ == "__main__":
    single_gpu_main()