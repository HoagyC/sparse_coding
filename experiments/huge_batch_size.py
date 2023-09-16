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
        self.dict = nn.Parameter(torch.randn(latent_size, input_size))
        self.encoder = nn.Parameter(torch.randn(input_size, latent_size))
        self.threshold = nn.Parameter(torch.zeros(latent_size))
        self.centering = nn.Parameter(torch.zeros(input_size))
        self.l1_alpha = l1_alpha
    
    def get_learned_dict(self):
        normed_dict = self.dict / self.dict.norm(dim=1, keepdim=True)
        return normed_dict

    def encode(self, x):
        x_centered = x - self.centering
        c = F.relu(torch.einsum("dn,bd->bn", self.encoder, x_centered) + self.threshold)
        
        return c

    def forward(self, x):
        c = self.encode(x)
        x_hat = torch.einsum("nd,bn->bd", self.get_learned_dict(), c) + self.centering

        mse_loss = F.mse_loss(x, x_hat)
        sparsity_loss = self.l1_alpha * c.norm(dim=1).mean()
        return mse_loss + sparsity_loss, mse_loss, sparsity_loss

    def to_device(self, device):
        self.to(device)

def process_main(rank, cfg, world_size):
    setup(rank, world_size)

    if rank == 0:
        secrets = json.load(open("secrets.json"))
        wandb.login(key=secrets["wandb_key"])
        wandb_run_name = "huge_batch_size"
        cfg.wandb_instance = wandb.init(
            project="sparse coding",
            config=dict(cfg),
            name=wandb_run_name,
            entity="sparse_coding",
        )

    # Create model and move it to GPU.
    model = SAE(1024, 16384, 1e-3)
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)

    torch.manual_seed(cfg.seed)

    for chunk_idx in cfg.chunk_order:
        # load data
        dataset = torch.load(f"{cfg.dataset_folder}/{chunk_idx}.pt", map_location="cpu")

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
            loss, mse, _ = ddp_model(x)
            loss.backward()

            if rank == 0:
                bar.update(x.shape[0] * world_size)
                wandb.log({
                    "loss": loss.item(),
                    "mse": mse.item()
                })

            optimizer.step()
        
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

if __name__ == "__main__":
    import argparse
    import os

    args = {
        "dataset_folder": "activation_data/layer_12",
        "output_dir": "huge_batch_size",
        "batch_size": 8192,
        "num_workers": 4,
        "chunk_order": [0, 1, 2, 3, 4, 5, 6, 7],
        "seed": 0
    }

    os.makedirs(args["output_dir"], exist_ok=True)

    from utils import dotdict

    cfg = dotdict(args)

    world_size = torch.cuda.device_count()
    mp.spawn(process_main, args=(cfg, world_size,), nprocs=world_size, join=True)