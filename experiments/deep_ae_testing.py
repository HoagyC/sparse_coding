import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os
import json
import numpy as np

class ShrinkageLayer(nn.Module):
    def __init__(self, d_activation, d_hidden, d_latent):
        super().__init__()
        
        self.f_in = nn.Linear(d_latent + 2 * d_activation, d_hidden)
        self.f_out = nn.Linear(d_hidden, d_latent)
    
    def forward(self, z, x, x_hat):
        in_vec = torch.cat([z, x, x_hat], dim=-1)
        h = F.gelu(self.f_in(in_vec))
        z = z + self.f_out(h)
        return z

class SparseAutoencoder(nn.Module):
    def __init__(self, d_activation, n_hidden, d_latent):
        super().__init__()
        self.d_activation = d_activation
        self.d_latent = d_latent

        self.encoder_in = nn.Linear(d_activation, d_latent)

        modules = [ShrinkageLayer(d_activation, d_latent * 2, d_latent) for _ in range(n_hidden)]
        #modules = [SimpleShrinkageLayer(d_latent, d_latent * 4) for _ in range(n_hidden)]

        self.dictionary = nn.Parameter(torch.randn(d_latent, d_activation))
        self.bias = nn.Parameter(torch.zeros(d_activation))
    
        self.encoder_modules = nn.ModuleList(modules)

    def encoder(self, x):
        z = F.softplus(self.encoder_in(x))
        for module in self.encoder_modules:
            x_hat = self.decoder(z)
            z = module(z, x, x_hat)
        return z

    def decoder(self, c):
        normed_dict = self.dictionary / torch.linalg.norm(self.dictionary, ord=2, dim=1, keepdim=True)
        return torch.einsum("nd,bn->bd", normed_dict, c) + self.bias[None, :]

    def forward(self, x):
        c = self.encoder(x)
        x_hat = self.decoder(c)
        return x_hat, c
    
    def losses(self, x, c, x_hat, l1_coef):
        reconstr = F.mse_loss(x, x_hat)
        l1_reg = l1_coef * torch.linalg.norm(c, ord=1, dim=-1).mean()
        return reconstr + l1_reg, reconstr, l1_reg

class NonlinearSparseAutoencoder(nn.Module):
    def __init__(self, d_activation, d_hidden, d_latent):
        super().__init__()
        self.d_activation = d_activation
        self.d_latent = d_latent

        self.encoder = nn.Sequential(
            nn.Linear(d_activation, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_latent),
            nn.Softplus(beta=100)
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_activation)
        )
    
    def forward(self, x):
        c = self.encoder(x)
        # scale so that c has unit norm
        c = c / torch.linalg.norm(c, ord=2, dim=-1, keepdim=True)
        x_hat = self.decoder(c)
        return x_hat, c
    
    def losses(self, x, c, x_hat, l1_coef):
        reconstr = F.mse_loss(x, x_hat)
        l1_reg = l1_coef * torch.linalg.norm(c, ord=1, dim=-1).mean()
        return reconstr + l1_reg, reconstr, l1_reg

def l1_schedule(max_l1=1e-3, warmup_steps=1000):
    def schedule(step):
        if step < warmup_steps:
            return max_l1 * step / warmup_steps
        else:
            return max_l1
    return schedule

if __name__ == "__main__":
    activation_folder = "activation_data/layer_2/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NonlinearSparseAutoencoder(512, 1024, 2048).to(device)

    secrets = json.load(open("secrets.json"))
    wandb.login(key=secrets["wandb_key"])
    wandb.init(
        project="sparse coding",
        #config=dict(cfg),
        name="fancy_autoencoder",
        entity="sparse_coding",
    )

    n_epochs = 100
    batch_size = 256

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    schedule = l1_schedule(1e-3, 1000)

    steps = 0

    for epoch in range(n_epochs):
        n_chunks = len(os.listdir(activation_folder))
        chunk_idxs = np.random.permutation(n_chunks)
        for chunk_idx in chunk_idxs:
            print("chunk", chunk_idx, "epoch", epoch)
            chunk = torch.load(activation_folder + f"{chunk_idx}.pt").to(device=device, dtype=torch.float32)

            dataloader = torch.utils.data.DataLoader(chunk, batch_size=batch_size, shuffle=True)

            for batch in dataloader:
                x = batch
                x_hat, c = model(x)

                loss, reconstr, l1_reg = model.losses(x, c, x_hat, schedule(steps))

                sparsity = {}

                fvu = (x - x_hat).pow(2).sum() / (x - x.mean()).pow(2).sum()
                sparsity["1e-5"] = (c > 1e-5).float().sum(dim=-1).mean()
                sparsity["1e-6"] = (c > 1e-6).float().sum(dim=-1).mean()
                sparsity["1e-7"] = (c > 1e-7).float().sum(dim=-1).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                wandb.log({
                    "loss": loss.item(),
                    "reconstr": reconstr.item(),
                    "l1_reg": l1_reg.item(),
                    "fvu": fvu.item(),
                    "sparsity_1e-5": sparsity["1e-5"].item(),
                    "sparsity_1e-6": sparsity["1e-6"].item(),
                    "sparsity_1e-7": sparsity["1e-7"].item(),
                })
            
                steps += 1