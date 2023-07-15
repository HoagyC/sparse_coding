import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.func import stack_module_state, functional_call

import torchopt

import copy

class SAE(nn.Module):
    def __init__(self, activation_size, n_dict_components, l1_coef=0.0, device=None):
        super(SAE, self).__init__()
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.decoder = nn.Parameter(torch.empty((n_dict_components, activation_size), device=self.device))
        # Initialize the decoder weights orthogonally
        nn.init.orthogonal_(self.decoder)

        self.encoder = nn.Sequential(nn.Linear(activation_size, n_dict_components, device=self.device), nn.ReLU())
        self.register_buffer("l1_coef", torch.tensor(l1_coef))

    def forward(self, x):
        c = self.encoder(x)
        
        # can't use this here due to vmap
        #self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)

        normed_weights = nn.functional.normalize(self.decoder, dim=0)

        x_hat = torch.einsum("ij,bi->bj", normed_weights, c)

        l_reconstruction = F.mse_loss(x_hat, x)
        l_l1 = self.l1_coef * torch.norm(c, 1, dim=1).mean()
        
        return l_reconstruction + l_l1

class FunctionalSAE:
    @staticmethod
    def init(activation_size, n_dict_components, l1_alpha, device=None):
        params = {}
        buffers = {}

        params["decoder"] = torch.empty((n_dict_components, activation_size), device=device)
        nn.init.orthogonal_(params["decoder"])

        params["encoder"] = torch.empty((n_dict_components, activation_size), device=device)
        nn.init.orthogonal_(params["encoder"])

        buffers["l1_alpha"] = torch.tensor(l1_alpha, device=device)
    
        return params, buffers
    
    @staticmethod
    def loss(params, buffers, batch):
        c = torch.einsum("nd,bd->bn", params["encoder"], batch)
        c = F.relu(c)

        normed_weights = nn.functional.normalize(params["decoder"], dim=0)

        x_hat = torch.einsum("nd,bn->bd", normed_weights, c)

        l_reconstruction = F.mse_loss(x_hat, batch)
        l_l1 = buffers["l1_alpha"] * torch.norm(c, 1, dim=1).mean()
        
        return l_reconstruction + l_l1