import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.func import stack_module_state, functional_call

import torchopt

import copy

class SAE(nn.Module):
    def __init__(self, activation_size, n_dict_components, l1_coef=0.0):
        super(SAE, self).__init__()
        self.decoder = nn.Parameter(torch.empty((n_dict_components, activation_size)))
        # Initialize the decoder weights orthogonally
        nn.init.orthogonal_(self.decoder)

        self.encoder = nn.Sequential(nn.Linear(activation_size, n_dict_components), nn.ReLU())
        self.register_buffer("l1_coef", torch.tensor(l1_coef))
    
    def forward(self, x):
        return self.encoder(x)

    def loss(self, x):
        c = self.encoder(x)
        
        # can't use this here due to vmap
        #self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)

        normed_weights = nn.functional.normalize(self.decoder, dim=0)

        x_hat = torch.einsum("ij,bi->bj", normed_weights, c)

        l_reconstruction = F.mse_loss(x_hat, x)
        l_l1 = self.l1_coef * torch.norm(c, 1, dim=1).mean()
        
        return l_reconstruction + l_l1