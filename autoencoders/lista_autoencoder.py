import torch
import torch.nn.functional as F

from autoencoders.learned_dict import LearnedDict

import optree

class FunctionalResidualDenoisingLayer:
    @staticmethod
    def init(d_hidden, dtype=torch.float32):
        params = {}
        params["weight"] = torch.randn(d_hidden, d_hidden, dtype=dtype) * 0.02

        params["bias"] = torch.randn(d_hidden, dtype=dtype) * 0.02
        return params
    
    @staticmethod
    def forward(params, x):
        x_ = F.gelu(x + params["bias"][None, :])
        x_ = torch.einsum("ij,bj->bi", params["weight"], x)
        return x + x_

class FunctionalResidualDenoisingSAE:
    @staticmethod
    def init(d_activation, n_features, n_hidden_layers, l1_alpha, dtype=torch.float32):
        params = {}
        params["decoder"] = torch.empty(n_features, d_activation, dtype=dtype)
        torch.nn.init.orthogonal_(params["decoder"])

        params["encoder_embedding"] = params["decoder"].clone().T
        params["encoder_layers"] = [
            FunctionalResidualDenoisingLayer.init(n_features, dtype=dtype)
            for _ in range(n_hidden_layers)
        ]
        params["encoder_bias"] = torch.randn(n_features, dtype=dtype) * 0.02

        buffers = {}
        buffers["l1_alpha"] = torch.tensor(l1_alpha, dtype=dtype)

        return params, buffers
    
    @staticmethod
    def encode(params, x):
        x = torch.einsum("ij,bi->bj", params["encoder_embedding"], x)
        for layer in params["encoder_layers"]:
            x = FunctionalResidualDenoisingLayer.forward(layer, x)
        return F.relu(x + params["encoder_bias"][None, :])

    @staticmethod
    def loss(params, buffers, batch):
        decoder_norms = torch.norm(params["decoder"], 2, dim=-1)
        learned_dict = params["decoder"] / torch.clamp(decoder_norms, 1e-8)[:, None]

        c = FunctionalResidualDenoisingSAE.encode(params, batch)

        x_hat = torch.einsum("ij,bi->bj", learned_dict, c)

        l_reconstruction = (x_hat - batch).pow(2).mean()
        l_sparsity = buffers["l1_alpha"] * torch.norm(c, 1, dim=-1).mean()

        loss_data = {
            "loss": l_reconstruction + l_sparsity,
            "l_reconstruction": l_reconstruction,
            "l_l1": l_sparsity
        }
        aux_data = {
            "c": c
        }

        return l_reconstruction + l_sparsity, (loss_data, aux_data)

    @staticmethod
    def to_learned_dict(params, buffers):
        return ResidualDenoisingSAE(params)

    @staticmethod
    def init_lr(n_hidden_layers, lr, lr_encoder=None):
        if lr_encoder is None:
            lr_encoder = lr

        lrs = {"decoder": lr}

        lrs["encoder_embedding"] = lr_encoder
        lrs["encoder_bias"] = lr_encoder
        lrs["encoder_layers"] = []
        for _ in range(n_hidden_layers):
            layer = {"weight": lr, "bias": lr}
            lrs["encoder_layers"].append(layer)

        return lrs

class ResidualDenoisingSAE(LearnedDict):
    def __init__(self, params):
        self.params = params

    def encode(self, x):
        return FunctionalResidualDenoisingSAE.encode(self.params, x)

    def to_device(self, device):
        self.params = optree.tree_map(lambda t: t.to(device=device), self.params)

    def get_learned_dict(self):
        decoder_norms = torch.norm(self.params["decoder"], 2, dim=-1)
        learned_dict = self.params["decoder"] / torch.clamp(decoder_norms, 1e-8)[:, None]

        return learned_dict