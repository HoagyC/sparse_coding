import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchopt
from torch.func import functional_call, stack_module_state


def affine(x, weight, bias):
    return torch.einsum("ij,bj->bi", weight, x) + bias


class FFLayer:
    @staticmethod
    def init(input_size, output_size, device=None, dtype=None):
        params = {}
        params["weight"] = torch.empty((output_size, input_size), device=device, dtype=dtype)
        nn.init.xavier_uniform_(params["weight"])

        params["bias"] = torch.empty((output_size,), device=device, dtype=dtype)
        nn.init.zeros_(params["bias"])

        return params

    @staticmethod
    def forward(params, x):
        return torch.clamp(affine(x, params["weight"], params["bias"]), min=0.0)


class SemiLinearSAE:
    @staticmethod
    def init(
        activation_size,
        n_dict_components,
        l1_alpha,
        device=None,
        dtype=None,
        hidden_size=None,
    ):
        params = {}
        buffers = {}

        if hidden_size is None:
            hidden_size = n_dict_components

        params["encoder_layers"] = [
            FFLayer.init(activation_size, hidden_size, device=device, dtype=dtype),
            FFLayer.init(hidden_size, n_dict_components, device=device, dtype=dtype),
        ]

        params["decoder"] = torch.empty((n_dict_components, activation_size), device=device, dtype=dtype)
        nn.init.xavier_uniform_(params["decoder"])

        buffers["l1_alpha"] = torch.tensor(l1_alpha, device=device, dtype=dtype)

        return params, buffers

    @staticmethod
    def loss(params, buffers, batch):
        c = batch
        for layer in params["encoder_layers"]:
            c = FFLayer.forward(layer, c)

        decoder_norms = torch.norm(params["decoder"], 2, dim=-1)
        normed_weights = params["decoder"] / torch.clamp(decoder_norms, 1e-8)[:, None]

        x_hat = torch.einsum("nd,bn->bd", normed_weights, c)

        l_reconstruction = (x_hat - batch).pow(2).mean()
        l_l1 = buffers["l1_alpha"] * torch.norm(c, 1, dim=-1).mean()

        loss_data = {
            "loss": l_reconstruction + l_l1,
            "l_reconstruction": l_reconstruction,
            "l_l1": l_l1,
        }

        aux_data = {
            "c": c,
        }

        return l_reconstruction + l_l1, (loss_data, aux_data)
