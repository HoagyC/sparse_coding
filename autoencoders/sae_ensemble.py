import torch
import torch.nn as nn
import torch.nn.functional as F

import torchopt

import copy

from autoencoders.learned_dict import LearnedDict, UntiedSAE, TiedSAE

class FunctionalSAE:
    @staticmethod
    def init(activation_size, n_dict_components, l1_alpha, bias_decay=0.0, device=None, dtype=None):
        params = {}
        buffers = {}

        params["encoder"] = torch.empty((n_dict_components, activation_size), device=device, dtype=dtype)
        nn.init.xavier_uniform_(params["encoder"])

        params["encoder_bias"] = torch.empty((n_dict_components,), device=device, dtype=dtype)
        nn.init.zeros_(params["encoder_bias"])

        params["decoder"] = torch.empty((n_dict_components, activation_size), device=device, dtype=dtype)
        nn.init.xavier_uniform_(params["decoder"])

        buffers["l1_alpha"] = torch.tensor(l1_alpha, device=device, dtype=dtype)
        buffers["bias_decay"] = torch.tensor(bias_decay, device=device, dtype=dtype)

        return params, buffers

    @staticmethod
    def to_learned_dict(params, buffers):
        return UntiedSAE(params["encoder"], params["decoder"], params["encoder_bias"])

    @staticmethod
    def encode(params, buffers, batch):
        c = torch.einsum("nd,bd->bn", params["encoder"], batch)
        c = c + params["encoder_bias"]
        c = torch.clamp(c, min=0.0)

        return c
    
    @staticmethod
    def loss(params, buffers, batch):
        c = torch.einsum("nd,bd->bn", params["encoder"], batch)
        c = c + params["encoder_bias"]
        c = torch.clamp(c, min=0.0)

        decoder_norms = torch.norm(params["decoder"], 2, dim=-1)
        learned_dict = params["decoder"] / torch.clamp(decoder_norms, 1e-8)[:, None]

        x_hat = torch.einsum("nd,bn->bd", learned_dict, c)

        l_reconstruction = (x_hat - batch).pow(2).mean()
        l_l1 = buffers["l1_alpha"] * torch.norm(c, 1, dim=-1).mean()
        l_bias_decay = buffers["bias_decay"] * torch.norm(params["encoder_bias"], 2)
        
        loss_data = {
            "loss": l_reconstruction + l_l1 + l_bias_decay,
            "l_reconstruction": l_reconstruction,
            "l_l1": l_l1,
            "l_bias_decay": l_bias_decay,
        }

        aux_data = {
            "c": c,
        }

        return l_reconstruction + l_l1 + l_bias_decay, (loss_data, aux_data)

class FunctionalTiedSAE:
    @staticmethod
    def init(activation_size, n_dict_components, l1_alpha, bias_decay=0.0, device=None, dtype=None):
        params = {}
        buffers = {}

        params["encoder"] = torch.empty((n_dict_components, activation_size), device=device, dtype=dtype)
        nn.init.xavier_uniform_(params["encoder"])

        params["encoder_bias"] = torch.empty((n_dict_components,), device=device, dtype=dtype)
        nn.init.zeros_(params["encoder_bias"])

        buffers["l1_alpha"] = torch.tensor(l1_alpha, device=device, dtype=dtype)
        buffers["bias_decay"] = torch.tensor(bias_decay, device=device, dtype=dtype)

        return params, buffers

    @staticmethod
    def to_learned_dict(params, buffers):
        return TiedSAE(params["encoder"], params["encoder_bias"])

    @staticmethod
    def loss(params, buffers, batch):
        decoder_norms = torch.norm(params["encoder"], 2, dim=-1)
        learned_dict = params["encoder"] / torch.clamp(decoder_norms, 1e-8)[:, None]

        c = torch.einsum("nd,bd->bn", learned_dict, batch)
        c = c + params["encoder_bias"]
        c = torch.clamp(c, min=0.0)

        x_hat = torch.einsum("nd,bn->bd", learned_dict, c)

        l_reconstruction = (x_hat - batch).pow(2).mean()
        l_l1 = buffers["l1_alpha"] * torch.norm(c, 1, dim=-1).mean()
        l_bias_decay = buffers["bias_decay"] * torch.norm(params["encoder_bias"], 2)
        
        loss_data = {
            "loss": l_reconstruction + l_l1 + l_bias_decay,
            "l_reconstruction": l_reconstruction,
            "l_l1": l_l1,
            "l_bias_decay": l_bias_decay,
        }

        aux_data = {
            "c": c,
        }

        return l_reconstruction + l_l1 + l_bias_decay, (loss_data, aux_data)