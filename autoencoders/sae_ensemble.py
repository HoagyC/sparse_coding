import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchopt

from autoencoders.learned_dict import (LearnedDict, ReverseSAE, TiedSAE,
                                       UntiedSAE)


class FunctionalSAE:
    @staticmethod
    def init(
        activation_size,
        n_dict_components,
        l1_alpha,
        bias_decay=0.0,
        device=None,
        dtype=None,
    ):
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
    def init(
        activation_size,
        n_dict_components,
        l1_alpha,
        bias_decay=0.0,
        device=None,
        dtype=None,
    ):
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
        return TiedSAE(params["encoder"], params["encoder_bias"], norm_encoder=True)

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


class FunctionalThresholdingSAE:
    @staticmethod
    def init(activation_size, n_dict_components, l1_alpha, device=None, dtype=None):
        params = {}
        buffers = {}

        params["encoder"] = torch.empty((n_dict_components, activation_size), device=device, dtype=dtype)
        nn.init.xavier_uniform_(params["encoder"])

        params["activation_scale"] = torch.empty((n_dict_components,), device=device, dtype=dtype)
        params["activation_gain"] = torch.empty((n_dict_components,), device=device, dtype=dtype)
        nn.init.ones_(params["activation_scale"])
        nn.init.zeros_(params["activation_gain"])

        buffers["l1_alpha"] = torch.tensor(l1_alpha, device=device, dtype=dtype)

        return params, buffers

    @staticmethod
    def encode(params, batch, learned_dict):
        c = torch.einsum("nd,bd->bn", learned_dict, batch)

        a_sq = params["activation_scale"].pow(2)
        c = (c + params["activation_gain"]) / torch.clamp(a_sq, 1e-8)
        c = F.relu6(60 * (c - 0.9)) / 6 + F.relu(c - 1)
        c = c * a_sq

        return c

    @staticmethod
    def loss(params, buffers, batch):
        dict_norms = torch.norm(params["encoder"], 2, dim=-1)
        learned_dict = params["encoder"] / torch.clamp(dict_norms, 1e-8)[:, None]

        c = FunctionalThresholdingSAE.encode(params, batch, learned_dict)

        x_hat = torch.einsum("nd,bn->bd", learned_dict, c)

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

    @staticmethod
    def to_learned_dict(params, buffers):
        return ThresholdingSAE(params)


class ThresholdingSAE(LearnedDict):
    def __init__(self, params):
        self.params = params

    def get_learned_dict(self):
        dict_norms = torch.norm(self.params["encoder"], 2, dim=-1)
        return self.params["encoder"] / torch.clamp(dict_norms, 1e-8)[:, None]

    def encode(self, batch):
        c = FunctionalThresholdingSAE.encode(self.params, batch, self.get_learned_dict())
        return c

    def to_device(self, device):
        self.params = {k: v.to(device) for k, v in self.params.items()}


# allows stacking between different dict sizes
class FunctionalMaskedTiedSAE:
    @staticmethod
    def init(
        activation_size,
        n_dict_components,
        n_components_stack,
        l1_alpha,
        bias_decay=0.0,
        device=None,
        dtype=None,
    ):
        params = {}
        buffers = {}

        params["encoder"] = torch.empty((n_components_stack, activation_size), device=device, dtype=dtype)
        nn.init.xavier_uniform_(params["encoder"])

        params["encoder_bias"] = torch.empty((n_components_stack,), device=device, dtype=dtype)
        nn.init.zeros_(params["encoder_bias"])

        buffers["l1_alpha"] = torch.tensor(l1_alpha, device=device, dtype=dtype)
        buffers["bias_decay"] = torch.tensor(bias_decay, device=device, dtype=dtype)
        buffers["dict_size"] = torch.tensor(n_dict_components, device=device, dtype=torch.long)
        buffers["coef_mask"] = torch.ones(n_components_stack, device=device, dtype=torch.bool)
        buffers["coef_mask"][:n_dict_components] = False

        return params, buffers

    @staticmethod
    def to_learned_dict(params, buffers):
        n_components = buffers["dict_size"].item()
        return TiedSAE(
            params["encoder"][:n_components],
            params["encoder_bias"][:n_components],
            norm_encoder=True,
        )

    @staticmethod
    def loss(params, buffers, batch):
        decoder_norms = torch.norm(params["encoder"], 2, dim=-1)
        learned_dict = params["encoder"] / torch.clamp(decoder_norms, 1e-8)[:, None]

        c = torch.einsum("nd,bd->bn", learned_dict, batch)
        c = c + params["encoder_bias"]
        c = torch.clamp(c, min=0.0)

        # fill unused coefficients with zeros
        c.masked_fill_(buffers["coef_mask"], 0.0)

        x_hat = torch.einsum("nd,bn->bd", learned_dict, c)

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


# allows stacking between different dict sizes
class FunctionalMaskedSAE:
    @staticmethod
    def init(
        activation_size,
        n_dict_components,
        n_components_stack,
        l1_alpha,
        bias_decay=0.0,
        device=None,
        dtype=None,
    ):
        params = {}
        buffers = {}

        params["encoder"] = torch.empty((n_components_stack, activation_size), device=device, dtype=dtype)
        nn.init.xavier_uniform_(params["encoder"])

        params["encoder_bias"] = torch.empty((n_components_stack,), device=device, dtype=dtype)
        nn.init.zeros_(params["encoder_bias"])

        params["decoder"] = torch.empty((n_components_stack, activation_size), device=device, dtype=dtype)
        nn.init.xavier_uniform_(params["decoder"])

        buffers["l1_alpha"] = torch.tensor(l1_alpha, device=device, dtype=dtype)
        buffers["bias_decay"] = torch.tensor(bias_decay, device=device, dtype=dtype)
        buffers["dict_size"] = torch.tensor(n_dict_components, device=device, dtype=torch.long)
        buffers["coef_mask"] = torch.ones(n_components_stack, device=device, dtype=torch.bool)
        buffers["coef_mask"][:n_dict_components] = False

        return params, buffers

    @staticmethod
    def to_learned_dict(params, buffers):
        n_components = buffers["dict_size"].item()
        return UntiedSAE(
            params["encoder"][:n_components],
            params["decoder"][:n_components],
            params["encoder_bias"][:n_components],
        )

    @staticmethod
    def loss(params, buffers, batch):
        decoder_norms = torch.norm(params["decoder"], 2, dim=-1)
        learned_dict = params["decoder"] / torch.clamp(decoder_norms, 1e-8)[:, None]

        c = torch.einsum("nd,bd->bn", params["encoder"], batch)
        c = c + params["encoder_bias"]
        c = torch.clamp(c, min=0.0)

        # fill unused coefficients with zeros
        c.masked_fill_(buffers["coef_mask"], 0.0)

        x_hat = torch.einsum("nd,bn->bd", learned_dict, c)

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


class FunctionalReverseSAE:
    @staticmethod
    def init(
        activation_size,
        n_dict_components,
        l1_alpha,
        bias_decay=0.0,
        device=None,
        dtype=None,
    ):
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
        return ReverseSAE(params["encoder"], params["encoder_bias"], norm_encoder=True)

    @staticmethod
    def loss(params, buffers, batch):
        decoder_norms = torch.norm(params["encoder"], 2, dim=-1)
        learned_dict = params["encoder"] / torch.clamp(decoder_norms, 1e-8)[:, None]

        c = torch.einsum("nd,bd->bn", learned_dict, batch)
        c = c + params["encoder_bias"]
        c = torch.clamp(c, min=0.0)
        feat_is_on = c > 0.0
        c[feat_is_on] = c[feat_is_on] - params["encoder_bias"].repeat(batch.shape[0], 1)[feat_is_on]

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
