import torch
from torch import nn

from autoencoders.learned_dict import LearnedDict, TiedSAE
from autoencoders.ensemble import DictSignature


class TiedPositiveSAE(LearnedDict):
    def __init__(self, encoder, encoder_bias, norm_encoder=False):
        self.encoder = encoder
        self.encoder.data = torch.abs(self.encoder.data)
        self.encoder_bias = encoder_bias
        self.norm_encoder = norm_encoder
        self.n_feats, self.activation_size = self.encoder.shape

    def get_learned_dict(self):
        norms = torch.norm(self.encoder, 2, dim=-1)
        return self.encoder / torch.clamp(norms, 1e-8)[:, None]

    def to_device(self, device):
        self.encoder = self.encoder.to(device)
        self.encoder_bias = self.encoder_bias.to(device)

    def encode(self, batch):
        self.encoder.clamp = torch.clamp(self.encoder, min=0.0)
        if self.norm_encoder:
            norms = torch.norm(self.encoder, 2, dim=-1)
            encoder = self.encoder / torch.clamp(norms, 1e-8)[:, None]
        else:
            encoder = self.encoder

        c = torch.einsum("nd,bd->bn", encoder, batch)
        c = c + self.encoder_bias
        c = torch.clamp(c, min=0.0)
        return c


class UntiedPositiveSAE(LearnedDict):
    def __init__(self, encoder, encoder_bias, decoder, norm_encoder=False):
        self.encoder = encoder
        self.encoder.data = torch.abs(self.encoder.data)
        self.decoder = decoder
        self.encoder_bias = encoder_bias
        self.norm_encoder = norm_encoder
        self.n_feats, self.activation_size = self.encoder.shape

    def get_learned_dict(self):
        norms = torch.norm(self.encoder, 2, dim=-1)
        return self.encoder / torch.clamp(norms, 1e-8)[:, None]

    def to_device(self, device):
        self.encoder = self.encoder.to(device)
        self.encoder_bias = self.encoder_bias.to(device)

    def encode(self, batch):
        self.encoder.clamp = torch.clamp(self.encoder, min=0.0)
        if self.norm_encoder:
            norms = torch.norm(self.encoder, 2, dim=-1)
            encoder = self.encoder / torch.clamp(norms, 1e-8)[:, None]
        else:
            encoder = self.encoder
        c = torch.einsum("nd,bd->bn", self.encoder, batch)
        c = c + self.encoder_bias
        c = torch.clamp(c, min=0.0)
        return c


class FunctionalPositiveTiedSAE(DictSignature):
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
        params["encoder"] = abs(params["encoder"])

        params["encoder_bias"] = torch.empty((n_dict_components,), device=device, dtype=dtype)
        # init at -1 each
        nn.init.constant_(params["encoder_bias"], -1)

        buffers["l1_alpha"] = torch.tensor(l1_alpha, device=device, dtype=dtype)
        buffers["bias_decay"] = torch.tensor(bias_decay, device=device, dtype=dtype)

        return params, buffers

    @staticmethod
    def to_learned_dict(params, buffers):
        return TiedSAE(params["encoder"], params["encoder_bias"], norm_encoder=True)

    @staticmethod
    def loss(params, buffers, batch):
        params["encoder"] = torch.clamp(params["encoder"], min=0.0)
        encoder_norms = torch.norm(params["encoder"], 2, dim=-1)
        learned_dict = params["encoder"] / torch.clamp(encoder_norms, 1e-8)[:, None]

        c = torch.einsum("nd,bd->bn", learned_dict, (batch + 0.18))
        c = c + params["encoder_bias"]
        c = torch.clamp(c, min=0.0)

        x_hat = torch.einsum("nd,bn->bd", learned_dict, c)

        l_reconstruction = ((x_hat - 0.18) - batch).pow(2).mean()
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
