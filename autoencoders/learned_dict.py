from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn
from torchtyping import TensorType

from typing import Tuple

from autoencoders.ensemble import DictSignature

_n_dict_components, _activation_size, _batch_size = (
    None, None, None
 ) # type: Tuple[None, None, None]

class LearnedDict(ABC):
    n_feats: int
    activation_size: int

    @abstractmethod
    def get_learned_dict(self) -> TensorType["_n_dict_components", "_activation_size"]:
        pass

    @abstractmethod
    def encode(self, batch: TensorType["_batch_size", "_activation_size"]) -> TensorType["_batch_size", "_n_dict_components"]:
        pass

    @abstractmethod
    def to_device(self, device):
        pass

    def decode(self, code: TensorType["_batch_size", "_n_dict_components"]) -> TensorType["_batch_size", "_activation_size"]:
        learned_dict = self.get_learned_dict()
        x_hat = torch.einsum("nd,bn->bd", learned_dict, code)
        return x_hat

    def center(self, batch: TensorType["_batch_size", "_activation_size"]) -> TensorType["_batch_size", "_activation_size"]:
        # overloadable method to center the batch for the (otherwise) linear model
        return batch
    
    def uncenter(self, batch: TensorType["_batch_size", "_activation_size"]) -> TensorType["_batch_size", "_activation_size"]:
        # inverse of `center`
        return batch

    def predict(self, batch: TensorType["_batch_size", "_activation_size"]) -> TensorType["_batch_size", "_activation_size"]:
        batch_centered = self.center(batch)
        c = self.encode(batch_centered)
        x_hat_centered = self.decode(c)
        x_hat = self.uncenter(x_hat_centered)
        return x_hat

    def n_dict_components(self):
        return self.get_learned_dict().shape[0]


class Identity(LearnedDict):
    def __init__(self, activation_size, device=None):
        self.n_feats = activation_size
        self.activation_size = activation_size
        self.device = "cpu" if device is None else device

    def get_learned_dict(self):
        return torch.eye(self.n_feats, device=self.device)

    def encode(self, batch):
        return batch

    def to_device(self, device):
        self.device = device

class IdentityPositive(LearnedDict):
    def __init__(self, activation_size, device=None):
        self.n_feats = activation_size
        self.activation_size = activation_size
        self.device = "cpu" if device is None else device

    def get_learned_dict(self):
        return torch.cat([torch.eye(self.n_feats, device=self.device), -torch.eye(self.n_feats, device=self.device)], dim=0)

    def encode(self, batch):
        return torch.clamp(torch.cat([batch, -batch], dim=-1), min=0.0)

    def to_device(self, device):
        self.device = device

class IdentityReLU(LearnedDict):
    def __init__(self, activation_size, bias: Optional[torch.Tensor] = None):
        self.n_feats = activation_size
        self.activation_size = activation_size
        if bias:
            self.bias = bias
        else:
            self.bias = torch.zeros(activation_size)
        assert self.bias.shape == (activation_size,)

    def get_learned_dict(self):
        return torch.eye(self.n_feats)

    def encode(self, batch):
        return torch.clamp(batch + self.bias, min=0.0)

    def to_device(self, device):
        self.bias = self.bias.to(device)


class RandomDict(LearnedDict):
    def __init__(self, activation_size, n_feats=None):
        if not n_feats:
            n_feats = activation_size
        self.n_feats = n_feats
        self.activation_size = activation_size
        self.encoder = torch.randn(n_feats, activation_size)
        self.encoder_bias = torch.zeros(n_feats)

    def get_learned_dict(self):
        return self.encoder

    def encode(self, batch):
        c = torch.einsum("nd,bd->bn", self.encoder, batch)
        c = c + self.encoder_bias
        c = torch.clamp(c, min=0.0)
        return c

    def to_device(self, device):
        self.encoder = self.encoder.to(device)
        self.encoder_bias = self.encoder_bias.to(device)


class UntiedSAE(LearnedDict):
    def __init__(self, encoder, decoder, encoder_bias):
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_bias = encoder_bias
        self.n_feats, self.activation_size = self.encoder.shape

    def get_learned_dict(self):
        norms = torch.norm(self.decoder, 2, dim=-1)
        return self.decoder / torch.clamp(norms, 1e-8)[:, None]

    def to_device(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder_bias = self.encoder_bias.to(device)

    def encode(self, batch):
        c = torch.einsum("nd,bd->bn", self.encoder, batch)
        c = c + self.encoder_bias
        c = torch.clamp(c, min=0.0)
        return c


class TiedSAE(LearnedDict):
    def __init__(self, encoder, encoder_bias, centering=(None, None, None), norm_encoder=True):
        self.encoder = encoder
        self.encoder_bias = encoder_bias
        self.norm_encoder = norm_encoder
        self.n_feats, self.activation_size = self.encoder.shape

        center_trans, center_rot, center_scale = centering

        if center_trans is None:
            center_trans = torch.zeros(self.activation_size)
        
        if center_rot is None:
            center_rot = torch.eye(self.activation_size)
            print(center_rot)
        
        if center_scale is None:
            center_scale = torch.ones(self.activation_size)
        
        self.center_trans = center_trans
        self.center_rot = center_rot
        self.center_scale = center_scale

    def initialize_missing(self):
        if not hasattr(self, "center_trans"):
            self.center_trans = torch.zeros(self.activation_size, device=self.encoder.device)
        
        if not hasattr(self, "center_rot"):
            self.center_rot = torch.eye(self.activation_size, device=self.encoder.device)
        
        if not hasattr(self, "center_scale"):
            self.center_scale = torch.ones(self.activation_size, device=self.encoder.device)

    def center(self, batch):
        return torch.einsum("cu,bu->bc", self.center_rot, batch - self.center_trans[None, :]) * self.center_scale[None, :]
    
    def uncenter(self, batch):
        return torch.einsum("cu,bc->bu", self.center_rot, batch / self.center_scale[None, :]) + self.center_trans[None, :]

    def get_learned_dict(self):
        norms = torch.norm(self.encoder, 2, dim=-1)
        return self.encoder / torch.clamp(norms, 1e-8)[:, None]

    def to_device(self, device):
        self.initialize_missing()

        self.encoder = self.encoder.to(device)
        self.encoder_bias = self.encoder_bias.to(device)

        self.center_trans = self.center_trans.to(device)
        self.center_rot = self.center_rot.to(device)
        self.center_scale = self.center_scale.to(device)

    def encode(self, batch):
        if self.norm_encoder:
            norms = torch.norm(self.encoder, 2, dim=-1)
            encoder = self.encoder / torch.clamp(norms, 1e-8)[:, None]
        else:
            encoder = self.encoder

        c = torch.einsum("nd,bd->bn", encoder, batch)
        c = c + self.encoder_bias
        c = torch.clamp(c, min=0.0)
        return c


class ReverseSAE(LearnedDict):
    """This is the same as a tied SAE, but we reverse the bias if the feature activation is non-zero before the decoder matrix"""

    def __init__(self, encoder, encoder_bias, norm_encoder=False):
        self.encoder = encoder
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
        if self.norm_encoder:
            norms = torch.norm(self.encoder, 2, dim=-1)
            encoder = self.encoder / torch.clamp(norms, 1e-8)[:, None]
        else:
            encoder = self.encoder

        c = torch.einsum("nd,bd->bn", encoder, batch)
        c = c + self.encoder_bias
        c = torch.clamp(c, min=0.0)
        return c

    def decode(self, c):
        if self.norm_encoder:
            norms = torch.norm(self.encoder, 2, dim=-1)
            encoder = self.encoder / torch.clamp(norms, 1e-8)[:, None]
        else:
            encoder = self.encoder

        feat_is_on = c > 0.0
        c[feat_is_on] = c[feat_is_on] - self.encoder_bias.repeat(c.shape[0], 1)[feat_is_on]
        x_hat = torch.einsum("dn,bn->bd", encoder, c)
        return x_hat


class AddedNoise(LearnedDict):
    def __init__(self, noise_mag, activation_size, device=None):
        self.noise_mag = noise_mag
        self.activation_size = activation_size
        self.device = "cpu" if device is None else device

    def get_learned_dict(self):
        return torch.eye(self.activation_size, device=self.device)

    def to_device(self, device):
        self.device = device

    def encode(self, batch):
        noise = torch.randn(batch.shape[0], self.activation_size, device=batch.device) * self.noise_mag
        return batch + noise


class Rotation(LearnedDict):
    def __init__(self, matrix, device=None):
        self.matrix = matrix
        self.activation_size = matrix.shape[0]
        self.device = "cpu" if device is None else device

        self.matrix = self.matrix.to(self.device)

    def get_learned_dict(self):
        return self.matrix

    def to_device(self, device):
        self.matrix = self.matrix.to(device)
        self.device = device

    def encode(self, batch):
        return torch.einsum("nd,bd->bn", self.matrix, batch)
