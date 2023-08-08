from typing import Optional

import torch
from torch import nn

from abc import ABC, abstractmethod
from torchtyping import TensorType

from autoencoders.ensemble import DictSignature

_n_dict_components, _activation_size, _batch_size = None, None, None

class LearnedDict(ABC):
    n_feats: int
    activation_size: int
    encoder: TensorType["_n_dict_components", "_activation_size"]
    encoder_bias: TensorType["_n_dict_components"]

    @abstractmethod
    def get_learned_dict(self) -> TensorType["_n_dict_components", "_activation_size"]:
        pass

    @abstractmethod
    def encode(self, batch: TensorType["_batch_size", "_activation_size"]) -> TensorType["_batch_size", "_n_dict_components"]:
        pass
    
    @abstractmethod
    def to_device(self, device):
        pass
    
    def predict(self, batch: TensorType["_batch_size", "_activation_size"]) -> TensorType["_batch_size", "_activation_size"]:
        c = self.encode(batch)
        learned_dict = self.get_learned_dict()
        x_hat = torch.einsum("nd,bn->bd", learned_dict, c)
        return x_hat
    
class Identity(LearnedDict):
    def __init__(self, activation_size):
        self.n_feats = activation_size
        self.activation_size = activation_size
    
    def get_learned_dict(self):
        return torch.eye(self.n_feats)
    
    def encode(self, batch):
        return batch
    
    def to_device(self, device):
        pass

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
        pass

class RandomDict(LearnedDict):
    def __init__(self, activation_size, n_feats = None):
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