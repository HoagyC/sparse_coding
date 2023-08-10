import torch
from torch import nn

from abc import ABC, abstractmethod
from torchtyping import TensorType

from autoencoders.ensemble import DictSignature

_n_dict_components, _activation_size, _batch_size = None, None, None

class LearnedDict(ABC):
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

    def predict(self, batch: TensorType["_batch_size", "_activation_size"]) -> TensorType["_batch_size", "_activation_size"]:
        c = self.encode(batch)
        x_hat = self.decode(c)
        return x_hat
    
    def n_dict_components(self):
        return self.get_learned_dict().shape[0]

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