#import run
from utils import *

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

import numpy as np

from transformers import GPT2Tokenizer
from transformer_lens import HookedTransformer
import pickle

from autoencoders.encoder import Encoder

class BatchedPCA():
    def __init__(self, n_dims, device):
        super().__init__()
        self.n_dims = n_dims
        self.device = device

        self.cov = torch.zeros((n_dims, n_dims), device=device)
        self.mean = torch.zeros((n_dims,), device=device)
        self.n_samples = 0
    
    def train_batch(self, activations):
        # activations: (batch_size, n_dims)
        batch_size = activations.shape[0]
        corrected = activations - self.mean.unsqueeze(0)
        new_mean = self.mean + torch.mean(corrected, dim=0) * batch_size / (self.n_samples + batch_size)
        cov_update = torch.einsum("bi,bj->bij", corrected, activations - new_mean.unsqueeze(0)).mean(dim=0)
        self.cov = self.cov * (self.n_samples / (self.n_samples + batch_size)) + cov_update * batch_size / (self.n_samples + batch_size)
        self.mean = new_mean
        self.n_samples += batch_size
    
    def get_pca(self):
        eigvals, eigvecs = torch.linalg.eigh(self.cov)
        return eigvals, eigvecs
    
    def get_dict(self):
        eigvals, eigvecs = self.get_pca()
        eigvecs = eigvecs[:, torch.argsort(eigvals, descending=True)].T
        return eigvecs
    
    def configure_optimizers(self, **kwargs):
        return None

Encoder.register(BatchedPCA)