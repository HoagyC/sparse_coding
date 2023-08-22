from datetime import datetime

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
import torch
from torchtyping import TensorType

from autoencoders.learned_dict import LearnedDict
from autoencoders.topk_encoder import TopKLearnedDict

_n_samples, _activation_size = None, None

class ICAEncoder(LearnedDict):
    def __init__(self, activation_size, n_components: int = 0):
        self.activation_size = activation_size
        if not n_components:
            self.n_feats = activation_size
        else:
            self.n_feats = n_components
        self.ica = FastICA()
        self.scaler = StandardScaler()
    
    def to_device(self, device):
        pass
    
    def encode(self, x):
        assert x.shape[1] == self.activation_size
        x_standardized = self.scaler.transform(x.cpu().numpy().astype(np.float64))
        c = self.ica.transform(x_standardized)
        return torch.tensor(c, device=x.device)
        
    def train(self, dataset: TensorType["_n_samples", "_activation_size"]):
        assert dataset.shape[1] == self.activation_size
        print(f"Fitting ICA on {dataset.shape[0]} activations")
        # Scale the data
        dataset_rescaled = self.scaler.fit_transform(dataset.cpu().numpy().astype(np.float64))
        ica_start = datetime.now()
        output = self.ica.fit_transform(dataset_rescaled) # 1GB of activations takes about 15m
        print(f"ICA fit in {datetime.now() - ica_start}")
        return output


    def get_learned_dict(self):
        return torch.tensor(self.ica.components_, dtype=torch.float32)
    
    def to_topk_dict(self, sparsity):
        positives = self.ica.components_.copy()
        negatives = -positives
        components = np.concatenate([positives, negatives], axis=0)
        return TopKLearnedDict(components, sparsity)