# def activation_NMF(dataset, n_activations):
#     nmf = NMF()
#     print(f"Fitting NMF on {n_activations} activations")
#     nmf_start = datetime.now()
#     data = next(iter(dataset))[0].cpu().numpy() # 1GB of activations takes an unknown but long time
#     # NMF doesn't support negative values, so shift the data to be positive
#     data -= data.min()
#     nmf.fit(data)
#     print(f"NMF fit in {datetime.now() - nmf_start}")
#     return nmf


from datetime import datetime

import numpy as np
import torch
from sklearn.decomposition import NMF
from torchtyping import TensorType

from typing import Tuple

from autoencoders.learned_dict import LearnedDict
from autoencoders.topk_encoder import TopKLearnedDict

_n_samples, _activation_size = (
    None, None
) # type: Tuple[None, None]


class NMFEncoder(LearnedDict):
    def __init__(self, activation_size, n_components=0, shift=0.0):
        self.activation_size = activation_size
        if not n_components:
            self.n_feats = activation_size
        else:
            self.n_feats = n_components
        self.nmf = NMF()
        self.shift = shift

    def to_device(self, device):
        pass

    def encode(self, x):
        if torch.min(x) < self.shift:
            print("Warning: data has values below expected minumum for NMF. This may cause errors.")
        x -= self.shift
        x.clamp_(min=0.0)
        c = self.nmf.transform(x.cpu().numpy().astype(np.float64))
        return torch.tensor(c, device=x.device)

    def train(self, dataset: TensorType["_n_samples", "_activation_size"]):
        if torch.min(dataset) < self.shift:
            self.shift = torch.min(dataset)
        dataset -= self.shift
        assert dataset.shape[1] == self.activation_size
        print(f"Fitting NMF on {dataset.shape[0]} activations")
        nmf_start = datetime.now()
        self.nmf.fit(dataset.cpu().numpy())  # 1GB of activations takes about 15m
        print(f"NMF fit in {datetime.now() - nmf_start}")

    # WARNING, you can't get the proper coefficient matrix H just by multiplying by the learned dictionary W
    def get_learned_dict(self):
        return torch.tensor(self.nmf.components_, dtype=torch.float32)

    def to_topk_dict(self, sparsity):
        return TopKLearnedDict(self.get_learned_dict(), sparsity)
