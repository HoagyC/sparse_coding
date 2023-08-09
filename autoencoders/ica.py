from datetime import datetime

from sklearn.decomposition import FastICA
import torch
from torchtyping import TensorType

from autoencoders.learned_dict import LearnedDict
from autoencoders.topk_encoder import TopKLearnedDict

_n_samples, _activation_size = None, None

class ICAEncoder(LearnedDict):
    def __init__(self, n_components, sparsity):
        self.activation_size = n_components
        self.n_feats = n_components
        self.sparsity = sparsity
        self.ica = FastICA()
    
    def to_device(self, device):
        pass
    
    def encode(self, x):
        c = self.ica.transform(x.cpu().numpy())
        return torch.tensor(c, device=x.device)
        
    def train(self, dataset: TensorType["_n_samples", "_activation_size"]):
        assert dataset.shape[1] == self.activation_size
        print(f"Fitting ICA on {dataset.shape[0]} activations")
        ica_start = datetime.now()
        self.ica.fit(dataset.cpu().numpy()) # 1GB of activations takes about 15m
        print(f"ICA fit in {datetime.now() - ica_start}")


    def get_learned_dict(self):
        return self.ica.components_