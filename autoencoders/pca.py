import torch

from autoencoders.learned_dict import LearnedDict, Rotation
from autoencoders.topk_encoder import TopKLearnedDict

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
    
    def to_learned_dict(self, sparsity):
        eigvals, eigvecs = self.get_pca()
        eigvecs = eigvecs[:, torch.argsort(eigvals, descending=True)].T
        return PCAEncoder(eigvecs, sparsity)
    
    def to_topk_dict(self, sparsity):
        eigvals, eigvecs = self.get_pca()
        eigvecs = eigvecs[:, torch.argsort(eigvals, descending=True)].T
        eigvecs_ = torch.cat([eigvecs, -eigvecs], dim=0)
        return TopKLearnedDict(eigvecs_, sparsity)
    
    def to_rotation_dict(self, n_components):
        return Rotation(self.get_dict()[:n_components])

class PCAEncoder(LearnedDict):
    def __init__(self, pca_dict, sparsity):
        normed_dict = pca_dict / torch.norm(pca_dict, dim=-1)[:, None]
        self.pca_dict = normed_dict
        self.sparsity = sparsity
        self.n_feats, self.activation_size = self.pca_dict.shape
    
    def to_device(self, device):
        self.pca_dict = self.pca_dict.to(device)
    
    def encode(self, x):
        # for every x, find the top-k PCA components

        scores = torch.einsum("ij,bj->bi", self.pca_dict, x)
        topk_idxs = torch.topk(scores.abs(), self.sparsity, dim=-1).indices

        code = torch.zeros_like(scores)
        code.scatter_(dim=-1, index=topk_idxs, src=scores.gather(dim=-1, index=topk_idxs))

        return code
    
    def get_learned_dict(self):
        return self.pca_dict