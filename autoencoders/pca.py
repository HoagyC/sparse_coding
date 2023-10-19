import torch

from autoencoders.learned_dict import LearnedDict, Rotation, TiedSAE
from autoencoders.topk_encoder import TopKLearnedDict

def calc_pca(activations, batch_size=512, device="cuda:0"):
    pca = BatchedPCA(activations.shape[1], device)

    for i in range(0, activations.shape[0], batch_size):
        j = min(i + batch_size, activations.shape[0])
        pca.train_batch(activations[i:j].to(device))

    return pca

def calc_mean(activations, batch_size=512, device="cuda:0"):
    mean = BatchedMean(activations.shape[1], device)

    for i in range(0, activations.shape[0], batch_size):
        j = min(i + batch_size, activations.shape[0])
        mean.train_batch(activations[i:j].to(device))

    return mean.get_mean()

class BatchedMean:
    def __init__(self, n_dims, device):
        super().__init__()
        self.n_dims = n_dims
        self.device = device

        self.mean = torch.zeros((n_dims,), device=device)
        self.n_samples = 0
    
    def train_batch(self, activations):
        batch_size = activations.shape[0]
        self.mean *= self.n_samples / (self.n_samples + batch_size)
        self.mean += torch.sum(activations, dim=0) / (self.n_samples + batch_size)
    
    def get_mean(self):
        return self.mean

class BatchedPCA:
    def __init__(self, n_dims, device):
        super().__init__()
        self.n_dims = n_dims
        self.device = device

        self.cov = torch.zeros((n_dims, n_dims), device=device)
        self.mean = torch.zeros((n_dims,), device=device)
        self.n_samples = 0

    def get_mean(self):
        return self.mean

    def train_batch(self, activations):
        # activations: (batch_size, n_dims)
        batch_size = activations.shape[0]
        corrected = activations - self.mean.unsqueeze(0)
        new_mean = self.mean + torch.mean(corrected, dim=0) * batch_size / (self.n_samples + batch_size)
        cov_update = torch.einsum("bi,bj->bij", corrected, activations - new_mean.unsqueeze(0)).mean(dim=0)
        self.cov = self.cov * (self.n_samples / (self.n_samples + batch_size)) + cov_update * batch_size / (
            self.n_samples + batch_size
        )
        self.mean = new_mean
        self.n_samples += batch_size

    def get_pca(self):
        cov_symm = (self.cov + self.cov.T) / 2
        eigvals, eigvecs = torch.linalg.eigh(cov_symm)
        return eigvals, eigvecs

    def get_centering_transform(self):
        eigvals, eigvecs = self.get_pca()

        # assert torch.all(eigvals > 0), "Covariance matrix is not positive definite"
        # clamp to avoid numerical issues
        eigvals = torch.clamp(eigvals, min=1e-6)

        scaling = 1 / torch.sqrt(eigvals)

        assert torch.all(~torch.isnan(scaling)), "Scaling has NaNs"

        return self.get_mean(), eigvecs, scaling

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

    def to_rotation_dict(self, n_components=None):
        if n_components is None:
            n_components = self.n_dims
        return Rotation(self.get_dict()[:n_components])

    def to_pve_rotation_dict(self, n_components=None):
        if n_components is None:
            n_components = self.n_dims
        dirs = self.get_dict()[:n_components]
        dirs_ = torch.cat([dirs, -dirs], dim=0)
        return TiedSAE(dirs_, torch.zeros(2 * n_components), centering=(self.get_mean(), None, None), norm_encoder=True)


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
