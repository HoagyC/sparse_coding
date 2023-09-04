import optree
import torch
import torch.nn.functional as F

import optimizers.sgdm
from autoencoders.learned_dict import LearnedDict

N_ITERS_OPT = 100


class DirectCoefOptimizer:
    @staticmethod
    def init(d_activation, n_features, l1_alpha, lr=1e-3, dtype=torch.float32):
        params = {}
        params["decoder"] = torch.randn(n_features, d_activation, dtype=dtype)

        buffers = {}
        buffers["l1_alpha"] = torch.tensor(l1_alpha, dtype=dtype)
        buffers["lr"] = torch.tensor(lr, dtype=dtype)

        return params, buffers

    @staticmethod
    def objective(c, normed_dict, batch, l1_alpha):
        x_hat = torch.einsum("ij,bi->bj", normed_dict, c)

        l_reconstruction = (x_hat - batch).pow(2).mean()
        l_sparsity = l1_alpha * torch.norm(c, 1, dim=-1).mean()

        losses = {
            "loss": l_reconstruction + l_sparsity,
            "l_reconstruction": l_reconstruction,
            "l_l1": l_sparsity,
        }

        aux = {"c": c}

        return l_reconstruction + l_sparsity, (losses, aux)

    @staticmethod
    def basis_pursuit(params, buffers, batch, normed_dict=None):
        if normed_dict is None:
            decoder_norms = torch.norm(params["decoder"], 2, dim=-1)
            normed_dict = params["decoder"] / torch.clamp(decoder_norms, 1e-8)[:, None]

        # hate this
        c = torch.zeros_like(torch.einsum("ij,bj->bi", normed_dict, batch))

        optimizer = optimizers.sgdm.SGDM(buffers["lr"], 0.9)
        optim_state = optimizer.init(c)

        for _ in range(N_ITERS_OPT):
            grads, _ = torch.func.grad(DirectCoefOptimizer.objective, has_aux=True)(c, normed_dict, batch, buffers["l1_alpha"])
            updates, optim_state = optimizer.update(grads, optim_state)
            c += updates
            c = F.relu(c)

        return c

    @staticmethod
    def loss(params, buffers, batch):
        decoder_norms = torch.norm(params["decoder"], 2, dim=-1)
        normed_dict = params["decoder"] / torch.clamp(decoder_norms, 1e-8)[:, None]

        with torch.no_grad():
            c = DirectCoefOptimizer.basis_pursuit(params, buffers, batch, normed_dict=normed_dict)

        x_hat = torch.einsum("ij,bi->bj", normed_dict, c)
        l_reconstruction = (x_hat - batch).pow(2).mean()

        return l_reconstruction, ({"loss": l_reconstruction}, {"c": c})

    @staticmethod
    def to_learned_dict(params, buffers):
        return DirectCoefSearch(params, buffers)


class DirectCoefSearch(LearnedDict):
    def __init__(self, params, buffers):
        self.params = params
        self.buffers = buffers

    def encode(self, x):
        return DirectCoefOptimizer.basis_pursuit(self.params, self.buffers, x)

    def get_learned_dict(self):
        decoder_norms = torch.norm(self.params["decoder"], 2, dim=-1)
        return self.params["decoder"] / torch.clamp(decoder_norms, 1e-8)[:, None]

    def to_device(self, device):
        self.params = optree.tree_map(lambda t: t.to(device), self.params)
        self.buffers = optree.tree_map(lambda t: t.to(device), self.buffers)
