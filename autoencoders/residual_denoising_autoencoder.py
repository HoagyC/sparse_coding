import torch
import torch.nn.functional as F

from autoencoders.learned_dict import LearnedDict

import optree

# https://arxiv.org/pdf/2008.02683.pdf
class LISTALayer:
    @staticmethod
    def init(d_activation, n_features, dtype=torch.float32):
        params = {}
        params["W"] = torch.empty(n_features, d_activation, dtype=dtype)
        torch.nn.init.orthogonal_(params["W"])
        params["theta"] = torch.randn(n_features, dtype=dtype) * 0.02
        params["rho"] = torch.tensor(0.1, dtype=dtype)
        return params
    
    @staticmethod
    def forward(params, y, b, x, A):
        # solves Ay = b

        m = torch.clamp(params["rho"], min=0.0, max=1.0)

        Ay = torch.einsum("ij,bi->bj", A, y)
        r = y + torch.einsum("ij,bj->bi", params["W"], b - Ay)
        x_ = F.relu(r + params["theta"][None, :])
        y_ = x_ + m * (x_ - x)
        return y_, x_

class FunctionalLISTADenoisingSAE:
    @staticmethod
    def init(d_activation, n_features, n_hidden_layers, l1_alpha, dtype=torch.float32):
        params = {}
        params["decoder"] = torch.empty(n_features, d_activation, dtype=dtype)
        torch.nn.init.orthogonal_(params["decoder"])

        #params["encoder_embedding"] = params["decoder"].clone().T
        params["encoder_layers"] = [
            LISTALayer.init(d_activation, n_features, dtype=dtype)
            for _ in range(n_hidden_layers)
        ]
        #params["encoder_bias"] = torch.randn(n_features, dtype=dtype) * 0.02

        buffers = {}
        buffers["l1_alpha"] = torch.tensor(l1_alpha, dtype=dtype)

        return params, buffers
    
    @staticmethod
    def encode(params, b, learned_dict):
        y = torch.einsum("ij,bj->bi", learned_dict, b)
        x = y
        for layer in params["encoder_layers"]:
            y, x = LISTALayer.forward(layer, y, b, x, learned_dict)
        return y

    @staticmethod
    def loss(params, buffers, batch):
        decoder_norms = torch.norm(params["decoder"], 2, dim=-1)
        learned_dict = params["decoder"] / torch.clamp(decoder_norms, 1e-8)[:, None]

        c = FunctionalLISTADenoisingSAE.encode(params, batch, learned_dict)

        x_hat = torch.einsum("ij,bi->bj", learned_dict, c)

        l_reconstruction = (x_hat - batch).pow(2).mean()
        l_sparsity = buffers["l1_alpha"] * torch.norm(c, 1, dim=-1).mean()

        loss_data = {
            "loss": l_reconstruction + l_sparsity,
            "l_reconstruction": l_reconstruction,
            "l_l1": l_sparsity
        }
        aux_data = {
            "c": c
        }

        return l_reconstruction + l_sparsity, (loss_data, aux_data)

    @staticmethod
    def to_learned_dict(params, buffers):
        return LISTADenoisingSAE(params)

    @staticmethod
    def init_lr(n_hidden_layers, lr, lr_encoder=None):
        if lr_encoder is None:
            lr_encoder = lr

        lrs = {"decoder": lr}

        lrs["encoder_embedding"] = lr_encoder
        lrs["encoder_bias"] = lr_encoder
        lrs["encoder_layers"] = []
        for _ in range(n_hidden_layers):
            layer = {"weight": lr, "bias": lr}
            lrs["encoder_layers"].append(layer)

        return lrs

class LISTADenoisingSAE(LearnedDict):
    def __init__(self, params):
        self.params = params

    def encode(self, x):
        learned_dict = self.get_learned_dict()

        return FunctionalLISTADenoisingSAE.encode(self.params, x, learned_dict)

    def to_device(self, device):
        self.params = optree.tree_map(lambda t: t.to(device=device), self.params)

    def get_learned_dict(self):
        decoder_norms = torch.norm(self.params["decoder"], 2, dim=-1)
        learned_dict = self.params["decoder"] / torch.clamp(decoder_norms, 1e-8)[:, None]
        return learned_dict

def shrinkage(x, b):
    #return x * torch.exp(- b.pow(2) / x.pow(2))
    return F.relu(x + b[None, :])

LISTA_ITERS = 3

class LISTA:
    @staticmethod
    def init(d_activation, n_features, dtype=torch.float32):
        params = {}
        params["W_e"] = torch.empty(d_activation, n_features, dtype=dtype)
        torch.nn.init.orthogonal_(params["W_e"])

        params["S"] = torch.empty(n_features, n_features, dtype=dtype)
        torch.nn.init.orthogonal_(params["S"])

        params["theta"] = torch.empty(n_features, dtype=dtype)
        torch.nn.init.normal_(params["theta"])

        return params
    
    @staticmethod
    def forward(params, batch, iters=LISTA_ITERS):
        b = torch.einsum("bi,ij->bj", batch, params["W_e"])
        z = shrinkage(b, params["theta"])

        for _ in range(iters):
            c = b + torch.einsum("bj,jg->bg", z, params["S"])
            z = shrinkage(c, params["theta"])
        
        return z

class FunctionalLISTASAE:
    @staticmethod
    def init(d_activation, n_features, l1_alpha, dtype=torch.float32):
        params, buffers = {}, {}

        params["LISTA"] = LISTA.init(d_activation, n_features, dtype=dtype)
        params["dict"] = torch.empty(n_features, d_activation, dtype=dtype)
        torch.nn.init.orthogonal_(params["dict"])

        buffers["l1_alpha"] = torch.tensor(l1_alpha, dtype=dtype)

        return params, buffers
    
    @staticmethod
    def encode(params, batch):
        c = LISTA.forward(params["LISTA"], batch)
        return c
    
    @staticmethod
    def loss(params, buffers, batch):
        normed_dict = params["dict"] / torch.norm(params["dict"], 2, dim=-1)[:, None]
        c = LISTA.forward(params["LISTA"], batch)
        x_hat = torch.einsum("ji,bj->bi", normed_dict, c)

        l_reconstruction = (x_hat - batch).pow(2).mean()
        l_sparsity = buffers["l1_alpha"] * torch.norm(c, 1, dim=-1).mean()

        loss_data = {
            "loss": l_reconstruction + l_sparsity,
            "l_reconstruction": l_reconstruction,
            "l_l1": l_sparsity
        }

        aux_data = {
            "c": c
        }

        return l_reconstruction + l_sparsity, (loss_data, aux_data)
    
    @staticmethod
    def to_learned_dict(params, buffers):
        return LISTASAE(params)

class LISTASAE(LearnedDict):
    def __init__(self, params):
        self.params = params
    
    def encode(self, batch):
        return FunctionalLISTASAE.encode(self.params, batch)
    
    def to_device(self, device):
        self.params = optree.tree_map(lambda t: t.to(device=device), self.params)
    
    def get_learned_dict(self):
        return self.params["dict"] / torch.norm(self.params["dict"], 2, dim=-1)[:, None]