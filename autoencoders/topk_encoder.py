import torch
import torch.nn.functional as F

from autoencoders.learned_dict import LearnedDict
from autoencoders.ensemble import DictSignature


class TopKEncoder(DictSignature):
    @staticmethod
    def init(d_activation, n_features, sparsity, dtype=torch.float32):
        params = {}
        params["dict"] = torch.randn(n_features, d_activation, dtype=dtype)

        buffers = {}
        buffers["sparsity"] = torch.tensor(sparsity, dtype=torch.long)

        return params, buffers

    @staticmethod
    def encode(b, sparsity, normed_dict):
        scores = torch.einsum("ij,bj->bi", normed_dict, b)
        topk = torch.topk(scores, sparsity, dim=-1).indices

        code = torch.zeros_like(scores)
        code.scatter_(dim=-1, index=topk, src=scores.gather(dim=-1, index=topk))

        return F.relu(code)

    @staticmethod
    def loss(params, buffers, batch):
        normed_dict = params["dict"] / torch.norm(params["dict"], dim=-1)[:, None]

        b = batch
        sparsity = buffers["sparsity"]
        code = TopKEncoder.encode(b, sparsity, normed_dict)
        b_ = torch.einsum("ij,bi->bj", normed_dict, code)

        loss = F.mse_loss(b, b_)

        return loss, ({"loss": loss}, {"c": code})

    @staticmethod
    def to_learned_dict(params, buffers):
        sparsity = buffers["sparsity"].item()
        normed_dict = params["dict"] / torch.norm(params["dict"], dim=-1)[:, None]
        return TopKLearnedDict(normed_dict, sparsity)


class TopKLearnedDict(LearnedDict):
    def __init__(self, dict, sparsity):
        self.dict = dict
        self.sparsity = sparsity
        self.n_feats, self.activation_size = self.dict.shape

    def to_device(self, device):
        self.dict = self.dict.to(device)

    def encode(self, x):
        return TopKEncoder.encode(x, self.sparsity, self.dict)

    def get_learned_dict(self):
        return self.dict
