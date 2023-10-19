import torch
import torch.nn as nn
import torch.nn.functional as F

# Reconstruction ICA
# http://ai.stanford.edu/~quocle/LeKarpenkoNgiamNg.pdf


class RICA(nn.Module):
    def __init__(
        self,
        activation_size,
        n_dict_components,
        sparsity_coef=0.0,
        sparsity_loss="smooth_l1",
    ):
        self.n_dict_components = n_dict_components
        self.activation_size = activation_size

        self.weights = nn.Parameter(torch.empty((n_dict_components, activation_size)))
        nn.init.xavier_uniform_(self.weights)

        self.sparsity_loss = sparsity_loss
        self.sparsity_coef = sparsity_coef

    def forward(self, x):
        c = torch.einsum("ij,bj->bi", self.weights, x)
        x_hat = torch.einsum("ij,bi->bj", self.weights, c)

        return x_hat, c

    def loss(self, x, x_hat, c):
        l_reconstruction = F.mse_loss(x, x_hat).mean(dim=0)

        if self.sparsity_loss == "smooth_l1":
            l_sparsity = F.smooth_l1_loss(c, torch.zeros_like(c)).mean(dim=0)
        elif self.sparsity_loss == "l1":
            l_sparsity = F.l1_loss(c, torch.zeros_like(c)).mean(dim=0)

        loss = l_reconstruction + self.sparsity_coef * l_sparsity

        return loss, l_reconstruction, l_sparsity

    def train_batch(self, batch, optimizer=None):
        if optimizer is None:
            raise ValueError("optimizer must be specified for RICA")

        optimizer.zero_grad()
        x_hat, c = self(batch)
        loss, l_reconstruction, l_sparsity = self.loss(batch, x_hat, c)
        loss.backward()
        optimizer.step()

        return loss.detach(), l_reconstruction.detach(), l_sparsity.detach()

    def get_dict(self):
        return self.weights

    def configure_optimizers(self, **kwargs):
        return torch.optim.Adam(self.parameters(), **kwargs)
