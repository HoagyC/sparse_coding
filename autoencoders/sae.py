import torch
import torch.nn as nn
import torch.nn.functional as F

def act_name_to_module(act_str):
    if act_str == "relu":
        return nn.ReLU()
    elif act_str == "gelu":
        return nn.GELU()
    elif act_str == "tanh":
        return nn.Tanh()
    elif act_str == "sigmoid":
        return nn.Sigmoid()
    elif act_str == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation function: {act_str}")

class SAE(nn.Module):
    def __init__(self, activation_size, n_dict_components, t_type=torch.float32, l1_coef=0.0, activation="relu", bias_l2_coef=0.0):
        super(SAE, self).__init__()
        self.decoder = nn.Linear(n_dict_components, activation_size, bias=False)
        # Initialize the decoder weights orthogonally
        nn.init.orthogonal_(self.decoder.weight)
        self.decoder = self.decoder.to(t_type)

        act_module = act_name_to_module(activation)

        self.encoder = nn.Sequential(nn.Linear(activation_size, n_dict_components).to(t_type), act_module)
        self.l1_coef = l1_coef
        self.bias_l2_coef = bias_l2_coef
        self.activation_size = activation_size
        self.n_dict_components = n_dict_components

    def forward(self, x):
        c = self.encoder(x)
        # Apply unit norm constraint to the decoder weights
        self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)

        x_hat = self.decoder(c)
        return x_hat, c
    
    def loss(self, x, x_hat, c):
        l_reconstruction = torch.nn.MSELoss()(x, x_hat)
        l_l1 = self.l1_coef * torch.norm(self.get_dict(), 1, dim=1).mean()
        l_bias_l2 = self.bias_l2_coef * torch.norm(self.decoder.bias, 2)
        return l_reconstruction + l_l1 + l_bias_l2, l_reconstruction, l_l1
    
    def train_batch(self, batch, optimizer=None):
        if optimizer is None:
            raise ValueError("optimizer must be specified for an SAE")

        optimizer.zero_grad()
        x_hat, c = self(batch)
        loss, l_reconstruction, l_l1 = self.loss(batch, x_hat, c)
        loss.backward()
        optimizer.step()
        return loss.detach(), l_reconstruction.detach(), l_l1.detach()

    def get_dict(self):
        return self.decoder.weight

    def configure_optimizers(self, **kwargs):
        return torch.optim.Adam(self.parameters(), **kwargs)

    @property
    def device(self):
        return next(self.parameters()).device

Encoder.register(SAE)

class TiedSAE(nn.Module):
    def __init__(self, activation_size, n_dict_components, l1_coef=0.0, activation="relu", bias_l2_coef=0.0):
        self.n_dict_components = n_dict_components
        self.activation_size = activation_size

        self.act_module = act_name_to_module(activation)

        self.weights = nn.Parameter(torch.empty((n_dict_components, activation_size)))
        self.bias = nn.Parameter(torch.zeros((n_dict_components,)))
        nn.init.xavier_uniform_(self.weights)

        self.l1_coef = l1_coef
    
    def forward(self, x):
        c = torch.einsum("ij,bj->bi", self.weights, x)
        c = c + self.bias
        c = self.act_module(c)
        x_hat = torch.einsum("ij,bi->bj", self.weights, c)

        return x_hat, c
    
    def loss(self, x, x_hat, c):
        l_reconstruction = F.mse_loss(x, x_hat).mean(dim=0)

        l_sparsity = self.sparsity_coef * F.l1_loss(c, torch.zeros_like(c)).mean(dim=0)
        l_bias_l2 = self.bias_l2_coef * torch.norm(self.bias, 2)

        loss = l_reconstruction + l_sparsity + l_bias_l2

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