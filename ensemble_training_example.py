import torch
import torchopt

from autoencoders.sae_ensemble import FunctionalSAE
from autoencoders.ensemble import FunctionalEnsemble
from sc_datasets.random_dataset import RandomDatasetGenerator

# we calculate gradients functionally so disable autograd for memory
torch.set_grad_enabled(False)

l1_exp_base = 10 ** (1 / 4)
n_features = 1024
d_activation = 512
n_dict_components = 2048
batch_size = 256
dataset = RandomDatasetGenerator(d_activation, n_features, batch_size, 5, 0.99, True, "cuda")


def mmcs(truth, dict):
    # truth: [n_features, d_activation]
    # dict: [n_dict_components, d_activation]

    cosine_sim = truth @ dict.T
    max_cosine_sim, _ = torch.max(cosine_sim, dim=0)
    return max_cosine_sim.mean()


l1_coefs = [1 * l1_exp_base**i for i in range(-16, -11)]

models = [FunctionalSAE.init(d_activation, n_dict_components, l1_alpha=l1_coef) for l1_coef in l1_coefs]
ensemble = FunctionalEnsemble(models, FunctionalSAE, torchopt.adam(lr=1e-3), {"lr": 1e-3})

for i in range(1000):
    minibatch = dataset.__next__().unsqueeze(0).expand(len(models), -1, -1)
    # minibatch = torch.randn(len(models), batch_size, d_activation, device="cuda", requires_grad=True)
    losses = ensemble.step_batch(minibatch)

    if i % 100 == 0:
        mmcss = torch.vmap(lambda y: mmcs(dataset.feats, y))(ensemble.params["decoder"])

        print(f"Step {i}")
        print(f"    Losses: {losses}")
        print(f"    MMCS: {mmcss}")
