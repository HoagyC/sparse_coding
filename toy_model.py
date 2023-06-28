"""
toy model of superposition(?) in MLPs
we want to look at the dynamics of a small number of MLPs in the middle of a transformer. we pick n random directions in an m dimensional space, each with a sparsity value, maybe correlation. we randomly select some of these according to their sparsity and add them together to form our input, which is a toy model of a residual space. for our target, we want a target random vector v_i added to the outputs for each feature that is added. for further experiments we can repeat this unit so that sequential computation is required. 

there should be a fidelity tradeoff with how many features we have and how sparse they are, as well as how many layers there are 

questions:
what does the performance look like in terms of n_features, sparsity?
does it use superposition, if so, when? are features distributed across multiple neurons?
how does the picture change when we add multiple layers?
"""

import torch

n_dimensions = 100
mlp_ratio = 1
n_features = 100
sparsity = 0.9
non_linearity = torch.nn.ReLU() # torch.nn.GELU()

# Generate random features
init_features = torch.randn(n_features, n_dimensions)
init_features = torch.nn.functional.normalize(init_features, dim=1)

target_features = torch.randn(n_features, n_dimensions)
target_features = torch.nn.functional.normalize(target_features, dim=1)

model = torch.nn.Sequential(
    torch.nn.Linear(n_dimensions, n_dimensions * mlp_ratio),
    torch.nn.ReLU(),
    torch.nn.Linear(n_dimensions * mlp_ratio, n_dimensions),
)





