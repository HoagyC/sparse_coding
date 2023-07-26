import torch
import optree

# functional SGD + momentum optimizer

class SGDM:
    def __init__(self, lr_groups, momentum: float):
        self.lr_groups = lr_groups
        self.momentum = momentum
    
    def init(self, params):
        device = optree.tree_flatten(params)[0][0].device

        momentum = optree.tree_map(torch.zeros_like, params)

        return {"momentum": momentum}

    def update(self, grads, states):
        momentum = states["momentum"]

        updated_momentum = optree.tree_map(
            lambda m, g: self.momentum * m + (1 - m) * g,
            momentum,
            grads
        )

        corrected_momentum = optree.tree_map(
            lambda m: m / (1 - self.momentum),
            updated_momentum
        )

        updates = optree.tree_map(
            lambda lr, m: - lr * m,
            self.lr_groups,
            corrected_momentum
        )

        return updates, {"momentum": updated_momentum}