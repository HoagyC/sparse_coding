import torch
import optree

# functional adam optimizer
# torchopt.adam doesn't implement
# parameter groups annoyingly, so had
# to reimplement adam

# ref:
# ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION
# Diederik P. Kingma, Jimmy Lei Ba
# https://arxiv.org/pdf/1412.6980.pdf

class Adam:
    def __init__(self, lr_groups, betas: tuple[float, float], eps: float):
        self.lr_groups = lr_groups
        self.betas = betas
        self.eps = eps
    
    def init(self, params):
        device = optree.tree_flatten(params)[0][0].device

        first_moments = optree.tree_map(torch.zeros_like, params)
        second_moments = optree.tree_map(torch.zeros_like, params)

        return {"first_moments": first_moments, "second_moments": second_moments, "step": torch.tensor(0, dtype=torch.long, device=device)}
    
    def update(self, grads, states):
        first_moments = states["first_moments"]
        second_moments = states["second_moments"]
        step = states["step"] + 1

        updated_first_moments = optree.tree_map(
            lambda m, g: self.betas[0] * m + (1.0 - self.betas[0]) * g,
            first_moments,
            grads
        )
        updated_second_moments = optree.tree_map(
            lambda v, g: self.betas[1] * v + (1.0 - self.betas[1]) * (g * g),
            second_moments,
            grads
        )

        bias_correction_0 = 1 - self.betas[0] ** step
        bias_correction_1 = 1 - self.betas[1] ** step

        lr_scaling = torch.sqrt(bias_correction_1) / bias_correction_0

        corrected_first_moments = optree.tree_map(
            lambda m: m / bias_correction_0,
            updated_first_moments
        )
        corrected_second_moments = optree.tree_map(
            lambda v: v / bias_correction_1,
            updated_second_moments
        )

        updates = optree.tree_map(
            lambda lr, m_hat, v_hat: - lr * m_hat / (torch.sqrt(v_hat) + self.eps),
            self.lr_groups,
            corrected_first_moments,
            corrected_second_moments
        )

        return updates, {"first_moments": updated_first_moments, "second_moments": updated_second_moments, "step": step}