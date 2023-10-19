import copy
from typing import List, Optional, Tuple, Type, Union

import optree
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchopt
from torch import Tensor


# beg for forgiveness from the gods of OOP!
# interfaces cower in fear of being arbitrarily
# warped into forms they were never meant to take.
class DictSignature:
    @staticmethod
    def to_learned_dict(params, buffers):
        pass

    @staticmethod
    def loss(params, buffers, batch):
        pass


def optim_str_to_func(optim_str):
    if optim_str == "adam":
        return torchopt.adam
    elif optim_str == "sgd":
        return torchopt.sgd
    else:
        raise ValueError("Unknown optimizer string: {}".format(optim_str))


# https://github.com/pytorch/pytorch/blob/main/torch/_functorch/functional_call.py#L236
def construct_stacked_leaf(
    tensors: Union[Tuple[Tensor, ...], List[Tensor]],
    device: Optional[Union[torch.device, str]] = None,
) -> Tensor:
    all_requires_grad = all(t.requires_grad for t in tensors)
    none_requires_grad = all(not t.requires_grad for t in tensors)
    if not all_requires_grad and not none_requires_grad:
        raise RuntimeError(f"Expected tensors from each model to have the same .requires_grad")
    result = torch.stack(tensors).to(device=device)
    if all_requires_grad:
        result = result.detach().requires_grad_()
    return result


# now recurses! (cool)
def stack_dict(models: list, device=None):
    tensors, treespecs = zip(*[optree.tree_flatten(model) for model in models])
    tensors = list(zip(*tensors))
    tensors_ = []
    for ts in tensors:
        tensors_.append(construct_stacked_leaf(ts, device=device))
    return optree.tree_unflatten(treespecs[0], tensors_)


def unstack_dict(params, n_models, device=None):
    tensors, treespec = optree.tree_flatten(params)
    tensors_ = [[] for _ in range(n_models)]
    for t in tensors:
        for i in range(n_models):
            tensors_[i].append(t[i].to(device=device))
    return [optree.tree_unflatten(treespec, ts) for ts in tensors_]


class FunctionalEnsemble:
    def __init__(
        self,
        models,
        sig: Type[DictSignature],
        optimizer_func,
        optimizer_kwargs,
        device=None,
        no_stacking=False,
    ):
        if device is None:
            self.device = models[0]["encoder"].device
        else:
            self.device = device

        self.n_models = len(models)
        params, buffers = tuple(zip(*models))
        self.params = stack_dict(params, device=self.device)
        self.buffers = stack_dict(buffers, device=self.device)

        self.sig = sig
        self.no_stacking = no_stacking

        self.optimizer_func = optimizer_func
        self.optimizer_kwargs = optimizer_kwargs

        self.optimizer = optimizer_func(**optimizer_kwargs)
        self.optim_states = torch.vmap(self.optimizer.init)(self.params)

        self.init_functions()

    def init_functions(self):
        if self.no_stacking:

            def calc_grads_(params, buffers, batch):
                return torch.func.grad(self.sig.loss, has_aux=True)(params, buffers, batch)

            def calc_grads(params, buffers, batch):
                grads, auxs = [], []
                for i in range(self.n_models):
                    params_ = optree.tree_map(lambda t: t[i], params)
                    buffers_ = optree.tree_map(lambda t: t[i], buffers)

                    g, a = calc_grads_(params_, buffers_, batch[i])
                    grads.append(g)
                    auxs.append(a)
                return stack_dict(grads), stack_dict(auxs)

            self.calc_grads = calc_grads
        else:

            def calc_grads(params, buffers, batch):
                return torch.func.grad(self.sig.loss, has_aux=True)(params, buffers, batch)

            self.calc_grads = torch.vmap(calc_grads)
        self.update = torch.vmap(self.optimizer.update)

    @staticmethod
    def from_state(state_dict):
        self = FunctionalEnsemble.__new__(FunctionalEnsemble)

        self.device = state_dict["device"]
        self.n_models = state_dict["n_models"]
        self.params = state_dict["params"]
        self.buffers = state_dict["buffers"]
        self.sig = state_dict["sig"]
        self.no_stacking = state_dict["no_stacking"]
        self.optimizer_func = state_dict["optimizer_func"]
        self.optimizer_kwargs = state_dict["optimizer_kwargs"]
        self.optim_states = state_dict["optim_states"]

        self.optimizer = self.optimizer_func(**self.optimizer_kwargs)

        self.init_functions()

        return self

    def unstack(self, device=None):
        params = unstack_dict(self.params, self.n_models, device=device)
        buffers = unstack_dict(self.buffers, self.n_models, device=device)
        return list(zip(params, buffers))

    def state_dict(self):
        return {
            "device": self.device,
            "n_models": self.n_models,
            "params": self.params,
            "buffers": self.buffers,
            "sig": self.sig,
            "no_stacking": self.no_stacking,
            "optimizer_func": self.optimizer_func,
            "optimizer_kwargs": self.optimizer_kwargs,
            "optim_states": self.optim_states,
        }

    def to_device(self, device):
        self.device = device

        self.params = optree.tree_map(lambda t: t.to(device), self.params)
        self.buffers = optree.tree_map(lambda t: t.to(device), self.buffers)
        self.optim_states = optree.tree_map(lambda t: t.to(device), self.optim_states)

    def to_shared_memory(self):
        optree.tree_map_(lambda t: t.share_memory_(), self.params)
        optree.tree_map_(lambda t: t.share_memory_(), self.buffers)
        optree.tree_map_(lambda t: t.share_memory_(), self.optim_states)

    def step_batch(self, minibatches, expand_dims=True):
        with torch.no_grad():
            if expand_dims:
                minibatches = minibatches.expand(self.n_models, *minibatches.shape)

            grads, (loss, aux) = self.calc_grads(self.params, self.buffers, minibatches)

            updates, new_optim_states = self.update(grads, self.optim_states)

            # write new optim states into self.optim_states tensors
            new_leaves, _ = optree.tree_flatten(new_optim_states)
            leaves, _ = optree.tree_flatten(self.optim_states)
            for new_leaf, leaf in zip(new_leaves, leaves):
                leaf = leaf.clone()
                leaf.copy_(new_leaf)

            torchopt.apply_updates(self.params, updates)

            return loss, aux
