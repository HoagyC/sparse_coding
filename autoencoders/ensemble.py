import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from torch.func import stack_module_state, functional_call

from typing import Union, Tuple, List, Optional

import torchopt
import optree

import copy

def optim_str_to_func(optim_str):
    if optim_str == "adam":
        return torchopt.adam
    elif optim_str == "sgd":
        return torchopt.sgd
    else:
        raise ValueError("Unknown optimizer string: {}".format(optim_str))

# https://github.com/pytorch/pytorch/blob/main/torch/_functorch/functional_call.py#L236
def construct_stacked_leaf(
    tensors: Union[Tuple[Tensor, ...], List[Tensor]], name: str, device: Optional[Union[torch.device, str]] = None
) -> Tensor:
    all_requires_grad = all(t.requires_grad for t in tensors)
    none_requires_grad = all(not t.requires_grad for t in tensors)
    if not all_requires_grad and not none_requires_grad:
        raise RuntimeError(
            f"Expected {name} from each model to have the same .requires_grad"
        )
    result = torch.stack(tensors).to(device=device)
    if all_requires_grad:
        result = result.detach().requires_grad_()
    return result

def stack_dict(all_params: List[dict], device=None):
    params = {
        k: construct_stacked_leaf(tuple(params[k] for params in all_params), k, device=device)
        for k in all_params[0]
    }
    return params

class FunctionalEnsemble():
    def __init__(self, models, loss_func, optimizer_func, optimizer_kwargs, device=None):
        if device is None:
            self.device = model_state_dicts[0]["device"]
        else:
            self.device = device

        self.n_models = len(models)
        params, buffers = tuple(zip(*models))
        self.params = stack_dict(params, device=self.device)
        self.buffers = stack_dict(buffers, device=self.device)

        self.loss_func = loss_func

        self.optimizer_func = optimizer_func
        self.optimizer_kwargs = optimizer_kwargs

        self.optimizer = optimizer_func(**optimizer_kwargs)
        self.optim_states = torch.vmap(self.optimizer.init)(self.params)
    
    @staticmethod
    def from_state(state_dict):
        self = FunctionalEnsemble.__new__(FunctionalEnsemble)

        self.device = state_dict["device"]
        self.n_models = state_dict["n_models"]
        self.params = state_dict["params"]
        self.buffers = state_dict["buffers"]
        self.loss_func = state_dict["loss_func"]
        self.optimizer_func = state_dict["optimizer_func"]
        self.optimizer_kwargs = state_dict["optimizer_kwargs"]
        self.optim_states = state_dict["optim_states"]

        self.optimizer = self.optimizer_func(**self.optimizer_kwargs)

        return self
    
    def state_dict(self):
        return {
            "device": self.device,
            "n_models": self.n_models,
            "params": self.params,
            "buffers": self.buffers,
            "loss_func": self.loss_func,
            "optimizer_func": self.optimizer_func,
            "optimizer_kwargs": self.optimizer_kwargs,
            "optim_states": self.optim_states
        }
    
    def to_device(self, device):
        self.device = device
        for _, param in self.params.items():
            param.to(device)
        for _, buffer in self.buffers.items():
            buffer.to(device)
        leaves, _ = optree.tree_flatten(self.optim_states)
        for leaf in leaves:
            leaf.to(device)
    
    def to_shared_memory(self):
        for _, param in self.params.items():
            param.share_memory_()
        for _, buffer in self.buffers.items():
            buffer.share_memory_()
        leaves, _ = optree.tree_flatten(self.optim_states)
        for leaf in leaves:
            leaf.share_memory_()
    
    def step_batch(self, minibatches, expand_dims=True):
        if expand_dims:
            minibatches = minibatches.expand(self.n_models, *minibatches.shape)
        
        def calc_grads(params, buffers, batch):
            return torch.func.grad(self.loss_func)(params, buffers, batch)
        
        grads = torch.vmap(calc_grads)(self.params, self.buffers, minibatches)
        updates, self.optim_states = torch.vmap(self.optimizer.update)(grads, self.optim_states)
        torchopt.apply_updates(self.params, updates)

# leaks memory somewhere; DO NOT USE
class VectorizedEnsemble():
    def __init__(self, models, optimizer_func, optimizer_kwargs, model_func, model_kwargs, device=None):
        if device is None:
            self.device = models[0].device
        else:
            self.device = device

        for model in models:
            model.to(self.device)        

        self.n_models = len(models)
        self.params, self.buffers = stack_module_state(models)

        self.optimizer_func = optimizer_func
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = optimizer_func(**optimizer_kwargs)

        self.optim_states = torch.vmap(self.optimizer.init)(self.params)

        self.model_func = model_func
        self.model_kwargs = model_kwargs
        self.modeldesc = model_func(**model_kwargs, device="meta")

    def to_device(self, device):
        self.device = device
        self.params = self.params.to(device)
        self.buffers = self.buffers.to(device)
        leaves, _ = optree.tree_flatten(self.optim_states)
        for leaf in leaves:
            leaf.to(device)

    @staticmethod
    def from_state(state_dict):
        self = VectorizedEnsemble.__new__(VectorizedEnsemble)
        self.n_models = state_dict["n_models"]
        self.device = state_dict["device"]
        self.params = state_dict["params"]
        self.buffers = state_dict["buffers"]
        self.optimizer_func = state_dict["optimizer_func"]
        self.optimizer_kwargs = state_dict["optimizer_kwargs"]

        self.optimizer = self.optimizer_func(**self.optimizer_kwargs)

        optim_states_leaves = state_dict["optim_states_leaves"]
        optim_states_treespec = state_dict["optim_states_treespec"]
        self.optim_states = optree.tree_unflatten(optim_states_treespec, optim_states_leaves)

        self.model_func = state_dict["model_func"]
        self.model_kwargs = state_dict["model_kwargs"]
        self.modeldesc = self.model_func(**self.model_kwargs, device="meta")

        return self

    def step_batch(self, minibatches, expand_dims=True):
        def compute_loss(params, buffers, minibatch):
            losses = functional_call(self.modeldesc, (params, buffers), minibatch)
            return losses
        
        def compute_grads(params, buffers, minibatch):
            return torch.func.grad(compute_loss, has_aux=False)(params, buffers, minibatch)
        
        if expand_dims:
            # minibatches: [batch_size, ...]
            minibatches = minibatches.expand(self.n_models, *minibatches.shape)

        grads = torch.vmap(compute_grads)(self.params, self.buffers, minibatches)
        updates, self.optim_states = torch.vmap(self.optimizer.update)(grads, self.optim_states)
        def apply_updates(params, updates):
            return torchopt.apply_updates(params, updates, inplace=False)

        self.params = torch.vmap(apply_updates)(self.params, updates)
    
    def state_dict(self):
        optim_states_leaves, optim_states_treespec = optree.tree_flatten(self.optim_states)
        return {
            "n_models": self.n_models,
            "device": self.device,
            "params": self.params,
            "buffers": self.buffers,
            "optimizer_func": self.optimizer_func,
            "optimizer_kwargs": self.optimizer_kwargs,
            "optim_states_leaves": optim_states_leaves,
            "optim_states_treespec": optim_states_treespec,
            "model_func": self.model_func,
            "model_kwargs": self.model_kwargs
        }

    def to_shared_memory(self):
        for _, p in self.params.items():
            p.share_memory_()
        for _, b in self.buffers.items():
            b.share_memory_()
        leaves, _ = optree.tree_flatten(self.optim_states)
        for leaf in leaves:
            leaf.share_memory_()


import torchopt
import optree

class DummyEnsemble:
    def __init__(self, device):
        self.device = device
        self.model = torch.empty(10, 128, device=device)
        torch.nn.init.normal_(self.model)
        self.optim_func = torchopt.adam
        self.optim_args = {"lr": 0.01}
        self.optimizer = self.optim_func(**self.optim_args)
        self.optim_state = self.optimizer.init(self.model)

    def step_batch(self, batch):
        grads = torch.func.grad(lambda d, b: (d @ b).sum())(self.model, batch)
        updates, self.optim_state = self.optimizer.update(grads, self.optim_state)
        torchopt.apply_updates(self.model, updates, inplace=True)
    
    @staticmethod
    def from_state(state_dict):
        self = DummyEnsemble.__new__(DummyEnsemble)
        self.model = state_dict["model"]
        self.optim_func = state_dict["optim_func"]
        self.optim_args = state_dict["optim_args"]
        self.optim_state = state_dict["optim_state"]
        self.optimizer = self.optim_func(**self.optim_args)
        return self

    def state_dict(self):
        return {
            "model": self.model,
            "optim_func": self.optim_func,
            "optim_args": self.optim_args,
            "optim_state": self.optim_state
        }
    
    def to_shared_memory(self):
        self.model.share_memory_()
        leaves, _ = optree.tree_flatten(self.optim_state)
        for leaf in leaves:
            leaf.share_memory_()
    
    def to(self, device):
        self.model = self.model.to(device)
        leaves, _ = optree.tree_flatten(self.optim_state)
        for leaf in leaves:
            leaf = leaf.to(device)