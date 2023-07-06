import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.func import stack_module_state, functional_call

import torchopt

import copy

class VectorizedEnsemble():
    def __init__(self, models, optimizer):
        self.params, self.buffers = stack_module_state(models)
        self.optimizer = optimizer
        self.optim_states = torch.vmap(self.optimizer.init)(self.params)

        self.modeldesc = copy.deepcopy(models[0]).to("meta")
    
    def to_device(self, device):
        self.params = self.params.to(device)
        self.buffers = self.buffers.to(device)
        self.optim_states = self.optim_states.to(device)

    @staticmethod
    def from_state(cls, state_dict):
        self = cls.__new__(cls)
        self.params = state_dict["params"]
        self.buffers = state_dict["buffers"]
        self.optimizer = state_dict["optimizer"]
        self.optim_states = state_dict["optim_states"]

        self.modeldesc = state_dict["modeldesc"]

    def step_batch(self, minibatches):
        def compute_loss(params, buffers, minibatch):
            losses = functional_call(self.modeldesc.loss, (params, buffers), minibatch)
            return (losses, losses)
        
        def compute_grads(params, buffers, minibatch):
            return torch.func.grad(compute_loss, has_aux=True)(params, buffers, minibatch)
        
        grads, aux = torch.vmap(compute_grads)(self.params, self.buffers, minibatches)
        updates, self.optim_states = torch.vmap(self.optimizer.update)(grads, self.optim_states)

        def apply_updates(params, updates):
            return torchopt.apply_updates(params, updates, inplace=False)

        self.params = torch.vmap(apply_updates)(self.params, updates)

        return aux
    
    def state_dict(self):
        return {
            "params": self.params,
            "buffers": self.buffers,
            "optimizer": self.optimizer,
            "optim_states": self.optim_states,
            "modeldesc": self.modeldesc
        }

    def share_memory(self):
        self.params.share_memory()
        self.buffers.share_memory()
        self.optim_states.share_memory()