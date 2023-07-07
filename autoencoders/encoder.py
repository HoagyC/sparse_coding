from typing import Optional, Any
from functools import singledispatch

from torchtyping import TensorType

import abc

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_dict(self) -> TensorType["n_dict_components", "activation_size"]:
        pass

    @abc.abstractmethod
    def train_batch(self, batch: TensorType["batch_size", "activation_size"], optimizer: Optional[Any] = None) -> Any:
        pass

    @abc.abstractmethod
    def configure_optimizers(self, **kwargs) -> Optional[Any]:
        pass