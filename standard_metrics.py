import torch
from torchtyping import TensorType
from typing import List, Tuple, Union, Optional

from autoencoders.learned_dict import LearnedDict

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use('Agg')

def mcs_duplicates(ground: LearnedDict, model: LearnedDict):
    # get max cosine sim between each model atom and all ground atoms
    cosine_sim = torch.einsum("md,gd->mg", model.get_learned_dict(), ground.get_learned_dict())
    max_cosine_sim = cosine_sim.max(dim=-1).values
    return max_cosine_sim

def mean_nonzero_activations(model: LearnedDict, batch: TensorType["batch_size", "activation_size"]):
    c = model.encode(batch)
    return (c > 0.0).float().mean(dim=0)

def plot_hist(scores: TensorType["n_dict_components"], x_label, y_label, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(scores, **kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return Image.fromarray(data, mode="RGB")

def plot_scatter(scores_x: TensorType["n_dict_components"], scores_y: TensorType["n_dict_components"], x_label, y_label, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(scores_x, scores_y, **kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return Image.fromarray(data, mode="RGB")

def plot_grid(scores: np.ndarray, first_tick_labels, second_tick_labels, first_label, second_label, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(scores, **kwargs)
    ax.set_xticks(np.arange(len(first_tick_labels)))
    ax.set_yticks(np.arange(len(second_tick_labels)))
    ax.set_xticklabels(first_tick_labels)
    ax.set_yticklabels(second_tick_labels)
    ax.set_xlabel(first_label)
    ax.set_ylabel(second_label)
    
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return Image.fromarray(data, mode="RGB")