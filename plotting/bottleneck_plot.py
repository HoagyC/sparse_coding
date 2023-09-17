import sys

sys.path.append("..")

import os
import shutil

from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_bottleneck_scores(layer):
    base_folder = "sparse_coding_aidan"
    scores = torch.load(f"/mnt/ssd-cluster/bottleneck_410m/dict_scores_layer_{layer}.pt")

    print(scores.keys())

    #diff_mean_scores = torch.load("diff_mean_scores_layer_2.pt")

    fig, ax = plt.subplots()

    ax.grid(True, alpha=0.5, linestyle="dashed")
    ax.set_axisbelow(True)

    #colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    #markers = ["x", "+", "*", "o", "v", "^", "<", ">", "s", "."]
    #styles = ["solid", "dashed", "dashdot", "dotted"]

    xs, ys, keys = [], [], []
    for key, score in scores.items():
        graph, div, corruption = zip(*sorted(score, key=lambda x: len(x[0])))
        graph_size = [len(g) for g in graph]
        print(graph_size, div, corruption)
        xs.append(graph_size)
        ys.append(div)
        keys.append(key)

    for key, x, y in zip(keys, xs, ys):
        style = "dashed"
        if key == "pca":
            label = "PCA"
            color = "Reds"
            style = "dotted"
            c = 0.5
        elif key == "learned_r4_1e-03":
            label = "Dict. alpha=1e-3"
            color = "Blues"
            c = 0.7
        elif key == "learned_r4_3e-04":
            label = "Dict. alpha=3e-4"
            color = "Blues"
            c = 0.5
        elif key == "learned_r4_1e-04":
            label = "Dict. alpha=1e-4"
            color = "Blues"
            c = 0.3
        
        cmap = plt.get_cmap(color)
        c = cmap(c)

        ax.plot(x, y, color=c, linestyle=style, label=label, alpha=1)

    ax.set_xlabel("No. Uncorrupted Features")
    ax.set_ylabel("KL-Divergence From Base")

    ax.set_title(f"Precision-Complexity Tradeoff Curve - Layer {layer}")

    ax.set_xlim(0, 1024)

    ax.legend(
        loc="upper right",
        framealpha=1,
    )

    #shutil.rmtree("graphs", ignore_errors=True)
    #os.mkdir("graphs", exist_ok=True)

    plt.savefig(f"graphs/bottleneck_scores_layer_{layer}.png")

    plt.close(fig)
    del fig, ax

if __name__ == "__main__":
    layers = [4, 6, 8, 10, 12, 14, 16, 18]

    for layer in layers:
        plot_bottleneck_scores(layer)