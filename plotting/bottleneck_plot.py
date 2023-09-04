import os
import shutil

from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

if __name__ == "__main__":
    base_folder = "sparse_coding_aidan"
    scores = torch.load(os.path.join(base_folder, "dict_scores_layer_2.pt"))

    scores = torch.load("dict_scores_layer_3.pt")

    #diff_mean_scores = torch.load("diff_mean_scores_layer_2.pt")

    fig, ax = plt.subplots()

    ax.grid(True, alpha=0.5, linestyle="dashed")
    ax.set_axisbelow(True)

    #colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    #markers = ["x", "+", "*", "o", "v", "^", "<", ">", "s", "."]
    #styles = ["solid", "dashed", "dashdot", "dotted"]

    xs, ys, keys = [], [], []
    for key, score in scores.items():
        _, graph, div, corruption = zip(*score)
        graph_size = [len(g) for g in graph]
        xs.append(graph_size)
        ys.append(div)
        keys.append(key)

    for key, x, y in zip(keys, xs, ys):
        if key == "PCA":
            color = "Reds"
            c = 0.5
        elif key == "Dict L1=1.0e-03":
            color = "Blues"
            c = 0.8
        elif key == "Dict L1=3.0e-04":
            color = "Blues"
            c = 0.6
        elif key == "Dict L1=1.0e-04":
            color = "Blues"
            c = 0.4
        elif key == "Dict L1=0.0e+00":
            color = "Blues"
            c = 0.2
        
        cmap = plt.get_cmap(color)
        c = cmap(c)

        ax.plot(x, y, color=c, linestyle="dashed", label=key, alpha=1)

    ax.set_xlabel("No. Uncorrupted Features")
    ax.set_ylabel("KL-Divergence From Base")

    # ax.set_xscale("log")

    ax.legend(
        loc="upper right",
        framealpha=1,
    )

    #shutil.rmtree("graphs", ignore_errors=True)
    #os.mkdir("graphs", exist_ok=True)

    plt.savefig(os.path.join(graph_folder, "score_size.png"))

    plt.close(fig)
    del fig, ax
