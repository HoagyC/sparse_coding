import os
import shutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

if __name__ == "__main__":
    base_folder = "sparse_coding_aidan"
    scores = torch.load(os.path.join(base_folder, "dict_scores_layer_2.pt"))

    diff_mean_scores = torch.load(os.path.join("diff_mean_scores_layer_2.pt"))

    fig, ax = plt.subplots()

    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    markers = ["x", "+", "*", "o", "v", "^", "<", ">", "s", "."]

    xs, ys, keys = [], [], []
    for key, score in scores.items():
        _, graph, div, corruption = zip(*score)
        graph_size = [len(g) for g in graph]
        xs.append(corruption)
        ys.append(div)
        keys.append(key)

    scales, corruption_AE, div_AE = zip(*diff_mean_scores)
    xs.append(corruption_AE)
    ys.append(div_AE)
    keys.append("diff-means editing")

    print(scales)

    for color, marker, key, x, y in zip(colors, markers, keys, xs, ys):
        ax.scatter(x, y, c=color, marker=marker, label=key, alpha=0.5)

    ax.set_xlabel("Corruption")
    ax.set_ylabel("Task-Specific Loss")

    # ax.set_xscale("log")

    ax.legend()
    graph_folder = os.path.join(base_folder, "graphs")

    shutil.rmtree(graph_folder, ignore_errors=True)
    os.mkdir(graph_folder)

    plt.savefig(os.path.join(graph_folder, "score_size.png"))

    plt.close(fig)
    del fig, ax
