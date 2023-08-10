import matplotlib
import torch
import matplotlib.pyplot as plt
import numpy as np

import os
import shutil

if __name__ == "__main__":

    scores = torch.load("dict_scores_layer_2.pt")

    fig, ax = plt.subplots()

    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    markers = ["x", "+", "*", "o", "v", "^", "<", ">", "s", "."]

    for (color, marker), (key, score) in zip(zip(colors, markers), scores.items()):
        _, graph, div, corruption = zip(*score)
        graph_size = [len(g) for g in graph]
        print(key, graph_size, div)
        ax.scatter(corruption, div, label=key, color=color, marker=marker, alpha=0.5)

    ax.set_xlabel("Corruption")
    ax.set_ylabel("Task-Specific Loss")

    #ax.set_xscale("log")

    ax.legend()

    shutil.rmtree("graphs", ignore_errors=True)
    os.mkdir("graphs")

    plt.savefig("graphs/score_size.png")

    plt.close(fig)
    del fig, ax