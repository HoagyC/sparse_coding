import sys

sys.path.append("..")

import os
import shutil

from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_bottleneck_scores():
    base_folder = "sparse_coding_aidan"
    scores = torch.load(f"feat_ident_results.pt")

    #print(scores.keys())

    #diff_mean_scores = torch.load("diff_mean_scores_layer_2.pt")

    fig, ax = plt.subplots()

    ax.grid(True, alpha=0.5, linestyle="dashed")
    ax.set_axisbelow(True)

    colors = ["Reds", "Blues", "Greens", "Purples", "Oranges", "Greys"]
    #markers = ["x", "+", "*", "o", "v", "^", "<", ">", "s", "."]
    styles = ["solid", "dashed", "dashdot", "dotted"]

    xs, ys, keys = [], [], []
    for key, score in scores:
        graph, div, corruption = zip(*sorted(score, key=lambda x: len(x[0])))
        graph_size = [len(g) for g in graph]
        print(graph_size, div, corruption)
        xs.append(corruption)
        ys.append(div)
        keys.append(key)

    for key, x, y, (style, color) in zip(keys, xs, ys, product(styles, colors)):
        #style = "dashed"
        #if key == "pca":
        #    label = "PCA"
        #    color = "Reds"
        #    style = "dotted"
        #    c = 0.5
        #elif key == "learned_r4_1e-03":
        #    label = "Dict. alpha=1e-3"
        #    color = "Blues"
        #    c = 0.7
        #elif key == "learned_r4_3e-04":
        #    label = "Dict. alpha=3e-4"
        #    color = "Blues"
        #    c = 0.5
        #elif key == "learned_r4_1e-04":
        #    label = "Dict. alpha=1e-4"
        #    color = "Blues"
        #    c = 0.3
        
        c = 0.5
        label = key

        cmap = plt.get_cmap(color)
        c = cmap(c)

        ax.plot(x, y, color=c, linestyle=style, label=label, alpha=1)

    ax.set_xlabel("Mean Edit Magnitude")
    ax.set_ylabel("KL-Divergence From Base")

    ax.set_title(f"Precision-Accuracy Tradeoff Curve")

    #ax.set_xscale("log")

    #ax.set_yscale("log")

    #ax.set_xlim(0.6, 1.2)

    #ax.set_ylim(0, 1.2)

    ax.legend(
        #loc="upper left",
        framealpha=1,
    )

    #shutil.rmtree("graphs", ignore_errors=True)
    #os.mkdir("graphs", exist_ok=True)

    plt.savefig(f"graphs/feature_ident_curve.png")

    plt.close(fig)
    del fig, ax

if __name__ == "__main__":
    plot_bottleneck_scores()