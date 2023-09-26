import sys

sys.path.append("..")

import os
import shutil

from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

stylemap = {
    "dict_1.00e-03": ("Dict, α=1e-03, R=4", ("dashed", "D"), "Blues", 1.0),
    "dict_3.00e-04": ("Dict, α=3e-04, R=4", ("dashed", "D"), "Blues", 0.8),
    "dict_1.00e-04": ("Dict, α=1e-04, R=4", ("dashed", "D"), "Blues", 0.6),
    "dict_0.00e+00": ("Dict, α=0, R=4", ("dashed", "D"), "Blues", 0.4),
    "pca_rot": ("PCA", ("dashdot", "o"), "Reds", 0.3),
    "pca_pve": ("Nonneg. PCA", ("dashdot", "o"), "Oranges", 0.7),
}

def plot_bottleneck_scores():
    base_folder = "sparse_coding_aidan"
    scores = torch.load(f"feat_ident_results_410m_l11.pt")

    xs, ys, zs, keys = [], [], [], []
    for key, score in scores:
        graph, div, corruption = zip(*sorted(score, key=lambda x: len(x[0])))
        graph_size = [len(g) for g in graph]
        print(graph_size, div, corruption)
        xs.append(graph_size)
        zs.append(corruption)
        ys.append(div)
        keys.append(key)

    colors = ["Reds", "Blues", "Greens", "Purples", "Oranges", "Greys"]
    #markers = ["x", "+", "*", "o", "v", "^", "<", ">", "s", "."]
    styles = ["solid", "dashed", "dashdot", "dotted"]
    markers = ["x", "+", "*", "o"]

    #print(scores.keys())

    #diff_mean_scores = torch.load("diff_mean_scores_layer_2.pt")

    fig, ax = plt.subplots()

    ax.grid(True, alpha=0.5, linestyle="dashed")
    ax.set_axisbelow(True)

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
        
        #c = 0.5
        #label = key

        label, (style, _), color, c = stylemap[key]

        cmap = plt.get_cmap(color)
        c = cmap(c)

        ax.plot(x, y, color=c, linestyle=style, label=label, alpha=1)

    ax.set_xlabel("Number of Patched Features")
    ax.set_ylabel("KL-Divergence From Variant")

    ax.set_title(f"Task Divergence versus Number of Patched Features - Layer 11")

    #ax.set_xscale("log")

    #ax.set_yscale("log")

    ax.set_xlim(0, 512)

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

    fig, ax = plt.subplots()

    ax.grid(True, alpha=0.5, linestyle="dashed")
    ax.set_axisbelow(True)

    for key, x, y, (marker, color) in zip(keys, zs, ys, product(markers, colors)):
        #c = 0.5
        #label = key

        label, (style, marker), color, c = stylemap[key]
        
        cmap = plt.get_cmap(color)
        c = cmap(c)

        ax.plot(x, y, color=c, linestyle=style, marker=matplotlib.markers.MarkerStyle(marker).scaled(0.75), alpha=0.5, label=label)

    ax.set_xlabel("Mean Edit Magnitude")
    ax.set_ylabel("KL-Divergence From Variant")

    ax.set_title(f"Task Divergence versus Edit Magnitude - Layer 11")

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

    plt.savefig(f"graphs/feature_ident_curve_edit.png")

    plt.close(fig)
    del fig, ax

if __name__ == "__main__":
    plot_bottleneck_scores()