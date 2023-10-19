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

def plot_bottleneck_scores(layer, title=False):
    base_folder = "sparse_coding_aidan"
    scores = torch.load(f"ioi_feat/feat_ident_results_l{layer}.pt")

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

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(6.0 * 2 if not title else 4.8 * 2, 4.8))

    if title:
        fig.suptitle(f"Layer {layer}")

    ax1.grid(True, alpha=0.5, linestyle="dashed")
    ax1.set_axisbelow(True)

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

        ax1.plot(x, y, color=c, linestyle=style, label=label, alpha=1)

    ax1.set_xlabel("Number of Patched Features")
    ax1.set_ylabel("KL Divergence From Target")

    #ax.set_xscale("log")

    #ax.set_yscale("log")

    ax1.set_xlim(0, 512)

    #ax.set_ylim(0, 1.2)

    #shutil.rmtree("graphs", ignore_errors=True)
    #os.mkdir("graphs", exist_ok=True)

    ax2.grid(True, alpha=0.5, linestyle="dashed")
    ax2.set_axisbelow(True)

    for key, x, y, (marker, color) in zip(keys, zs, ys, product(markers, colors)):
        #c = 0.5
        #label = key

        label, (style, marker), color, c = stylemap[key]
        
        cmap = plt.get_cmap(color)
        c = cmap(c)

        ax2.plot(x, y, color=c, linestyle=style, marker=matplotlib.markers.MarkerStyle(marker).scaled(0.75), alpha=0.5, label=label)

    ax2.set_xlabel("Mean Edit Magnitude")
    #ax2.set_ylabel("KL Divergence From Target")

    #ax.set_xscale("log")

    #ax.set_yscale("log")

    #ax.set_xlim(0.6, 1.2)

    #ax.set_ylim(0, 1.2)

    ax2.legend(
        #loc="upper left",
        framealpha=1,
    )

    #shutil.rmtree("graphs", ignore_errors=True)
    #os.mkdir("graphs", exist_ok=True)

    plt.tight_layout()

    plt.savefig(f"graphs_ioi/feature_ident_curve_l{layer}.png")

    plt.close(fig)

if __name__ == "__main__":
    TITLE = True
    plot_bottleneck_scores(3, title=TITLE)
    plot_bottleneck_scores(5, title=TITLE)
    plot_bottleneck_scores(7, title=TITLE)
    plot_bottleneck_scores(11, title=TITLE)
    plot_bottleneck_scores(15, title=TITLE)
    plot_bottleneck_scores(19, title=TITLE)
    plot_bottleneck_scores(23, title=TITLE)