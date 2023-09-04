import itertools
import os
import shutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

BASE_FOLDER = "~/sparse_coding_aidan"


def plot_bottleneck_scores():
    graphs_folder = os.path.join(BASE_FOLDER, "graphs")
    shutil.rmtree(graphs_folder, ignore_errors=True)
    os.mkdir(graphs_folder)

    scores = torch.load(os.path.join(BASE_FOLDER, "dict_scores_layer_3.pt"))

    print(list(scores.keys()))

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
    # markers = ["x", "+", "*", "o", "v", "^", "<", ">", "s", "."]
    styles = ["solid", "dashed", "dashdot", "dotted"]

    taus, sizes, task_metrics, corruptions, keys = [], [], [], [], []

    for key, score in scores.items():
        tau, graph, task_metric, corruption = zip(*score)
        taus.append(tau)
        sizes.append([len(g) for g in graph])
        task_metrics.append(task_metric)
        corruptions.append(corruption)
        keys.append(key)

    fig, ax = plt.subplots()

    for (style, color), key, x, y in zip(itertools.product(styles, colors), keys, sizes, task_metrics):
        print(key)
        ax.plot(x, y, c=color, linestyle=style, label=key, alpha=0.5)

    ax.set_xlabel("Bottleneck Size")
    ax.set_ylabel("Per-Task Metric")

    ax.legend()

    fig.savefig(os.path.join(graphs_folder, "bottleneck_scores.png"))


def plot_erasure_scores():
    graphs_folder = os.path.join(BASE_FOLDER, "graphs")
    shutil.rmtree(graphs_folder, ignore_errors=True)
    os.mkdir(graphs_folder)

    leace_score, leace_edit, base_score = torch.load(os.path.join(BASE_FOLDER, "leace_scores_layer_2.pt"))

    scores = torch.load(os.path.join(BASE_FOLDER, "erasure_scores_layer_2.pt"))

    kl_divs = torch.load(os.path.join(BASE_FOLDER, "kl_div_scores_layer_2.pt"))

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

    edit_sizes, prediction_ability, kl_div_scores, keys = [], [], [], []
    for key, score in scores.items():
        _, pred, corruption = zip(*score)
        kl_div = [kl_divs[key + "_" + str(i)] for i in range(len(score))]
        edit_sizes.append(corruption)
        prediction_ability.append(pred)
        kl_div_scores.append(kl_div)
        keys.append(key)

    edit_sizes.append([leace_edit])
    prediction_ability.append([leace_score])
    kl_div_scores.append([kl_divs["LEACE"]])
    keys.append("LEACE")

    fig, ax = plt.subplots()

    for color, marker, key, x, y in zip(colors, markers, keys, edit_sizes, prediction_ability):
        ax.scatter(x, y, c=color, marker=marker, label=key, alpha=0.5)

    ax.axhline(y=base_score, color="red", linestyle="dashed", label="Base")

    ax.set_xlabel("Mean Edit")
    ax.set_ylabel("Prediction Ability")

    ax.legend()

    plt.savefig(os.path.join(graphs_folder, "erasure_by_edit_magnitude.png"))

    plt.close(fig)
    del fig, ax

    fig, ax = plt.subplots()

    for color, marker, key, x, y in zip(colors, markers, keys, kl_div_scores, prediction_ability):
        ax.scatter(x, y, c=color, marker=marker, label=key, alpha=0.5)

    ax.axhline(y=base_score, color="red", linestyle="dashed", label="Base")

    ax.set_xlabel("KL Divergence")
    ax.set_ylabel("Prediction Ability")

    ax.legend()

    plt.savefig(os.path.join(graphs_folder, "erasure_by_kl_div.png"))


if __name__ == "__main__":
    # plot_bottleneck_scores()
    plot_erasure_scores()
