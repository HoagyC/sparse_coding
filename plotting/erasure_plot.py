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

def plot_leace_scores_across_depth(title="various settings", name="various_settings"):
    layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 23]

    files = [
        torch.load(f"output_erasure_410m/general_{layer}_gender.pt")
        for layer in layers
    ]

    from matplotlib.legend_handler import HandlerTuple

    leace_scores = [files[l]["leace"][0] for l in range(len(layers))]
    mean_scores = [files[l]["mean"][0] for l in range(len(layers))]
    mean_affine_scores = [files[l]["mean_affine"][0] for l in range(len(layers))]

    base_score = files[0]["base"]

    fig, (ax2, ax1) = plt.subplots(2, 1, sharex=True)

    ax1.grid(True, alpha=0.5, linestyle="dashed")
    ax1.set_axisbelow(True)

    ax1.plot(leace_scores, label="LEACE", marker="+")
    ax1.plot(mean_scores, label="Mean", marker="x")
    ax1.plot(mean_affine_scores, label="Mean, Affine", marker=".")

    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers)

    ax1.axhline(y=base_score, color="red", linestyle="dashed", label="Base Perf.")
    ax1.axhline(y=0.5, color="grey", linestyle="dashed", label="Majority")

    #ax1.set_xlabel("Layer")
    ax1.set_ylabel("Model Prediction Ability")

    leace_edits = [files[l]["leace"][1] for l in range(len(layers))]
    mean_edits = [files[l]["mean"][1] for l in range(len(layers))]
    mean_affine_edits = [files[l]["mean_affine"][1] for l in range(len(layers))]

    ax2.grid(True, alpha=0.5, linestyle="dashed")
    ax2.set_axisbelow(True)

    ax2.plot(leace_edits, label="LEACE", marker="+")
    ax2.plot(mean_edits, label="Mean", marker="x")
    ax2.plot(mean_affine_edits, label="Mean, Affine", marker=".")

    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Mean Edit Magnitude")

    #ax.set_yscale("log")

    ax2.set_ylim(bottom=0)

    handles, labels = ax1.get_legend_handles_labels()
    ax2.legend(
        handles,
        labels,
        loc='upper center',
        facecolor="white",
        framealpha=1,
        ncol=2,
    )

    fig.suptitle(title)

    plt.savefig(f"graphs/erasure_across_depth_410m_{name}.png")

def plot_scores_across_depth(both_datasets=True):
    layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

    files = [
        torch.load(f"output_erasure_410m/eval_layer_{layer}_gender.pt")
        for layer in layers
    ]

    transfer_files = [
        torch.load(f"output_erasure_410m/eval_layer_{layer}_pronoun.pt")
        for layer in layers
    ]

    do_dataset_plot(files, "gender", layers, "Concept Erasure on the Primary Task")
    do_dataset_plot(transfer_files, "pronoun", layers, "Transferred Concept Erasure on the Secondary Task")

def do_dataset_plot(files, name, layers, title):
    from matplotlib.legend_handler import HandlerTuple

    leace_scores = [files[l]["leace"][0] for l in range(len(layers))]
    mean_scores = [files[l]["means"][0] for l in range(len(layers))]
    max_dict_scores = [files[l]["dict"][0][1] for l in range(len(layers))]
    max_rand_scores = [files[l]["random"][0][1] for l in range(len(layers))]

    base_score = files[0]["base"]

    fig, (ax2, ax1) = plt.subplots(2, 1, sharex=True)

    ax1.grid(True, alpha=0.5, linestyle="dashed")
    ax1.set_axisbelow(True)

    #ax1.plot(leace_scores, label="LEACE", marker="+")
    ax1.plot(mean_scores, label="Mean", marker="x", color="orange")
    ax1.plot(max_dict_scores, label="Dict. Feature", marker=".", color="green")
    ax1.plot(max_rand_scores, label="Rand. Feature", marker=".", color="red")

    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers)

    ax1.axhline(y=base_score, color="red", linestyle="dashed", label="Base Perf.")
    ax1.axhline(y=0.5, color="grey", linestyle="dashed", label="Majority")

    #ax1.set_xlabel("Layer")
    ax1.set_ylabel("Model Prediction Ability")

    leace_edits = [files[l]["leace"][1] for l in range(len(layers))]
    mean_edits = [files[l]["means"][1] for l in range(len(layers))]
    max_dict_edits = [files[l]["dict"][0][2] for l in range(len(layers))]
    max_rand_edits = [files[l]["random"][0][2] for l in range(len(layers))]

    ax2.grid(True, alpha=0.5, linestyle="dashed")
    ax2.set_axisbelow(True)

    #ax2.plot(leace_edits, label="LEACE", marker="+")
    ax2.plot(mean_edits, label="Mean", marker="x", color="orange")
    ax2.plot(max_dict_edits, label="Dict Feature", marker=".", color="green")
    ax2.plot(max_rand_edits, label="Rand. Feature", marker=".", color="red")

    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Mean Edit Magnitude")

    #ax.set_yscale("log")

    ax2.set_ylim(bottom=0)

    handles, labels = ax1.get_legend_handles_labels()
    ax2.legend(
        handles,
        labels,
        loc='upper center',
        facecolor="white",
        framealpha=1,
        ncol=2,
    )

    fig.suptitle(title)

    plt.savefig(f"graphs/erasure_across_depth_410m_{name}.png")

def plot_kl_div_across_depth():
    layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

    files = [
        torch.load(f"output_erasure_410m/kl_div_scores_layer_{layer}.pt")
        for layer in layers
    ]

    from matplotlib.legend_handler import HandlerTuple

    leace_scores = [files[l]["LEACE"][0] for l in range(len(layers))]
    mean_scores = [files[l]["means"][0] for l in range(len(layers))]
    max_dict_scores = [files[l]["dict"][0] for l in range(len(layers))]
    max_rand_scores = [files[l]["random"][0] for l in range(len(layers))]

    fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(6, 3))

    ax1.grid(True, alpha=0.5, linestyle="dashed")
    ax1.set_axisbelow(True)

    ax1.plot(leace_scores, label="LEACE", marker="+")
    ax1.plot(mean_scores, label="Mean", marker="x")
    ax1.plot(max_dict_scores, label="Dict. Feature", marker=".")
    ax1.plot(max_rand_scores, label="Rand. Feature", marker=".")

    #ax1.plot([4, 5], [mean_scores[4], 1], color="#ff7f0e", linestyle="dashed")
    #ax1.scatter([5], [0.2], marker="^", color="#ff7f0e")
    #ax1.text(5, 0.185, f"{mean_scores[-1]:.1f}", va="center", ha="center")

    #ax1.set_ylim(-0.01, 0.21)

    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers)

    ax1.set_yscale("log")

    #ax1.axhline(y=base_score, color="red", linestyle="dashed", label="Base Perf.")
    #ax1.axhline(y=0, color="grey", linestyle="dashed")

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("KL-Divergence")

    fig.suptitle("KL-Divergence From Base Model Under Erasure")

    handles, labels = ax1.get_legend_handles_labels()

    ax1.legend(
        handles,
        labels,
        facecolor="white",
        framealpha=1,
        loc="upper left"
    )

    fig.tight_layout()

    plt.savefig("graphs/kl_across_depth.png")

if __name__ == "__main__":
    # plot_bottleneck_scores()
    plot_scores_across_depth()
    #plot_kl_div_across_depth()
    #plot_leace_scores_across_depth()