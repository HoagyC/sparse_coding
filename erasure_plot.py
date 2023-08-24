import matplotlib
import torch
import matplotlib.pyplot as plt
import numpy as np

import os
import shutil

import itertools

def plot_bottleneck_scores():
    shutil.rmtree("graphs", ignore_errors=True)
    os.mkdir("graphs")

    scores = torch.load("dict_scores_layer_3.pt")

    print(list(scores.keys()))

    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    #markers = ["x", "+", "*", "o", "v", "^", "<", ">", "s", "."]
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

    fig.savefig("graphs/bottleneck_scores.png")

def plot_erasure_scores():
    shutil.rmtree("graphs", ignore_errors=True)
    os.mkdir("graphs")

    leace_score, leace_edit, base_score = torch.load("leace_scores_layer_2.pt")

    scores = torch.load("erasure_scores_layer_2.pt")

    kl_divs = torch.load("kl_div_scores_layer_2.pt")

    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
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

    plt.savefig("graphs/erasure_by_edit_magnitude.png")

    plt.close(fig)
    del fig, ax

    fig, ax = plt.subplots()

    for color, marker, key, x, y in zip(colors, markers, keys, kl_div_scores, prediction_ability):
        ax.scatter(x, y, c=color, marker=marker, label=key, alpha=0.5)
    
    ax.axhline(y=base_score, color="red", linestyle="dashed", label="Base")

    ax.set_xlabel("KL Divergence")
    ax.set_ylabel("Prediction Ability")

    ax.legend()

    plt.savefig("graphs/erasure_by_kl_div.png")

def plot_scores_across_depth():
    files = [
        torch.load(f"output_erasure_last_position_only/eval_layer_{layer}.pt")
        for layer in range(0, 6)
    ]

    from matplotlib.legend_handler import HandlerTuple

    leace_scores = [files[l]["leace"][0] for l in range(0, 6)]
    mean_scores = [files[l]["means"][0] for l in range(0, 6)]
    max_dict_scores = [min([x for _, x, _ in files[l]["dict"]]) for l in range(0, 6)]
    base_score = files[0]["base"]

    fig, (ax2, ax1) = plt.subplots(2, 1, sharex=True)

    ax1.grid(True, alpha=0.5, linestyle="dashed")
    ax1.set_axisbelow(True)

    ax1.plot(leace_scores, label="LEACE", marker=".")
    ax1.plot(mean_scores, label="Mean", marker=".")
    ax1.plot(max_dict_scores, label="Dict. Feature", marker=".")

    ax1.set_xticks(range(0, 6))
    ax1.set_xticklabels(range(0, 6))

    ax1.axhline(y=base_score, color="red", linestyle="dashed", label="Base Perf.")
    ax1.axhline(y=0.5, color="grey", linestyle="dashed", label="Majority")

    #ax1.set_xlabel("Layer")
    ax1.set_ylabel("Prediction Ability")

    leace_edits = [files[l]["leace"][1] for l in range(0, 6)]
    mean_edits = [files[l]["means"][1] for l in range(0, 6)]
    max_dict_edits = [max([(x,y) for _, x, y in files[l]["dict"]], key=lambda p: p[0])[1] for l in range(0, 6)]

    ax2.grid(True, alpha=0.5, linestyle="dashed")
    ax2.set_axisbelow(True)

    ax2.plot(leace_edits, label="LEACE", marker=".")
    ax2.plot(mean_edits, label="Mean", marker=".")
    ax2.plot(max_dict_edits, label="Dict Feature", marker=".")

    ax2.set_xticks(range(0, 6))
    ax2.set_xticklabels(range(0, 6))

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Mean Edit Magnitude")

    #ax.set_yscale("log")

    ax2.set_ylim(bottom=0)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        facecolor="white",
        framealpha=1,
    )

    plt.savefig("graphs/erasure_across_depth_last_pos.png")

def plot_kl_div_across_depth():
    files = [
        torch.load(f"output_erasure/kl_div_scores_layer_{layer}.pt")
        for layer in range(0, 6)
    ]

    from matplotlib.legend_handler import HandlerTuple

    leace_scores = [files[l]["LEACE"][0] for l in range(0, 6)]
    mean_scores = [files[l]["means"][0] for l in range(0, 6)]
    max_dict_scores = [files[l]["dict"][0] for l in range(0, 6)]

    fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(6, 3))

    ax1.grid(True, alpha=0.5, linestyle="dashed")
    ax1.set_axisbelow(True)

    ax1.plot(np.arange(6), leace_scores, label="LEACE", marker=".")
    ax1.plot(np.arange(5), mean_scores[:5], label="Mean", marker=".")
    ax1.plot(np.arange(6), max_dict_scores, label="Dict. Feature", marker=".")

    ax1.plot([4, 5], [mean_scores[4], 1], color="#ff7f0e", linestyle="dashed")
    ax1.scatter([5], [0.2], marker="^", color="#ff7f0e")
    ax1.text(5, 0.185, f"{mean_scores[-1]:.1f}", va="center", ha="center")

    ax1.set_ylim(-0.01, 0.21)

    ax1.set_xticks(range(0, 6))
    ax1.set_xticklabels(range(0, 6))

    #ax1.axhline(y=base_score, color="red", linestyle="dashed", label="Base Perf.")
    #ax1.axhline(y=0, color="grey", linestyle="dashed")

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("KL-Divergence")

    handles, labels = ax1.get_legend_handles_labels()

    ax1.legend(
        handles,
        labels,
        facecolor="white",
        framealpha=1,
        loc="center right"
    )

    fig.tight_layout()

    plt.savefig("graphs/kl_across_depth.png")

if __name__ == "__main__":
    #plot_bottleneck_scores()
    #plot_erasure_scores()
    plot_scores_across_depth()
    #plot_kl_div_across_depth()