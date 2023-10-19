import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

base_path = "/mnt/ssd-cluster/auto_interp_results"
plots_folder = "/mnt/ssd-cluster/plots"


def get_score(lines: List[str], mode: str):
    if mode == "top":
        return float(lines[-3].split(" ")[-1])
    elif mode == "random":
        return float(lines[-2].split(" ")[-1])
    elif mode == "top_random":
        score_line = [line for line in lines if "Score: " in line][0]
        return float(score_line.split(" ")[1])
    else:
        raise ValueError(f"Unknown mode: {mode}")


def read_scores(results_folder: str, score_mode: str = "top") -> Dict[str, Tuple[List[int], List[float]]]:
    scores: Dict[str, Tuple[List[int], List[float]]] = {}
    transforms = os.listdir(results_folder)
    transforms = [transform for transform in transforms if os.path.isdir(os.path.join(results_folder, transform))]
    if "sparse_coding" in transforms:
        transforms.remove("sparse_coding")
        transforms = ["sparse_coding"] + transforms

    for transform in transforms:
        transform_scores = []
        transform_ndxs = []
        # list all the features by looking for folders
        feat_folders = [x for x in os.listdir(os.path.join(results_folder, transform)) if x.startswith("feature_")]
        if len(feat_folders) == 0:
            continue
        print(f"{transform=}, {len(feat_folders)=}")
        for feature_folder in feat_folders:
            feature_ndx = int(feature_folder.split("_")[1])
            folder = os.path.join(results_folder, transform, feature_folder)
            if not os.path.exists(folder) or not os.path.exists(os.path.join(folder, "explanation.txt")):
                continue
            explanation_text = open(os.path.join(folder, "explanation.txt")).read()
            # score should be on the second line but if explanation had newlines could be on the third or below
            # score = float(explanation_text.split("\n")[1].split(" ")[1])
            lines = explanation_text.split("\n")
            score = get_score(lines, score_mode)

            print(f"{feature_ndx=}, {transform=}, {score=}")
            transform_scores.append(score)
            transform_ndxs.append(feature_ndx)

        scores[transform] = (transform_ndxs, transform_scores)

    return scores


def read_results(activation_name: str, score_mode: str, plots_folder: str) -> None:
    results_folder = os.path.join(plots_folder, activation_name)

    scores = read_scores(
        results_folder, score_mode
    )  # Dict[str, Tuple[List[int], List[float]]], where the tuple is (feature_ndxs, scores)
    if len(scores) == 0:
        print(f"No scores found for {activation_name}")
        return
    transforms = scores.keys()

    plt.clf()  # clear the plot

    # plot the scores as a violin plot
    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "pink",
        "black",
        "brown",
        "cyan",
        "magenta",
        "grey",
    ]

    # fix yrange from -0.2 to 0.6
    plt.ylim(-0.2, 0.6)
    # add horizontal grid lines every 0.1
    plt.yticks(np.arange(-0.2, 0.6, 0.1))
    plt.grid(axis="y", color="grey", linestyle="-", linewidth=0.5, alpha=0.3)
    # first we need to get the scores into a list of lists
    scores_list = [scores[transform][1] for transform in transforms]
    # remove any transforms that have no scores
    scores_list = [scores for scores in scores_list if len(scores) > 0]
    violin_parts = plt.violinplot(scores_list, showmeans=False, showextrema=False)
    for i, pc in enumerate(violin_parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_edgecolor(colors[i % len(colors)])
        pc.set_alpha(0.3)

    # add x labels
    plt.xticks(np.arange(1, len(transforms) + 1), transforms, rotation=90)

    # add standard errors around the means but don't plot the means
    cis = [1.96 * np.std(scores[transform][1], ddof=1) / np.sqrt(len(scores[transform][1])) for transform in transforms]
    for i, transform in enumerate(transforms):
        plt.errorbar(
            i + 1,
            np.mean(scores[transform][1]),
            yerr=cis[i],
            fmt="o",
            color=colors[i % len(colors)],
            elinewidth=2,
            capsize=20,
        )

    plt.title(f"{activation_name} {score_mode}")
    plt.xlabel("Transform")
    plt.ylabel("GPT-4-based interpretability score")
    plt.xticks(rotation=90)

    # and a thicker line at 0
    plt.axhline(y=0, linestyle="-", color="black", linewidth=1)

    plt.tight_layout()
    save_path = os.path.join(results_folder, f"{score_mode}_means_and_violin.png")
    print(f"Saving means and violin graph to {save_path}")
    plt.savefig(save_path)


if __name__ == "__main__":
    score_modes = ["top", "random", "top_random"]

    activation_names = [x for x in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, x))]

    for activation_name in activation_names:
        for score_mode in score_modes:
            read_results(activation_name, score_mode, plots_folder)
