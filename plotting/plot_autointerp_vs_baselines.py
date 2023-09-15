import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

base_path = "/mnt/ssd-cluster/auto_interp_results"
plots_folder = "/mnt/ssd-cluster/plots"

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from plotting.plot_autointerp_violins import read_scores


def read_all_layers(score_mode: str, layer_loc: str) -> None:
    layers = list(range(6))
    activation_names = [os.path.join(base_path, f"l{layer}_{layer_loc}") for layer in layers]
    all_scores: List[Dict] = []
    for activation_name in activation_names:
        results_folder = os.path.join(plots_folder, activation_name)
        scores = read_scores(
            results_folder, score_mode
        )  # Dict[str, Tuple[List[int], List[float]]], where the tuple is (feature_ndxs, scores)
        all_scores.append(scores)

    # transforms are only those that are in all layers
    transforms_set = set(all_scores[0].keys())
    for scores in all_scores:
        transforms_set = transforms_set.intersection(scores.keys())
    transforms = list(transforms_set)

    # filter transforms
    if layer_loc == "mlp":
        chosen_transforms = ["untied_r1.0_l1a0.00032"]
    else:
        chosen_transforms = ["tied_r2.0_l1a0.00086"]

    chosen_transforms += ["identity_relu", "random", "ica", "pca"]
    transforms = [transform for transform in transforms if transform in chosen_transforms]

    # sort transforms so that all with "tied" in the name come first, and then sort alphabetically
    transforms.sort()
    transforms.sort(key=lambda x: (x.find("tied") == -1, x))

    print(f"{transforms=}")

    # plot the scores as a grouped bar chart, with all scores from each layer together, showing mean and confidence interval
    # first we need to get the scores into a list of lists
    scores_list = [[scores[transform][1] for transform in transforms] for scores in all_scores]
    # remove any transforms that have no scores
    scores_list = [[scores for scores in scores_list if len(scores) > 0] for scores_list in scores_list]
    # now we need to get the means and confidence intervals for each transform
    means = [[np.mean(scores) for scores in scores_list] for scores_list in scores_list]
    cis = [[1.96 * np.std(scores, ddof=1) / np.sqrt(len(scores)) for scores in scores_list] for scores_list in scores_list]
    # now we can plot the bar chart
    plt.clf()  # clear the plot
    # set yaxis min to 0
    if score_mode == "random":
        top = 0.2
    else:
        top = 0.35
    plt.ylim(bottom=0, top=top)
    # add horizontal grid lines every 0.1
    plt.yticks(np.arange(0, top, 0.1))
    plt.grid(axis="y", color="grey", linestyle="-", linewidth=0.5, alpha=0.3)
    # add x labels
    plt.xticks(np.arange(1, len(layers) + 1), [f"{i}" for i in layers])
    # add standard errors around the means but don't plot the means
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
        "yellow",
        "lime",
    ]
    markers = [
        "o",
        "v",
        "^",
        "*",
        "x",
        "<",
        ">",
        "s",
        "p",
        "P",
        "h",
        "H",
        "+",
        "x",
        "X",
        "D",
        "d",
        "|",
        "_",
    ]
    for i in layers:
        for j, transform in enumerate(transforms):
            plt.errorbar(
                i + 1 + (j * 0.07),
                means[i][j],
                yerr=cis[i][j],
                fmt="o",
                color=colors[j % len(colors)],
                elinewidth=1,
                capsize=0,
                markersize=8,
                marker=markers[j % len(markers)],
            )

    plt.xlabel("Layer")
    plt.ylabel("Mean automated interpretability score")
    # remove bounding box, retain axes
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    # and a thicker line at 0
    # add a legend of colours for each transform, noting that it starts from 1
    transform_names = []
    for i, transform in enumerate(transforms):
        if "tied" in transform:
            transform_names.append(f"Sparse Coding")
        elif transform == "ica":
            transform_names.append("ICA")
        elif transform == "pca":
            transform_names.append("PCA")
        elif "identity" in transform:
            transform_names.append("Identity ReLU")
        elif transform == "random":
            transform_names.append("Random")   
        else:
            transform_names.append(f"{i+1}: {transform.replace('_', ' ')}")
    plt.legend(transform_names, loc="upper right")
    if layer_loc == "mlp":
        # Add score mode as a title
        if score_mode == "random":
            plt.title("Random-only scoring")
        elif score_mode == "top_random":
            plt.title("Top-and-random scoring")

    plt.axhline(y=0, linestyle="-", color="black", linewidth=1)
    plt.tight_layout()
    save_path = os.path.join(plots_folder, f"{layer_loc}_{score_mode}_means_and_cis_new.png")
    print(f"Saving means and cis graph to {save_path}")
    plt.savefig(save_path)


if __name__ == "__main__":
    read_all_layers("top", "residual")
    read_all_layers("random", "residual")
    read_all_layers("top_random", "residual")
    read_all_layers("top", "mlp")
    read_all_layers("random", "mlp")
    read_all_layers("top_random", "mlp")
