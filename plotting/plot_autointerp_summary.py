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
        scores = read_scores(results_folder, score_mode) # Dict[str, Tuple[List[int], List[float]]], where the tuple is (feature_ndxs, scores)
        all_scores.append(scores)
    
    # transforms are only those that are in all layers
    transforms_set = set(all_scores[0].keys())
    for scores in all_scores:
        transforms_set = transforms_set.intersection(scores.keys())
    transforms = list(transforms_set)

    # sort transforms so that all with "tied" in the name come first, and then sort alphabetically
    transforms.sort()
    transforms.sort(key=lambda x: (x.find("tied") == -1, x))

    print(f"{transforms=}")

    # plot the scores as a grouped bar chart, with all scores from each layer together, showing mean and confidence interval
    # first we need to get the scores into a list of lists
    scores_list = [[scores[transform][1] for transform in transforms] for scores in all_scores]
    # remove any transforms that have no scores
    scores_list = [[scores for scores in scores_list if len(scores) > 0] for scores_list in scores_list]
    # now we need to get the means and confidence intervals for each transform
    means = [[np.mean(scores) for scores in scores_list] for scores_list in scores_list]
    cis = [[1.96 * np.std(scores, ddof=1) / np.sqrt(len(scores)) for scores in scores_list] for scores_list in scores_list]
    # now we can plot the bar chart
    plt.clf() # clear the plot
    # fix yrange from -0.2 to 0.6
    plt.ylim(-0.2, 0.6)
    # add horizontal grid lines every 0.1
    plt.yticks(np.arange(-0.2, 0.6, 0.1))
    plt.grid(axis="y", color="grey", linestyle="-", linewidth=0.5, alpha=0.3)
    # add x labels
    plt.xticks(np.arange(1, len(layers) + 1), [f"Layer {i}" for i in layers], rotation=90)
    # add standard errors around the means but don't plot the means
    colors = ["red", "blue", "green", "orange", "purple", "pink", "black", "brown", "cyan", "magenta", "grey", "yellow", "lime"]
    for i in layers:
        for j, transform in enumerate(transforms):
            if "tied" in transform and not "032" in transform:
                continue
            # note, n transforms is 13
            plt.errorbar(i + 1 + (j * 0.05), means[i][j], yerr=cis[i][j], fmt="o", color=colors[j % len(colors)], elinewidth=1, capsize=0)
        
    plt.title(f"{layer_loc} {score_mode}")
    plt.xlabel("Transform")
    plt.ylabel("GPT-4-based interpretability score")
    plt.xticks(rotation=90)
    # and a thicker line at 0
    # add a legend of colours for each transform, noting that it starts from 1
    plt.legend([f"{i+1}: {transform}" for i, transform in enumerate(transforms)], loc="upper left", bbox_to_anchor=(1, 1))
    plt.axhline(y=0, linestyle="-", color="black", linewidth=1)
    plt.tight_layout()
    save_path = os.path.join(plots_folder, f"{layer_loc}_{score_mode}_means_and_cis.png")
    print(f"Saving means and cis graph to {save_path}")
    plt.savefig(save_path)

if __name__ == "__main__":
    # read_all_layers("top", "residual")
    # read_all_layers("random", "residual")
    # read_all_layers("top_random", "residual")
    read_all_layers("top", "mlp")
    read_all_layers("random", "mlp")
    read_all_layers("top_random", "mlp")
            