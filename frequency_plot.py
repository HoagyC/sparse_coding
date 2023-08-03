import os

from typing import List, Tuple, Union

import standard_metrics
import matplotlib
import torch

import numpy as np

if __name__ == "__main__":
    chunk_range = [9]
    learned_dict_files = os.listdir("/mnt/ssd-cluster/bigrun0308")

    resid_dicts = [f for f in learned_dict_files if "resid" in f]
    mlp_dicts = [f for f in learned_dict_files if "mlp" in f]

    layer_0_dicts = [f for f in learned_dict_files if "layer_0" in f]
    layer_1_dicts = [f for f in learned_dict_files if "layer_1" in f]
    layer_2_dicts = [f for f in learned_dict_files if "layer_2" in f]
    layer_3_dicts = [f for f in learned_dict_files if "layer_3" in f]
    layer_4_dicts = [f for f in learned_dict_files if "layer_4" in f]
    layer_5_dicts = [f for f in learned_dict_files if "layer_5" in f]

    ratio0_5_dicts = [f for f in learned_dict_files if "r0" in f]
    ratio1_dicts = [f for f in learned_dict_files if "r1" in f]
    ratio2_dicts = [f for f in learned_dict_files if "r2" in f]
    ratio4_dicts = [f for f in learned_dict_files if "r4" in f]
    ratio8_dicts = [f for f in learned_dict_files if "r8" in f]
    ratio16_dicts = [f for f in learned_dict_files if "r16" in f]
    ratio32_dicts = [f for f in learned_dict_files if "r32" in f]

    learned_dict_locs = list(set(layer_0_dicts) & set(mlp_dicts) & set(ratio4_dicts))
    learned_dict_tuples = [(x.split("sweep_")[-1], torch.load(os.path.join("/mnt/ssd-cluster/bigrun0308", x))) for x in learned_dict_locs]

    dataset = torch.load("pilechunks_l2_resid/0.pt")
    sample_idxs = np.random.choice(len(dataset), 5000, replace=False)

    device = torch.device("cuda:0")

    sample = dataset[sample_idxs].to(dtype=torch.float32, device=device)

    datapoint_sets = []
    for i, (name, learned_dicts) in enumerate(learned_dict_tuples):
        datapoints: List[Tuple[float, float, float]] = []
        for learned_dict, hyperparams in learned_dicts:
            learned_dict.to_device(device)
            r_sq = standard_metrics.fraction_variance_unexplained(learned_dict, sample).item()
            sparsity = standard_metrics.mean_nonzero_activations(learned_dict, sample).sum().item()
            datapoints.append((r_sq, sparsity, hyperparams["l1_alpha"]))
        datapoint_sets.append((name, datapoints))

    colors = ["Purples", "Blues", "Greens", "Oranges", "Reds"]
    markers = ["o", "v", "s", "P", "X"]
    #labels = ["0.5", "1", "2", "4", "8"]
    #labels = [str(r) for r in learned_dict_files]

    import matplotlib.pyplot as plt
    import math

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for k, (name, datapoints) in enumerate(datapoint_sets):
        r_sq, sparsity, l1_alpha = zip(*datapoints)
        ax.scatter(sparsity, r_sq, c=[math.log10(l1) for l1 in l1_alpha], label=name, cmap=colors[k % len(colors)], vmin=-5, vmax=-2, marker=markers[k])
        if i == len(datapoints) - 1:
            # write the l1_alpha values on every 5th point and highlight them
            for j, (x, y) in enumerate(zip(sparsity, r_sq)):
                if j % 5 == 0:
                    ax.annotate(f"{l1_alpha[j]:.1}", (x, y))
                    ax.scatter([x], [y], c="black")
    

    ax.set_xlabel("Mean no. features active")
    ax.set_ylabel("Unexplained Variance")
    ax.legend()
    plt.savefig(f"freq_plot_compare_l0_r4_resid.png")
