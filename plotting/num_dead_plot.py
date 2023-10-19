from typing import List, Tuple

import matplotlib
import numpy as np
import torch

import standard_metrics

if __name__ == "__main__":
    learned_dict_files = [0.5, 1, 2, 4, 8]

    learned_dicts = [torch.load(f"output_{f}/_10/learned_dicts.pt") for f in learned_dict_files]

    datapoints = []

    dataset = torch.load("activation_data/0.pt")
    sample_idxs = np.random.choice(len(dataset), 100000, replace=False)

    sample = dataset[sample_idxs].to(torch.float32).to("cuda:0")

    subsample_sizes = [100000]

    for learned_dict_set in learned_dicts:
        datapoint_series = []
        for learned_dict, hyperparams in learned_dict_set:
            learned_dict.to_device("cuda:0")
            mean_nz = standard_metrics.mean_nonzero_activations(learned_dict, sample)
            sparsity = mean_nz.sum().item()
            # fuv = standard_metrics.fraction_variance_unexplained(learned_dict, sample).item()
            dead_count: List[Tuple[int, float, float]] = []
            for s in subsample_sizes:
                num_dead = mean_nz[:s].count_nonzero().item()
                datapoint_series.append((num_dead, sparsity, hyperparams["l1_alpha"]))
            del learned_dict
        datapoints.append(datapoint_series)

    colors = ["Purples", "Blues", "Greens", "Oranges", "Reds"]
    labels = ["0.5", "1", "2", "4", "8"]

    import math

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, datapoint_series in enumerate(datapoints):
        num_dead, sparsity, l1s = zip(*datapoint_series)
        ax.scatter(
            num_dead,
            sparsity,
            c=[math.log10(x) for x in l1s],
            label=labels[i],
            cmap=colors[i],
            vmin=-5,
            vmax=-1,
        )
    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Num Active Neurons")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    plt.savefig("num_dead_plot.png")
