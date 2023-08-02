import standard_metrics
import matplotlib
import torch

import numpy as np

if __name__ == "__main__":
    chunk_range = list(range(8))
    learned_dict_files = [
        "output_4_rd_deep/_0/learned_dicts.pt",
        "output_4_rd_deep/_1/learned_dicts.pt",
        "output_4_rd_deep/_2/learned_dicts.pt",
        "output_4_rd_deep/_3/learned_dicts.pt",
        "output_4_rd_deep/_4/learned_dicts.pt",
        "output_4_rd_deep/_5/learned_dicts.pt",
        "output_4_rd_deep/_6/learned_dicts.pt",
        "output_4_rd_deep/_7/learned_dicts.pt",
    ]

    learned_dict_sets = [[torch.load(f) for f in set] for set in learned_dict_files]

    dataset = torch.load("pilechunks_l2_resid/0.pt")
    sample_idxs = np.random.choice(len(dataset), 5000, replace=False)

    device = torch.device("cuda:0")

    sample = dataset[sample_idxs].to(dtype=torch.float32, device=device)

    datapoint_sets = []
    for i, learned_dicts in enumerate(learned_dict_sets):
        datapoints = []
        for learned_dict_set in learned_dicts:
            datapoint_series = []
            for learned_dict, hyperparams in learned_dict_set:
                learned_dict.to_device(device)
                r_sq = standard_metrics.fraction_variance_unexplained(learned_dict, sample).item()
                sparsity = standard_metrics.mean_nonzero_activations(learned_dict, sample).sum().item()
                datapoint_series.append((r_sq, sparsity, hyperparams["l1_alpha"]))
            datapoints.append(datapoint_series)
        datapoint_sets.append(datapoints)

    colors = ["Purples", "Blues", "Greens", "Oranges", "Reds"]
    markers = ["o", "v", "s", "P", "X"]
    #labels = ["0.5", "1", "2", "4", "8"]
    #labels = [str(r) for r in learned_dict_files]
    labels = [str(r) for r in chunk_range]

    import matplotlib.pyplot as plt
    import math

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for k, datapoints in enumerate(datapoint_sets):
        for i, datapoint_series in enumerate(datapoints[:32]):
            r_sq, sparsity, l1_alpha = zip(*datapoint_series)
            ax.scatter(sparsity, r_sq, c=[math.log10(l1) for l1 in l1_alpha], label=labels[i], cmap=colors[i % len(colors)], vmin=-5, vmax=-2, marker=markers[k])
            if i == len(datapoints) - 1:
                # write the l1_alpha values on every 5th point and highlight them
                for j, (x, y) in enumerate(zip(sparsity, r_sq)):
                    if j % 5 == 0:
                        ax.annotate(f"{l1_alpha[j]:.1}", (x, y))
                        ax.scatter([x], [y], c="black")
    

    ax.set_xlabel("Mean no. features active")
    ax.set_ylabel("Unexplained Variance")
    ax.legend()
    plt.savefig("freq_plot_compare.png")
