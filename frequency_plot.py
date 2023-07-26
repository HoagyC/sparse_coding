import standard_metrics
import matplotlib
import torch

import numpy as np

if __name__ == "__main__":
    learned_dict_files = [
        "output_4_rd_deep/_1/learned_dicts.pt",
        "output_4_rd_deep/_2/learned_dicts.pt",
        "output_4_rd_deep/_3/learned_dicts.pt",
        "output_4_rd_deep/_4/learned_dicts.pt",
        "output_4_rd_deep/_5/learned_dicts.pt",
        "output_4_rd_deep/_6/learned_dicts.pt",
        "output_4_rd_deep/_7/learned_dicts.pt",
    ]

    learned_dicts = [torch.load(f) for f in learned_dict_files]

    datapoints = []

    dataset = torch.load("activation_data/0.pt")
    sample_idxs = np.random.choice(len(dataset), 5000, replace=False)

    device = torch.device("cuda:0")

    sample = dataset[sample_idxs].to(dtype=torch.float32, device=device)

    for learned_dict_set in learned_dicts:
        datapoint_series = []
        for learned_dict, hyperparams in learned_dict_set:
            learned_dict.to_device(device)
            r_sq = standard_metrics.fraction_variance_unexplained(learned_dict, sample).item()
            sparsity = standard_metrics.mean_nonzero_activations(learned_dict, sample).sum().item()
            datapoint_series.append((r_sq, sparsity, hyperparams["l1_alpha"]))
        datapoints.append(datapoint_series)
    
    colors = ["Purples", "Blues", "Greens", "Oranges", "Reds", "Purples", "Blues", "Greens", "Oranges", "Reds"]
    #labels = ["0.5", "1", "2", "4", "8"]
    #labels = [str(r) for r in learned_dict_files]
    labels = [str(r) for r in range(8)]

    import matplotlib.pyplot as plt
    import math

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, datapoint_series in enumerate(datapoints):
        r_sq, sparsity, l1_alpha = zip(*datapoint_series)
        ax.scatter(sparsity, r_sq, c=[math.log10(l1) for l1 in l1_alpha], label=labels[i], cmap=colors[i], vmin=-5, vmax=-2)
    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Unexplained Variance")
    ax.legend()
    plt.savefig("freq_plot.png")