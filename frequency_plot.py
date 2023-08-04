import os
import math
from typing import List, Tuple

import torch
import matplotlib.pyplot as plt
import numpy as np

import standard_metrics

if __name__ == "__main__":
    chunk_range = [9]
    learned_dict_files = os.listdir("/mnt/ssd-cluster/bigrun0308")

    resid_dicts = [f for f in learned_dict_files if "resid" in f]
    mlp_dicts = [f for f in learned_dict_files if "mlp" in f]

    layer_0_dicts = [f for f in learned_dict_files if "l0" in f]
    layer_1_dicts = [f for f in learned_dict_files if "l1" in f]
    layer_2_dicts = [f for f in learned_dict_files if "l2" in f]
    layer_3_dicts = [f for f in learned_dict_files if "l3" in f]
    layer_4_dicts = [f for f in learned_dict_files if "l4" in f]
    layer_5_dicts = [f for f in learned_dict_files if "l5" in f]

    ratio0_5_dicts = [f for f in learned_dict_files if "r0" in f]
    ratio1_dicts = [f for f in learned_dict_files if "r1" in f]
    ratio2_dicts = [f for f in learned_dict_files if "r2" in f]
    ratio4_dicts = [f for f in learned_dict_files if "r4" in f]
    ratio8_dicts = [f for f in learned_dict_files if "r8" in f]
    ratio16_dicts = [f for f in learned_dict_files if "r16" in f]
    ratio32_dicts = [f for f in learned_dict_files if "r32" in f]

    experiments = [
        ("l0_resid", [layer_0_dicts, resid_dicts]),
        ("l1_resid", [layer_1_dicts, resid_dicts]),
        ("l2_resid", [layer_2_dicts, resid_dicts]),
        ("l3_resid", [layer_3_dicts, resid_dicts]),
        ("l4_resid", [layer_4_dicts, resid_dicts]),
        ("l5_resid", [layer_5_dicts, resid_dicts]),
        ("l0_mlp", [layer_0_dicts, mlp_dicts]),
        ("l1_mlp", [layer_1_dicts, mlp_dicts]),
        ("l2_mlp", [layer_2_dicts, mlp_dicts]),
        ("l3_mlp", [layer_3_dicts, mlp_dicts]),
        ("l4_mlp", [layer_4_dicts, mlp_dicts]),
        ("l5_mlp", [layer_5_dicts, mlp_dicts]),
    ]
    
    for graph_name, categories in experiments:

        learned_dict_locs = list(set.intersection(*[set(x) for x in categories]))
        learned_dict_locs.sort(key=lambda x: int(x.split("_")[-1][1:])) # sort by ratio
        print(f"Found {len(learned_dict_locs)} lists of dicts for experiment {graph_name}")

        learned_dict_tuples = [(x.split("sweep_")[-1], torch.load(os.path.join("/mnt/ssd-cluster/bigrun0308", x, "_9", "learned_dicts.pt"))) for x in learned_dict_locs]

        dataset = torch.load(f"/mnt/ssd-cluster/single_chunks/{graph_name}/0.pt")
        sample_idxs = np.random.choice(len(dataset), 5000, replace=False)

        device = torch.device("cuda:0")

        sample = dataset[sample_idxs].to(dtype=torch.float32, device=device)

        datapoint_sets = []
        for i, (run_name, learned_dicts) in enumerate(learned_dict_tuples):
            datapoints: List[Tuple[float, float, float]] = []
            for learned_dict, hyperparams in learned_dicts:
                learned_dict.to_device(device)
                r_sq = standard_metrics.fraction_variance_unexplained(learned_dict, sample).item()
                sparsity = standard_metrics.mean_nonzero_activations(learned_dict, sample).sum().item()
                datapoints.append((r_sq, sparsity, hyperparams["l1_alpha"]))
            datapoint_sets.append((run_name, datapoints))

        colors = ["Purples", "Blues", "Greens", "Oranges", "Reds", "Greys", "YlOrBr", "YlOrRd", "OrRd"]
        markers = ["o", "v", "s", "P", "X"]
        #labels = ["0.5", "1", "2", "4", "8"]
        #labels = [str(r) for r in learned_dict_files]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for k, (run_name, datapoints) in enumerate(datapoint_sets):
            r_sq, sparsity, l1_alpha = zip(*datapoints)
            ax.scatter(sparsity, r_sq, c=[math.log10(l1) for l1 in l1_alpha], label=run_name, cmap=colors[k % len(colors)], vmin=-5, vmax=-2)
            if i == len(datapoints) - 1:
                # write the l1_alpha values on every 5th point and highlight them
                for j, (x, y) in enumerate(zip(sparsity, r_sq)):
                    if j % 5 == 0:
                        ax.annotate(f"{l1_alpha[j]:.1}", (x, y))
                        ax.scatter([x], [y], c="black")
        

        ax.set_xlabel("Mean no. features active")
        ax.set_ylabel("Unexplained Variance")
        ax.legend()
        plt.savefig(f"freq_plot_compare_{graph_name}.png")
