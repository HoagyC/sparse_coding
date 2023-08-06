import os
import math
from typing import List, Tuple, Dict, Any

import torch
import matplotlib.pyplot as plt
import numpy as np

from autoencoders.learned_dict import LearnedDict
import standard_metrics

if __name__ == "__main__":
    chunk_range = [9]
    learned_dict_files = [os.path.join("/mnt/ssd-cluster/bigrun0308", x) for x in os.listdir("/mnt/ssd-cluster/bigrun0308")]
    learned_dict_files += [f for f in os.listdir(".") if f.startswith("output_attn")]
    learned_dict_files += [f for f in os.listdir(".") if f.startswith("output_sweep")]

    resid_dicts = [f for f in learned_dict_files if "resid" in f]
    mlp_dicts = [f for f in learned_dict_files if "mlp" in f]
    mlp_dicts = [f for f in mlp_dicts if "mlp_out" not in f]
    attn_dicts = [f for f in learned_dict_files if "attn" in f]
    mlp_out_dicts = [f for f in learned_dict_files if "mlp_out" in f]

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

    tied_dicts = [f for f in learned_dict_files if "tied" in f]
    untied_dicts = [f for f in learned_dict_files if not "tied" in f]

    # experiments = [
    #     ("l0_resid", [layer_0_dicts, resid_dicts]),
    #     ("l1_resid", [layer_1_dicts, resid_dicts]),
    #     ("l2_resid", [layer_2_dicts, resid_dicts]),
    #     ("l3_resid", [layer_3_dicts, resid_dicts]),
    #     ("l4_resid", [layer_4_dicts, resid_dicts]),
    #     ("l5_resid", [layer_5_dicts, resid_dicts]),
    #     ("l0_mlp", [layer_0_dicts, mlp_dicts]),
    #     ("l1_mlp", [layer_1_dicts, mlp_dicts]),
    #     ("l2_mlp", [layer_2_dicts, mlp_dicts]),
    #     ("l3_mlp", [layer_3_dicts, mlp_dicts]),
    #     ("l4_mlp", [layer_4_dicts, mlp_dicts]),
    #     ("l5_mlp", [layer_5_dicts, mlp_dicts]),
    # ]
    experiments = [
        ("l0_resid", [[layer_0_dicts, resid_dicts]]),
        # ("l0_mlp", [[layer_0_dicts, mlp_dicts, tied_dicts], [layer_0_dicts, mlp_dicts, untied_dicts]]),
        ("l0_attn", [[layer_0_dicts, attn_dicts]]),
        ("l1_attn", [[layer_1_dicts, attn_dicts]]),
        ("l2_attn", [[layer_2_dicts, attn_dicts]]),
        ("l3_attn", [[layer_3_dicts, attn_dicts]]),
        ("l4_attn", [[layer_4_dicts, attn_dicts]]),
        ("l5_attn", [[layer_5_dicts, attn_dicts]]),
        ("l0_mlp_out", [[layer_0_dicts, mlp_out_dicts]]),
        ("l2_mlp_out", [[layer_2_dicts, mlp_out_dicts]]),
    ]

    for graph_name, categories in experiments:
        learned_dicts_nested: List[List[Tuple[str, List[Tuple[LearnedDict, Dict[Any, Any]]]]]] = []
        for subcategory in categories:
            learned_dict_loc_list = list(set.intersection(*[set(x) for x in subcategory]))
            learned_dict_loc_list.sort(key=lambda x: int(x.split("_")[-1][1:])) # sort by ratio
            learned_dict_lists = [(x.split("sweep_")[-1], torch.load(os.path.join(x, "_9", "learned_dicts.pt"))) for x in learned_dict_loc_list]
            learned_dicts_nested.append(learned_dict_lists)

        print(f"Found {sum(len(x) for x in learned_dicts_nested)} lists of dicts for experiment {graph_name}")

        dataset = torch.load(f"/mnt/ssd-cluster/single_chunks/{graph_name}/0.pt")
        sample_idxs = np.random.choice(len(dataset), 5000, replace=False)

        device = torch.device("cuda:0")

        sample = dataset[sample_idxs].to(dtype=torch.float32, device=device)

        all_data: List[List[Tuple[str, List[Tuple[float, float, float]]]]] = []
        for learned_dict_list in learned_dicts_nested:
            datapoint_series: List[Tuple[str, List[Tuple[float, float, float]]]] = []
            for run_name, learned_dict_set in learned_dict_list:
                datapoints: List[Tuple[float, float, float]] = []
                for learned_dict, hyperparams in learned_dict_set:
                    learned_dict.to_device(device)
                    r_sq = standard_metrics.fraction_variance_unexplained(learned_dict, sample).item()
                    sparsity = standard_metrics.mean_nonzero_activations(learned_dict, sample).sum().item()
                    datapoints.append((r_sq, sparsity, hyperparams["l1_alpha"]))
                datapoint_series.append((run_name, datapoints))
            all_data.append(datapoint_series)

        colors = ["Purples", "Blues", "Greens", "Oranges", "Reds", "Greys", "YlOrBr", "YlOrRd", "OrRd"]
        markers = ["o", "v", "s", "P", "X"]
        #labels = ["0.5", "1", "2", "4", "8"]
        #labels = [str(r) for r in learned_dict_files]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for k, datapoint_lists in enumerate(all_data):
            for i, (run_name, datapoints) in enumerate(datapoint_lists):
                r_sq, sparsity, l1_alpha = zip(*datapoints)
                ax.scatter(sparsity, r_sq, c=[math.log10(l1) for l1 in l1_alpha], label=run_name, cmap=colors[i % len(colors)], vmin=-5, vmax=-2, marker=markers[k % len(markers)])
                if i == len(datapoints) - 1:
                    # write the l1_alpha values on every 5th point and highlight them
                    for j, (x, y) in enumerate(zip(sparsity, r_sq)):
                        if j % 5 == 0:
                            ax.annotate(f"{l1_alpha[j]:.1}", (x, y))
                            ax.scatter([x], [y], c="black")

        # cap the x axis at 512, but allow smaller
        l, r = ax.get_xlim()
        ax.set_xlim(0, min(r, 512))
        ax.set_ylim(0, 1)
        
        ax.set_xlabel("Mean no. features active")
        ax.set_ylabel("Unexplained Variance")
        ax.legend()
        plt.savefig(f"freq_plot_compare_{graph_name}.png")
        print(f"Saved plot for {graph_name}")
