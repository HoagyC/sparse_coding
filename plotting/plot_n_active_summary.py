import sys
import pickle
import os
from typing import List, Tuple
import multiprocessing as mp

import matplotlib.pyplot as plt
import torch

LOCAL_DIR = os.path.join(os.path.dirname(__file__), "..")
LOAD_DIR = "/mnt/ssd-cluster/bigrun0308"
PLOT_DATA_DIR = "/mnt/ssd-cluster/plot_data"
PLOTS_DIR = "/mnt/ssd-cluster/plots"

sys.path.append(LOCAL_DIR)

from standard_metrics import calc_feature_n_active

def plot_all():
    all_data = {}
    layer_locs = ["residual", "mlp"]
    tied_strs = ["", "untied_"]
    layers = list(range(6))
    for layer_loc in layer_locs:
        for layer in layers:
            for tied_str in tied_strs:
                run_name = f"{tied_str}l{layer}_{layer_loc}"
                if not os.path.exists(os.path.join(PLOT_DATA_DIR, f"n_active_{run_name}.pkl")):
                    continue
            
                with open(os.path.join(PLOT_DATA_DIR, f"n_active_{run_name}.pkl"), "rb") as f:
                    all_data[run_name] = pickle.load(f)
    
    # now we want to plot the data on one figure with a subplot for each layer
    fig, axs = plt.subplots(4, 6, figsize=(15, 10))

    for layer in layers:
        for layer_loc in layer_locs:
            for tied_str in tied_strs:
                row = 0 if tied_str == "" else 2
                row += 1 if layer_loc == "mlp" else 0
                col = layer
                ax = axs[row, col]
                try:
                    layer_data = all_data[f"{tied_str}l{layer}_{layer_loc}"]
                except KeyError:
                    continue
                for epoch, ratio_data in layer_data:
                    ax.plot(*zip(*ratio_data), label=epoch)
                ax.legend()
                ax.set_xlabel("L1 Alpha")
                ax.set_xscale("log")

                # ax2.set_xscale("log")
                # ax2.set_xlabel("L1 Alpha")
                # ax2.set_ylabel("Number of features alive (>10 non-zero)")
                # for epoch, epoch_data in layer_data:
                #     abs_data = [(x[0], activation_dim_map[layer_loc] * epoch * x[1]) for x in epoch_data]
                #     ax2.plot(*zip(*abs_data), label=epoch)
                # ax2.legend()
                # ax2.set_title(f"Plot of total active features for {layer_loc} layer {layer}")
    
    # set a single ylabel for each row
    for row in range(4):
        axs[row, 0].set_ylabel("No. active features")
        # add text to the left of the graph to indicate which layer type is being plotted and whether tied or untied
        if row == 0:
            axs[row, 0].text(-0.5, 0.5, "Residual, Tied", transform=axs[row, 0].transAxes, va="center", rotation=90)
        elif row == 1:
            axs[row, 0].text(-0.5, 0.5, "MLP, Tied", transform=axs[row, 0].transAxes, va="center", rotation=90)
        elif row == 2:
            axs[row, 0].text(-0.5, 0.5, "Residual, Untied", transform=axs[row, 0].transAxes, va="center", rotation=90)
        elif row == 3:
            axs[row, 0].text(-0.5, 0.5, "MLP, Untied", transform=axs[row, 0].transAxes, va="center", rotation=90)

        

    for col in range(6):
        axs[0, col].set_title(f"Layer {col}")
    fig.suptitle("Number of active features for each layer")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "n_active_summary.png"))



if __name__ == "__main__":
    plot_all()