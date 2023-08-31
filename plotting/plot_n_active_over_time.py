import multiprocessing as mp
import os
import pickle
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch

LOCAL_DIR = os.path.join(os.path.dirname(__file__), "..")
LOAD_DIR = "/mnt/ssd-cluster/bigrun0308"
PLOT_DATA_DIR = "/mnt/ssd-cluster/plot_data"
PLOTS_DIR = "/mnt/ssd-cluster/plots"

sys.path.append(LOCAL_DIR)

from standard_metrics import calc_feature_n_active


def ratio_map(ratio: int) -> float:
    # maps ratio as a str/int to the actual ratio
    if ratio > 0:
        return float(ratio)
    else:
        return 0.5


activation_dim_map = {"mlp": 2048, "residual": 512}


def run_for_layer(args) -> None:
    layer, layer_loc, device, reload, tied = args

    if reload or not os.path.exists(os.path.join(PLOT_DATA_DIR, f"n_active_{tied}_l{layer}_{layer_loc}_long.pkl")):
        # check that has root permission
        if os.geteuid() != 0:
            raise PermissionError("Must run as root to load the data")
        plt.clf()
        chunk_loc = f"/mnt/ssd-cluster/single_chunks/l{layer}_{layer_loc}/0.pt"
        activations = torch.load(chunk_loc).to(torch.float32).to(device)
        layer_data: List[Tuple[int, List[Tuple[float, float]]]] = []

        epochs = [0, 10, 20, 30, 40, 50, 59]
        ratio = "1.0"
        for epoch in epochs:
            dicts_loc = f"{tied}_{layer_loc}_l{layer}_r{ratio}_long"
            # for ratio in tied_ratios:
            #     dicts_loc = f"tied_{layer_loc}_l{layer}_r{ratio}"
            print(dicts_loc)
            if not os.path.exists(os.path.join(LOAD_DIR, dicts_loc, f"_{epoch}", "learned_dicts.pt")):
                continue
            all_dicts = torch.load(os.path.join(LOAD_DIR, dicts_loc, f"_{epoch}", "learned_dicts.pt"))
            dead_feats_data_series = []
            for learned_dict, hparams in all_dicts:
                batch_size = int(5e4 // (float(ratio) + 1))
                learned_dict.to_device(device)
                n_active_count = torch.zeros(learned_dict.n_feats, device=device)
                for i in range(0, len(activations), batch_size):
                    batch = activations[i : i + batch_size]
                    feat_activations = learned_dict.encode(batch)
                    n_active_count += calc_feature_n_active(feat_activations)

                n_active_total = (n_active_count > 10).sum().item()
                l1a = hparams["l1_alpha"]
                if l1a == 0:
                    l1a = 8e-5
                print(
                    layer,
                    layer_loc,
                    ratio,
                    hparams["l1_alpha"],
                    n_active_total,
                    n_active_total / learned_dict.n_feats,
                )
                dead_feats_data_series.append((l1a, n_active_total / learned_dict.n_feats))

            layer_data.append((epoch, dead_feats_data_series))
            print(f"Finished ratio {epoch} {layer} {layer_loc}")

        # save the data
        os.makedirs(PLOT_DATA_DIR, exist_ok=True)
        pickle.dump(
            layer_data,
            open(
                os.path.join(PLOT_DATA_DIR, f"n_active_untied_l{layer}_{layer_loc}.pkl"),
                "wb",
            ),
        )
        print(f"Finished layer {layer} {layer_loc}")

    else:
        layer_data = pickle.load(
            open(
                os.path.join(PLOT_DATA_DIR, f"n_active_untied_l{layer}_{layer_loc}_overtime.pkl"),
                "rb",
            )
        )
        print(f"Loaded layer {layer} {layer_loc}")

    plt.clf()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=plt.figaspect(1 / 3))
    ax.set_xscale("log")
    ax.set_xlabel("L1 Alpha")
    ax.set_ylabel("Fraction of features alive (>10 non-zero)")
    for epoch, ratio_data in layer_data:
        ax.plot(*zip(*ratio_data), label=epoch)
    ax.legend()
    ax.set_title(f"Plot of % active features for {layer_loc} layer {layer}")

    ax2.set_xscale("log")
    ax2.set_xlabel("L1 Alpha")
    ax2.set_ylabel("Number of features alive (>10 non-zero)")
    for epoch, epoch_data in layer_data:
        abs_data = [(x[0], activation_dim_map[layer_loc] * epoch * x[1]) for x in epoch_data]
        ax2.plot(*zip(*abs_data), label=epoch)
    ax2.legend()
    ax2.set_title(f"Plot of total active features for {layer_loc} layer {layer}")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f"active_plot_{tied}_l{layer}_{layer_loc}_overtime{ratio}.png"))
    print(f"Saved layer {layer} {layer_loc} and plotted")


if __name__ == "__main__":
    devices = ["cuda:7", "cuda:1", "cuda:2", "cuda:6", "cuda:4", "cuda:5"]
    layer_loc = "mlp"
    reload = False
    tied = "untied"
    long_tied_ratios = ["0.25", "0.5", "1.0", "2.0", "4.0", "8.0", "16"]
    # untied_ratios = ['0.25', '0.5', '1.0', '2.0', '4.0', '8.0']
    untied_ratios = [2,4,8,16,32,64,128,256]
    # layers = list(range(6))
    layers = [0, 1, 2, 3, 4, 5]
    with mp.Pool(6) as pool:
        pool.map(
            run_for_layer,
            [(layer, layer_loc, devices[i], reload, tied) for i, layer in enumerate(layers)],
        )

    # layer_loc = "residual"
    # with mp.Pool(6) as pool:
    #     pool.map(run_for_layer, [(layer, layer_loc, devices[i], reload) for i, layer in enumerate(layers)])
