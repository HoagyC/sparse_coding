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
    layer, layer_loc, ratios, device, reload, tied = args
    n_chunks = 59 if layer_loc == "mlp" else 9

    if reload or not os.path.exists(os.path.join(PLOT_DATA_DIR, f"n_active_ratio_{tied}_l{layer}_{layer_loc}.pkl")):
        # check that has root permission
        if os.geteuid() != 0:
            raise PermissionError("Must run as root to load the data")
        plt.clf()
        chunk_loc = f"/mnt/ssd-cluster/single_chunks/l{layer}_{layer_loc}/0.pt"
        activations = torch.load(chunk_loc).to(torch.float32).to(device)
        layer_data: List[Tuple[int, List[Tuple[float, float]]]] = []
        for ratio in ratios:
            dicts_loc = f"{tied}_{layer_loc}_l{layer}_r{ratio}"
            print(dicts_loc)
            if not os.path.exists(os.path.join(LOAD_DIR, dicts_loc, f"_{n_chunks}", "learned_dicts.pt")):
                print(f"Skipping {dicts_loc}")
                continue
            all_dicts = torch.load(os.path.join(LOAD_DIR, dicts_loc, f"_{n_chunks}", "learned_dicts.pt"))
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
                print(
                    layer,
                    layer_loc,
                    ratio,
                    hparams["l1_alpha"],
                    n_active_total,
                    n_active_total / learned_dict.n_feats,
                )
                dead_feats_data_series.append((hparams["l1_alpha"], n_active_total / learned_dict.n_feats))

            layer_data.append((ratio, dead_feats_data_series))
            print(f"Finished ratio {ratio} {layer} {layer_loc}")

        # save the data
        os.makedirs(PLOT_DATA_DIR, exist_ok=True)
        pickle.dump(
            layer_data,
            open(
                os.path.join(PLOT_DATA_DIR, f"n_active_ratio_{tied}_l{layer}_{layer_loc}.pkl"),
                "wb",
            ),
        )
        print(f"Finished layer {layer} {layer_loc}")

    else:
        layer_data = pickle.load(
            open(
                os.path.join(PLOT_DATA_DIR, f"n_active_ratio_{tied}_l{layer}_{layer_loc}.pkl"),
                "rb",
            )
        )
        print(f"Loaded layer {layer} {layer_loc}")

    plt.clf()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=plt.figaspect(1 / 3))
    ax.set_xscale("log")
    ax.set_xlabel("L1 Alpha")
    ax.set_ylabel("Fraction of features alive (>10 non-zero)")
    for ratio, ratio_data in layer_data:
        ax.plot(*zip(*ratio_data), label=ratio)
    ax.legend()
    ax.set_title(f"Plot of % active features for {layer_loc} layer {layer}")

    ax2.set_xscale("log")
    ax2.set_xlabel("L1 Alpha")
    ax2.set_ylabel("Number of features alive (>10 non-zero)")
    for ratio, ratio_data in layer_data:
        float_ratio = float(ratio)
        abs_data = [(x[0], activation_dim_map[layer_loc] * float_ratio * x[1]) for x in ratio_data]
        ax2.plot(*zip(*abs_data), label=ratio)
    ax2.legend()
    ax2.set_title(f"Plot of total active features for {layer_loc} layer {layer}")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f"active_plot_ratio_{tied}_l{layer}_{layer_loc}.png"))
    print(f"Saved layer {layer} {layer_loc} and plotted")


if __name__ == "__main__":
    devices = ["cuda:7", "cuda:1", "cuda:4", "cuda:6", "cuda:0", "cuda:5"]
    reload = True
    residual_ratios = ["0", "1", "2", "4", "8", "16", "32", "64", "128", "256"]
    mlp_ratios = ["0.5", "1.0", "2.0", "4.0", "8.0"]
    layers = list(range(6))

    for layer_loc in ["residual"]:
        ratios = mlp_ratios if layer_loc == "mlp" else residual_ratios
        for tied in ["tied"]:
            with mp.Pool(6) as pool:
                pool.map(
                    run_for_layer,
                    [(layer, layer_loc, ratios, devices[i], reload, tied) for i, layer in enumerate(layers)],
                )
