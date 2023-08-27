import itertools
import math
import os
import pickle
import shutil
import sys
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import standard_metrics
from autoencoders.learned_dict import LearnedDict
from autoencoders.pca import BatchedPCA, PCAEncoder

load_dir = "/mnt/ssd-cluster/bigrun0308"
plot_data_dir = "/mnt/ssd-cluster/plot_data"
plot_dir = "/mnt/ssd-cluster/plots"


def plot_by_group() -> None:
    chunk_range = [59]
    learned_dict_files = [os.path.join(load_dir, x) for x in os.listdir(load_dir)]
    # learned_dict_files += [f for f in os.listdir(".") if f.startswith("output_attn")]
    # learned_dict_files += [f for f in os.listdir(".") if f.startswith("output_sweep")]
    # learned_dict_files = [f for f in os.listdir(".") if f.startswith("lr1e-3")]

    resid_dicts = [f for f in learned_dict_files if "resid" in f]
    mlp_dicts = [f for f in learned_dict_files if "mlp" in f]
    mlp_dicts = [f for f in mlp_dicts if "mlpout" not in f]
    attn_dicts = [f for f in learned_dict_files if "attn" in f]
    mlp_out_dicts = [f for f in learned_dict_files if "mlpout" in f]

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

    tied_dicts = [f for f in learned_dict_files if "/tied" in f]
    untied_dicts = [f for f in learned_dict_files if "untied" in f]
    long_dicts = [f for f in learned_dict_files if "long" in f]

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
        ("l0_mlp_untied", [[layer_0_dicts, mlp_dicts, long_dicts, untied_dicts]]),
        ("l1_mlp_untied", [[layer_1_dicts, mlp_dicts, long_dicts, untied_dicts]]),
        ("l2_mlp_untied", [[layer_2_dicts, mlp_dicts, long_dicts, untied_dicts]]),
        ("l3_mlp_untied", [[layer_3_dicts, mlp_dicts, long_dicts, untied_dicts]]),
        ("l4_mlp_untied", [[layer_4_dicts, mlp_dicts, long_dicts, untied_dicts]]),
        ("l5_mlp_untied", [[layer_5_dicts, mlp_dicts, long_dicts, untied_dicts]]),
        ("l0_mlp_tied", [[layer_0_dicts, mlp_dicts, long_dicts, tied_dicts]]),
        ("l1_mlp_tied", [[layer_1_dicts, mlp_dicts, long_dicts, tied_dicts]]),
        ("l2_mlp_tied", [[layer_2_dicts, mlp_dicts, long_dicts, tied_dicts]]),
        ("l3_mlp_tied", [[layer_3_dicts, mlp_dicts, long_dicts, tied_dicts]]),
        ("l4_mlp_tied", [[layer_4_dicts, mlp_dicts, long_dicts, tied_dicts]]),
        ("l5_mlp_tied", [[layer_5_dicts, mlp_dicts, long_dicts, tied_dicts]]),
    ]

    for graph_name, categories in experiments:
        learned_dicts_nested: List[List[Tuple[str, List[Tuple[LearnedDict, Dict[Any, Any]]]]]] = []
        for subcategory in categories:
            learned_dict_loc_list = list(set.intersection(*[set(x) for x in subcategory]))
            print(learned_dict_loc_list)
            learned_dict_loc_list.sort(key=lambda x: int(float((x.split("_")[-2][1:]))))  # sort by ratio
            learned_dict_lists = []
            for x in learned_dict_loc_list:
                name = x.split("_")[-2][1:]
                try:
                    learned_dicts = torch.load(os.path.join(x, "_59", "learned_dicts.pt"))
                except:
                    print(f"Could not load learned dicts for {x}")
                    continue
                learned_dict_lists.append((name, learned_dicts))
            learned_dicts_nested.append(learned_dict_lists)

        chunk_name = "_".join(graph_name.split("_")[:2])
        print(f"Found {sum(len(x) for x in learned_dicts_nested)} lists of dicts for experiment {graph_name}")

        dataset = torch.load(f"/mnt/ssd-cluster/single_chunks/{chunk_name}/0.pt")
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

        pickle.dump(plot_dir, open(f"all_data_{graph_name}.pkl", "wb"))

        colors = [
            "Purples",
            "Blues",
            "Greens",
            "Oranges",
            "Reds",
            "Greys",
            "YlOrBr",
            "YlOrRd",
            "OrRd",
        ]
        markers = ["o", "v", "s", "P", "X"]
        # labels = ["0.5", "1", "2", "4", "8"]
        # labels = [str(r) for r in learned_dict_files]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for k, datapoint_lists in enumerate(all_data):
            for i, (run_name, datapoints) in enumerate(datapoint_lists):
                r_sq, sparsity, l1_alpha = zip(*datapoints)
                cs = [math.log10(l1) if l1 != 0 else -5 for l1 in l1_alpha]
                ax.scatter(
                    sparsity,
                    r_sq,
                    label=run_name,
                    cmap=colors[i % len(colors)],
                    vmin=-5,
                    vmax=-2,
                    marker=markers[k % len(markers)],
                )
                # if i == len(datapoints) - 1:
                #     # write the l1_alpha values on every 5th point and highlight them
                #     for j, (x, y) in enumerate(zip(sparsity, r_sq)):
                #         if j % 5 == 0:
                #             ax.annotate(f"{l1_alpha[j]:.1}", (x, y))
                #             ax.scatter([x], [y], c="black")

        # cap the x axis at 512, but allow smaller
        l, r = ax.get_xlim()
        ax.set_xlim(0, min(r, 512))
        ax.set_ylim(0, 1)

        ax.set_xlabel("Mean no. features active")
        ax.set_ylabel("Unexplained Variance")
        ax.legend()
        # set legend opacity to 1
        leg = ax.get_legend()
        leg.legendHandles[0].set_alpha(1)
        ax.set_title(f"Sparsity vs. Unexplained Variance for {graph_name}")
        plt.savefig(os.path.join(plot_dir, f"freq_plot_compare_{graph_name}_long.png"))
        print(f"Saved plot for {graph_name}")


def plot_fuv_sparsity():
    learned_dict_files = [
        # "output_4_rd_deep/_0/learned_dicts.pt",
        # "output_4_rd_deep/_1/learned_dicts.pt",
        # "output_4_rd_deep/_2/learned_dicts.pt",
        # "output_4_rd_deep/_3/learned_dicts.pt",
        # "output_4_rd_deep/_4/learned_dicts.pt",
        # "output_4_rd_deep/_5/learned_dicts.pt",
        # "output_4_rd_deep/_6/learned_dicts.pt",
        # "output_4_lista/_7/learned_dicts.pt",
        # "output_4_lista_deep/_7/learned_dicts.pt",
        # "output_4_lista_neg/_7/learned_dicts.pt",
        # "output_4_rd_deep/_7/learned_dicts.pt",
        # "output_4_tied/_7/learned_dicts.pt",
        # "/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r1/_9/learned_dicts.pt",
        # "/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r2/_9/learned_dicts.pt",
        # "/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r4/_9/learned_dicts.pt",
        # "/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r8/_9/learned_dicts.pt",
        # "/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r16/_9/learned_dicts.pt",
        # "/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r32/_9/learned_dicts.pt",
        # "output_topk/_27/learned_dicts.pt",
        # "/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r0/_9/learned_dicts.pt",
        # "/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r2/_9/learned_dicts.pt",
        # "/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r8/_9/learned_dicts.pt",
        # "/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r16/_9/learned_dicts.pt",
        # "output_topk_synthetic/_49/learned_dicts.pt",
        # "output_tied_synthetic/_49/learned_dicts.pt",
        "lr1e-3__mlp_l1_r1/_110/learned_dicts.pt"
    ]

    ground_truth_file = "output_topk_synthetic/generator.pt"

    ground_truth = torch.load(ground_truth_file).sparse_component_dict.to("cuda:7")

    file_labels = [
        "TopK",
        "Tied Linear",
    ]

    learned_dict_sets = {}

    for label, learned_dict_file in zip(file_labels, learned_dict_files):
        learned_dicts = torch.load(learned_dict_file)
        dict_sizes = list(set([hyperparams["dict_size"] for _, hyperparams in learned_dicts]))
        for learned_dict, hyperparams in learned_dicts:
            name = label + " " + str(hyperparams["dict_size"])

            if name not in learned_dict_sets:
                learned_dict_sets[name] = []
            learned_dict_sets[name].append((learned_dict, hyperparams))

    device = torch.device("cuda:7")

    dataset = torch.load("activation_data_synthetic/0.pt").to(dtype=torch.float32, device=device)

    pca = BatchedPCA(dataset.shape[1], device)

    """
    datapoint = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.float32)
    pca_test = PCAEncoder(torch.eye(datapoint.shape[1]), 2)
    print(pca_test.encode(datapoint))
    code = pca_test.encode(datapoint)
    print(pca_test.predict(code))
    """

    print("Training PCA")

    batch_size = 5000
    for i in tqdm.tqdm(range(0, len(dataset), batch_size)):
        j = min(i + batch_size, len(dataset))
        batch = dataset[i:j]
        pca.train_batch(batch)

    sample_idxs = np.random.choice(len(dataset), 50000, replace=False)
    sample = dataset[sample_idxs]

    del dataset

    print("Scoring PCA")

    pca_scores = []
    for sparsity in tqdm.tqdm(range(1, sample.shape[1] // 2, 4)):
        pca_dict = pca.to_learned_dict(sparsity)
        # pca_dict = PCAEncoder(torch.eye(sample.shape[1], device=device), sparsity)
        fvu = standard_metrics.fraction_variance_unexplained(pca_dict, sample).item()
        pca_scores.append((sparsity, fvu))

    # means = []
    # variances = []
    # datapoints = []
    scores = {}
    for label, learned_dict_set in learned_dict_sets.items():
        # points = []
        scores[label] = []
        for learned_dict, hyperparams in learned_dict_set:
            learned_dict.to_device(device)
            fvu = standard_metrics.fraction_variance_unexplained(learned_dict, sample).item()
            sparsity = standard_metrics.mean_nonzero_activations(learned_dict, sample).sum().item()
            mcs = standard_metrics.mmcs_to_fixed(learned_dict, ground_truth).item()
            scores[label].append((sparsity, fvu, mcs))

    colors = ["Purples", "Blues", "Greens", "Oranges"]
    markers = ["o", "x", "s", "P", "v"]
    styles = ["dotted", "dashdot", "solid", "dashdot"]
    settings = itertools.product(styles, colors)

    os.makedirs("graphs", exist_ok=True)
    shutil.rmtree("graphs")
    os.makedirs("graphs", exist_ok=True)

    """
    l1_vals = sorted(list(set([l1 for _, l1, _ in means])))
    dict_sizes = sorted(list(set([d for _, _, d in means])))

    mean_img = np.zeros((len(l1_vals), len(dict_sizes)))
    var_img = np.zeros((len(l1_vals), len(dict_sizes)))

    for mean, l1, dict_size in means:
        mean_img[l1_vals.index(l1), dict_sizes.index(dict_size)] = mean
    
    for var, l1, dict_size in variances:
        var_img[l1_vals.index(l1), dict_sizes.index(dict_size)] = var
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    img = ax.imshow(mean_img.T, cmap="plasma")
    ax.set_yticks(range(len(dict_sizes)))
    ax.set_yticklabels(dict_sizes)
    ax.set_xticks(range(len(l1_vals)))
    ax.set_xticklabels([f"{l1_val:.2e}" for l1_val in l1_vals], rotation=90)
    ax.set_ylabel("Dictionary Size")
    ax.set_xlabel("l1_alpha")
    ax.set_title("Mean no. activations")
    fig.colorbar(img)
    plt.savefig("graphs/mean_activations.png")
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    img = ax.imshow(var_img.T, cmap="plasma")
    ax.set_yticks(range(len(dict_sizes)))
    ax.set_yticklabels(dict_sizes)
    ax.set_xticks(range(len(l1_vals)))
    ax.set_xticklabels([f"{l1_val:.2e}" for l1_val in l1_vals], rotation=90)
    ax.set_ylabel("Dictionary Size")
    ax.set_xlabel("l1_alpha")
    ax.set_title("Variance no. activations")
    fig.colorbar(img)
    plt.savefig("graphs/var_activations.png")
    plt.close(fig)

    for sparsity, clamped_sparsity, l1_alpha, dict_size in tqdm.tqdm(datapoints):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(sparsity, bins=50)
        ax.set_yscale("log")
        ax.set_xlabel("Mean no. activations")
        ax.set_ylabel("Frequency")
        ax.set_title(f"l1_alpha={l1_alpha:.2e}, dict_size={dict_size}")
        plt.savefig(f"graphs/log_freq_plot_{l1_alpha:.2e}_{dict_size}.png")
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(clamped_sparsity, bins=50)
        ax.set_yscale("log")
        ax.set_xlabel("Mean no. activations")
        ax.set_ylabel("Frequency")
        ax.set_title(f"l1_alpha={l1_alpha:.2e}, dict_size={dict_size}")
        plt.savefig(f"graphs/log_freq_plot_clamped_{l1_alpha:.2e}_{dict_size}.png")
        plt.close(fig)
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    legend_lines = []
    legend_names = []
    for (style, color), (label, series) in zip(settings, scores.items()):
        # cmap = matplotlib.cm.get_cmap(color)
        # norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        # colors_ = [cmap(norm(mcs)) for _, _, mcs in series]
        sorted_series = sorted(series, key=lambda x: x[0])
        sparsity, fvu, mcs = zip(*sorted_series)

        points = np.array([fvu, mcs]).T.reshape(-1, 1, 2)
        c = np.ones_like(np.array(mcs))
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        cs = 0.5 * (c[:-1] + c[1:])
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        cmap = matplotlib.cm.get_cmap(color)
        lc = matplotlib.collections.LineCollection(segments, cmap=cmap, norm=norm, linestyle=style)
        lc.set_array(cs)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)

        legend_lines.append(Line2D([0], [0], color=cmap(0.5), linestyle=style, linewidth=2))
        legend_names.append(label)

        # ax.plot(sparsity, fvu, label=label, color=color, linestyle=style)

    # pca_xs = [s for s, _ in pca_scores]
    # pca_ys = [s for _, s in pca_scores]
    # ax.plot(pca_xs, pca_ys, color="red", linestyle="dashed")

    # legend_lines.append(Line2D([0], [0], color="red", linestyle="dashed"))
    # legend_names.append("PCA")

    ax.set_xlabel("Fraction Variance Unexplained")
    ax.set_ylabel("MCS")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend(legend_lines, legend_names)

    # ax.set_yscale("log")

    # fix legend colors
    # legend = ax.get_legend()
    # for i, l in enumerate(legend.legend_handles[:-1]):
    #    l.set_color(matplotlib.colormaps.get_cmap(colors[i % len(colors)])(0.8))

    plt.savefig("mcs_fvu.png")

    ax.set_yscale("log")

    plt.savefig("mcs_fvu_log.png")


if __name__ == "__main__":
    plot_by_group()
