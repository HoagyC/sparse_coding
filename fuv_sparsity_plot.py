import standard_metrics
import matplotlib
import torch

from autoencoders.pca import BatchedPCA, PCAEncoder

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.lines import Line2D
import matplotlib
import math
import shutil
import os
import tqdm

import itertools

if __name__ == "__main__":
    learned_dict_files = [
        #"output_4_rd_deep/_0/learned_dicts.pt",
        #"output_4_rd_deep/_1/learned_dicts.pt",
        #"output_4_rd_deep/_2/learned_dicts.pt",
        #"output_4_rd_deep/_3/learned_dicts.pt",
        #"output_4_rd_deep/_4/learned_dicts.pt",
        #"output_4_rd_deep/_5/learned_dicts.pt",
        #"output_4_rd_deep/_6/learned_dicts.pt",
        #"output_4_lista/_7/learned_dicts.pt",
        # "output_4_lista_deep/_7/learned_dicts.pt",
        #"output_4_lista_neg/_7/learned_dicts.pt",
        #"output_4_rd_deep/_7/learned_dicts.pt",
        #"output_4_tied/_7/learned_dicts.pt",
        #"/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r1/_9/learned_dicts.pt",
        #"/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r2/_9/learned_dicts.pt",
        #"/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r4/_9/learned_dicts.pt",
        #"/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r8/_9/learned_dicts.pt",
        #"/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r16/_9/learned_dicts.pt",
        #"/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r32/_9/learned_dicts.pt",
        #"output_topk/_27/learned_dicts.pt",
        #"/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r0/_9/learned_dicts.pt",
        #"/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r2/_9/learned_dicts.pt",
        #"/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r8/_9/learned_dicts.pt",
        #"/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r16/_9/learned_dicts.pt",
        "output_topk_synthetic/_49/learned_dicts.pt",
        "output_tied_synthetic/_49/learned_dicts.pt",
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
        #pca_dict = PCAEncoder(torch.eye(sample.shape[1], device=device), sparsity)
        fvu = standard_metrics.fraction_variance_unexplained(pca_dict, sample).item()
        pca_scores.append((sparsity, fvu))

    #means = []
    #variances = []
    #datapoints = []
    scores = {}
    for label, learned_dict_set in learned_dict_sets.items():
        #points = []
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
        #cmap = matplotlib.cm.get_cmap(color)
        #norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        #colors_ = [cmap(norm(mcs)) for _, _, mcs in series]
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

        #ax.plot(sparsity, fvu, label=label, color=color, linestyle=style)

    #pca_xs = [s for s, _ in pca_scores]
    #pca_ys = [s for _, s in pca_scores]
    #ax.plot(pca_xs, pca_ys, color="red", linestyle="dashed")

    #legend_lines.append(Line2D([0], [0], color="red", linestyle="dashed"))
    #legend_names.append("PCA")

    ax.set_xlabel("Fraction Variance Unexplained")
    ax.set_ylabel("MCS")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend(legend_lines, legend_names)

    #ax.set_yscale("log")

    # fix legend colors
    #legend = ax.get_legend()
    #for i, l in enumerate(legend.legend_handles[:-1]):
    #    l.set_color(matplotlib.colormaps.get_cmap(colors[i % len(colors)])(0.8))

    plt.savefig("mcs_fvu.png")

    ax.set_yscale("log")

    plt.savefig("mcs_fvu_log.png")