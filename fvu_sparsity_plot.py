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

def score_dict(score, label, hyperparams, learned_dict, dataset, ground_truth=None):
    if score == "mcs":
        return standard_metrics.mmcs_to_fixed(learned_dict, ground_truth).item()
    elif score == "fvu":
        return standard_metrics.fraction_variance_unexplained(learned_dict, dataset).item()
    elif score == "sparsity":
        return standard_metrics.mean_nonzero_activations(learned_dict, dataset).sum().item()
    elif score == "l1":
        return hyperparams["l1_alpha"]
    elif score == "neg_log_l1":
        return -np.log(hyperparams["l1_alpha"])

def generate_scores(learned_dict_files, dataset_file=None, generator_file=None, x_score="sparsity", y_score="fvu", c_score="mcs", group_by="dict_size", other_dicts=[], device="cuda:7"):
    if generator_file is not None:
        generator = torch.load(generator_file)
        ground_truth = generator.sparse_component_dict.to(device)
    else:
        generator, ground_truth = None, None

    if dataset_file is None and generator is not None:
        dataset = torch.cat([next(generator) for _ in tqdm.tqdm(range(512))]).to(dtype=torch.float32, device=device)
    else:
        dataset = torch.load(dataset_file).to(dtype=torch.float32, device=device)

    learned_dict_sets = {}

    for label, learned_dict_file in learned_dict_files:
        learned_dicts = torch.load(learned_dict_file)
        #groups = list(set([hyperparams[group_by] for _, hyperparams in learned_dicts]))
        for learned_dict, hyperparams in learned_dicts:
            name = label + " " + str(hyperparams[group_by])

            if name not in learned_dict_sets:
                learned_dict_sets[name] = []
            learned_dict_sets[name].append((learned_dict, hyperparams))

    score_pca_topk = "pca_topk" in other_dicts
    score_pca_rot = "pca_rot" in other_dicts
    score_pca = score_pca_topk or score_pca_rot

    if score_pca:
        pca = BatchedPCA(dataset.shape[1], device)

        print("Training PCA")

        batch_size = 5000
        for i in tqdm.tqdm(range(0, len(dataset), batch_size)):
            j = min(i + batch_size, len(dataset))
            batch = dataset[i:j]
            pca.train_batch(batch)

        if score_pca_topk:
            learned_dict_sets["PCA (TopK)"] = [(pca.to_topk_dict(k), {"dict_size": 512, "k": k}) for k in range(1, dataset.shape[1] // 2, 8)]
        if score_pca_rot:
            learned_dict_sets["PCA (Static)"] = [(pca.to_rotation_dict(n), {"dict_size": 512, "n": n}) for n in range(1, dataset.shape[1], 8)]

    sample_idxs = np.random.choice(len(dataset), 50000, replace=False)
    sample = dataset[sample_idxs]

    del dataset

    scores = {}
    for label, learned_dict_set in learned_dict_sets.items():
        #points = []
        scores[label] = []
        for learned_dict, hyperparams in learned_dict_set:
            learned_dict.to_device(device)

            x = score_dict(x_score, label, hyperparams, learned_dict, sample, ground_truth)
            y = score_dict(y_score, label, hyperparams, learned_dict, sample, ground_truth)

            if c_score is not None:
                c = score_dict(c_score, label, hyperparams, learned_dict, sample, ground_truth)
            else:
                c = 0.5

            scores[label].append((x, y, c))

    return scores

def scores_derivative(scores):
    scores_ = {}
    for label in scores.keys():
        sorted_series = sorted(scores[label], key=lambda x: x[0])
        sorted_series = [sorted_series[0]] + [sorted_series[i] for i in range(1, len(sorted_series)) if sorted_series[i][0] != sorted_series[i - 1][0]]
        x, y, shade = zip(*sorted_series)

        dydx = np.gradient(y, x)
        x_ = (np.array(x)[:-1] + np.array(x)[1:]) / 2
        c_ = (np.array(shade)[:-1] + np.array(shade)[1:]) / 2

        scores_[label] = list(zip(x_, dydx, c_))

    return scores_

def scores_logx(scores):
    scores_ = {}
    for label in scores.keys():
        sorted_series = sorted(scores[label], key=lambda x: x[0])
        x, y, shade = zip(*sorted_series)

        x_ = np.log(np.array(x))
        c_ = np.array(shade)

        scores_[label] = list(zip(x_, y, c_))

    return scores_

def scores_logy(scores):
    scores_ = {}
    for label in scores.keys():
        sorted_series = sorted(scores[label], key=lambda x: x[0])
        x, y, shade = zip(*sorted_series)

        y_ = np.log(np.array(y))
        c_ = np.array(shade)

        scores_[label] = list(zip(x, y_, c_))

    return scores_

def plot_scores(scores, settings, xlabel, ylabel, xrange, yrange, title, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    legend_lines = []
    legend_names = []
    for label in scores.keys():
        cmap = matplotlib.cm.get_cmap(settings[label]["color"])
        style = settings[label]["style"]
        sorted_series = sorted(scores[label], key=lambda x: x[0])
        x, y, shade = zip(*sorted_series)

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        c = np.array(shade)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        cs = 0.5 * (c[:-1] + c[1:])
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        lc = matplotlib.collections.LineCollection(segments, cmap=cmap, norm=norm, linestyle=style)
        lc.set_array(cs)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)

        legend_lines.append(Line2D([0], [0], color=cmap(0.5), linestyle=style, linewidth=2))
        legend_names.append(label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)

    ax.legend(legend_lines, legend_names)

    plt.savefig(f"{filename}.png")

def get_limits(scores):
    x_min = math.inf
    x_max = -math.inf
    y_min = math.inf
    y_max = -math.inf

    for label in scores.keys():
        sorted_series = sorted(scores[label], key=lambda x: x[0])
        x, y, shade = zip(*sorted_series)

        x_min = min(x_min, min(x))
        x_max = max(x_max, max(x))
        y_min = min(y_min, min(y))
        y_max = max(y_max, max(y))

    return (x_min, x_max), (y_min, y_max)

if __name__ == "__main__":
    os.makedirs("graphs", exist_ok=True)
    shutil.rmtree("graphs", ignore_errors=True)
    os.makedirs("graphs", exist_ok=True)

    colors = ["Purples", "Blues", "Greens", "Oranges"]
    markers = ["o", "x", "s", "P", "v"]
    styles = ["dotted", "dashdot", "solid", "dashdot"]

    settings = {
        "Linear 256": {"color": "Purples", "style": "dashdot"},
        "Linear 512": {"color": "Blues", "style": "dashdot"},
        "Linear 1024": {"color": "Greens", "style": "dashdot"},
        "Linear 2048": {"color": "Oranges", "style": "dashdot"},
        "PCA (Static)": {"color": "Reds", "style": "dotted"},
        "PCA (TopK)": {"color": "Reds", "style": "dashed"},
    }

    plots = [
        (
            f"{n_ground} features, {n_nz} active, {noise_mag:.2E} noise mag",
            f"plot_{noise_mag:.2E}_{n_ground}_{n_nz}",
            f"output_synthetic_{noise_mag:.2E}_{n_ground}_{n_nz}/generator.pt",
            [
                ("Linear", f"output_synthetic_{noise_mag:.2E}_{n_ground}_{n_nz}/_9/learned_dicts.pt"),
            ]
        )
        for noise_mag, n_ground, n_nz in itertools.product([0.05], [1024], [200])
    ]

    for title, filename, generator, learned_dict_files in plots:
        scores = generate_scores(learned_dict_files, generator_file=generator)
        plot_scores(scores, settings, "Sparsity", "FVU", (0, 512), (0, 1), title, f"graphs/{filename}.png")