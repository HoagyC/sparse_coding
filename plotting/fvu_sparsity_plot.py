import itertools
import math
import os
import shutil
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import standard_metrics
from autoencoders.pca import BatchedPCA, PCAEncoder

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
        return -np.log10(hyperparams["l1_alpha"])
    elif score == "dict_size":
        return hyperparams["dict_size"]
    elif score == "top_fvu":
        return standard_metrics.fraction_variance_unexplained_top_activating(learned_dict, dataset)[0].item()
    elif score == "rest_fvu":
        return standard_metrics.fraction_variance_unexplained_top_activating(learned_dict, dataset)[1].item()



def area_under_fvu_sparsity_curve(learned_dict_files, dataset_file=None, generator_file=None, device="cuda:7"):
    if generator_file is not None:
        generator = torch.load(generator_file)
        ground_truth = generator.sparse_component_dict.to(device)
    else:
        generator, ground_truth = None, None

    if dataset_file is None and generator is not None:
        dataset = torch.cat([next(generator) for _ in tqdm.tqdm(range(512))]).to(dtype=torch.float32, device=device)
    else:
        dataset = torch.load(dataset_file).to(dtype=torch.float32, device=device)

    activation_width = dataset.shape[1]

    sample_idxs = np.random.choice(len(dataset), 50000, replace=False)
    sample = dataset[sample_idxs].to(device)

    del dataset

    score_series = {}
    for label, learned_dict_file in tqdm.tqdm(learned_dict_files):
        learned_dicts = torch.load(learned_dict_file)
        # groups = list(set([hyperparams[group_by] for _, hyperparams in learned_dicts]))
        for learned_dict, hyperparams in learned_dicts:
            learned_dict.to_device(device)

            dict_size = hyperparams["dict_size"]
            if dict_size not in score_series:
                score_series[dict_size] = [(1, 0), (0, activation_width)]
            fvu = standard_metrics.fraction_variance_unexplained(learned_dict, sample).item()
            fvu = np.clip(fvu, 0, 1)
            sparsity = standard_metrics.mean_nonzero_activations(learned_dict, sample).sum().item()
            score_series[dict_size].append((fvu, sparsity))

    areas = []
    for dict_size, score_series_ in score_series.items():
        score_series_ = sorted(score_series_, key=lambda x: x[0])
        x, y = zip(*score_series_)
        areas.append((dict_size, np.trapz(y, x)))

    return areas


def score_representedness(learned_dict_files, generator_file, label_fmt="{dict_size}", device="cuda:7"):
    generator = torch.load(generator_file)
    ground_truth = generator.sparse_component_dict.to(device)

    scores = {}
    for _, learned_dict_file in learned_dict_files:
        learned_dicts = torch.load(learned_dict_file)
        for learned_dict, hyperparams in learned_dicts:
            learned_dict.to_device(device)

            if hyperparams not in scores:
                scores[hyperparams] = []
            scores[hyperparams].append(standard_metrics.representedness(ground_truth, learned_dict))

    mean_integrals = {}
    for hyperparams, score in scores.items():
        mean_integrals[hyperparams] = np.trapz(np.mean(score))

    return mean_integrals


def generate_scores(
    learned_dict_files,
    dataset_file=None,
    generator_file=None,
    x_score="sparsity",
    y_score="fvu",
    c_score=None,
    group_by="dict_size",
    label_format="{name} {val:.2E}",
    other_dicts=[],
    device="cuda:7",
):
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
        # groups = list(set([hyperparams[group_by] for _, hyperparams in learned_dicts]))
        for learned_dict, hyperparams in learned_dicts:
            name = label_format.format(name=label, val=hyperparams[group_by])

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
            learned_dict_sets["PCA (TopK)"] = [
                (pca.to_topk_dict(k), {"dict_size": 512, "k": k}) for k in range(1, dataset.shape[1] // 2, 8)
            ]
        if score_pca_rot:
            learned_dict_sets["PCA (Static)"] = [
                (pca.to_rotation_dict(n), {"dict_size": 512, "n": n}) for n in range(1, dataset.shape[1], 8)
            ]

    sample_idxs = np.random.choice(len(dataset), 20000, replace=False)
    sample = dataset[sample_idxs]

    del dataset

    scores = {}
    for label, learned_dict_set in tqdm.tqdm(learned_dict_sets.items()):
        # points = []
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
        sorted_series = [sorted_series[0]] + [
            sorted_series[i] for i in range(1, len(sorted_series)) if sorted_series[i][0] != sorted_series[i - 1][0]
        ]
        x, y, shade = zip(*sorted_series)

        dydx = np.gradient(y, x)
        x_ = (np.array(x)[:-1] + np.array(x)[1:]) / 2
        c_ = (np.array(shade)[:-1] + np.array(shade)[1:]) / 2

        scores_[label] = list(zip(x_, dydx, c_))

    return scores_


def scores_derivative_(scores):
    sorted_series = sorted(scores, key=lambda x: x[0])
    sorted_series = [sorted_series[0]] + [
        sorted_series[i] for i in range(1, len(sorted_series)) if sorted_series[i][0] != sorted_series[i - 1][0]
    ]
    x, y = zip(*sorted_series)

    dydx = np.gradient(y, x)
    x_ = (np.array(x)[:-1] + np.array(x)[1:]) / 2

    return list(zip(x_, dydx))


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


def plot_scores(scores, settings, xlabel, ylabel, xrange, yrange, title, filename, logx=False, logy=False, crange=None):
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

        if crange is not None:
            c = (c - crange[0]) / (crange[1] - crange[0])

        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        cs = 0.5 * (c[:-1] + c[1:])
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

        mc = c.mean()

        if settings[label]["points"]:
            ax.scatter(x, y, c=c, cmap=cmap, norm=norm, marker=style, s=10)

            legend_lines.append(
                Line2D(
                    [0],
                    [0],
                    color=cmap(mc),
                    marker=style,
                    linestyle="None",
                    markersize=10,
                )
            )
            legend_names.append(label)
        else:
            lc = matplotlib.collections.LineCollection(segments, cmap=cmap, norm=norm, linestyle=style)
            lc.set_array(cs)
            lc.set_linewidth(2)
            line = ax.add_collection(lc)

            legend_lines.append(Line2D([0], [0], color=cmap(mc), linestyle=style, linewidth=2))
            legend_names.append(label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # ax.set_xscale("log")

    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)

    if logx:
        ax.set_xscale("log")
    
    if logy:
        ax.set_yscale("log")

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

    colors = ["Reds", "Blues", "Greens", "Purples", "Oranges", "Greys"]
    styles = ["x", "+", ".", "*"]
    styles = ["dashed", "dashdot", "dotted", "solid"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for _ in range(1):
        layer = 2
        files = [
            (f"Dicts", f"output/learned_dicts_epoch_0.pt")
        ]

        title = "Tied Residual"
        filename = "fvu_sparsity"

        dataset_file = "activation_data/layer_9/0.pt"

        scores = generate_scores(files, dataset_file, c_score="neg_log_l1", group_by="dict_size", device=device)

        print(scores)

        settings = {
            label: {"style": style, "color": color, "points": False}
            for (style, color), label in zip(itertools.product(styles, colors), scores.keys())
        }

        plot_scores(
            scores,
            settings,
            "sparsity",
            "top-fvu",
            (0, 768),
            (0, 1),
            "Threshold Activation Perf.",
            f"graphs/{filename}",
            logx=False,
            logy=False,
            crange=(2, 4)
        )