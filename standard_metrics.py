import asyncio
from functools import partial
from itertools import product
import multiprocessing as mp
import os
import pickle
from typing import List, Tuple, Union, Any, Dict, Literal

from datasets import load_dataset
from einops import rearrange
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader
from torchtyping import TensorType

import tqdm

from transformer_lens import HookedTransformer

from autoencoders.learned_dict import LearnedDict

from activation_dataset import setup_data

from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn import metrics

matplotlib.use('Agg')

_batch_size, _activation_size, _n_dict_components, _fragment_len, _n_sentences, _n_dicts = None, None, None, None, None, None # type: Tuple[None, None, None, None, None, None]

def run_with_model_intervention(transformer: HookedTransformer, model: LearnedDict, tensor_name, tokens, other_hooks=[], **kwargs):
    def intervention(tensor, hook=None):
        B, L, C = tensor.shape
        reshaped_tensor = tensor.reshape(B * L, C)

        prediction = model.predict(tensor)
        reshaped_prediction = prediction.reshape(B * L, C)

        return reshaped_prediction

    return transformer.run_with_hooks(
        tokens,
        fwd_hooks = other_hooks + [(
            tensor_name,
            intervention
        )],
        **kwargs
    )


Location = Tuple[int, Literal["residual", "mlp"]]

def get_model_tensor_name(location: Location) -> str:
    if location[1] == "residual":
        return f"blocks.{location[0]}.hook_resid_post"
    elif location[1] == "mlp":
        return f"blocks.{location[0]}.mlp.hook_post"
    elif location[1] == "attn_concat":
        return f"blocks.{location[0]}.attn.hook_z"
    else:
        raise ValueError(f"Location '{location[1]}' not supported")


def ablate_feature_intervention(model, location, feature):
    def go(tensor, hook=None):
        B, L, C = tensor.shape
        
        activation_at_position = tensor[:, feature[0], :]
        feat_activations = model.encode(activation_at_position)
        ablation_mask = torch.zeros_like(feat_activations)
        ablation_mask[:, feature[1]] = 1.0
        ablated_feat_activations = feat_activations * ablation_mask
        ablation = torch.einsum("nd,bn->bd", model.get_learned_dict(), ablated_feat_activations)

        tensor[:, feature[0], :] -= ablation

        return tensor
    
    return go

def cache_all_activations(
    transformer: HookedTransformer,
    models: Dict[Location, LearnedDict],
    tokens: TensorType["_n_sentences", "_fragment_len"],
    **kwargs
) -> Dict[Location, torch.Tensor]:
    tensor_names = [get_model_tensor_name(location) for location in models.keys()]

    _, activation_cache = transformer.run_with_cache(
        tokens,
        names_filter=tensor_names,
        **kwargs
    )

    activations = {}
    for location, model in models.items():
        tensor_name = get_model_tensor_name(location)
        tensor = activation_cache[tensor_name]

        B, L, C = tensor.shape
        tensor = tensor.reshape(B * L, C)
        encoded = model.encode(tensor)
        activations[location] = encoded.reshape(B, L, -1)
    
    return activations

# model location, feature index
FeatureIdx = Tuple[int, int]
Feature = Tuple[Location, FeatureIdx]
FeatureNoPos = Tuple[Location, int]

def build_ablation_graph(
    transformer: HookedTransformer,
    models: Dict[Location, LearnedDict],
    tokens: TensorType["_n_sentences", "_fragment_len"],
    features_to_ablate: Dict[Location, List[FeatureIdx]] = {},
    target_features: Dict[Location, List[FeatureIdx]] = {},
) -> Dict[Tuple[Feature, Feature], float]:
    
    B, L = tokens.shape

    if not features_to_ablate:
        features_to_ablate = {location: list(product(range(L), range(model.get_learned_dict().shape[0]))) for location, model in models.items()}

    if not target_features:
        target_features = {}
    
    all_features = [(location, feature) for location, features in {**features_to_ablate, **target_features}.items() for feature in features]

    activations = cache_all_activations(transformer, models, tokens)

    graph = {}
    for location, model in models.items():
        for feature in tqdm.tqdm(features_to_ablate[location]):
            tensor_name = get_model_tensor_name(location)

            ablated_activations = cache_all_activations(
                transformer,
                models,
                tokens,
                fwd_hooks = [(
                    tensor_name,
                    ablate_feature_intervention(model, location, feature)
                )]
            )

            # maybe later on compress into a single op
            for location_, feature_ in all_features:
                if location_ == location and feature_ == feature:
                    continue
                
                unablated = activations[location_][:, feature_[0], feature_[1]]
                ablated = ablated_activations[location_][:, feature_[0], feature_[1]]
                graph[(location, feature), (location_, feature_)] = torch.norm(unablated - ablated, dim=-1).mean().item()
    
    return graph

def ablate_feature_intervention_non_positional(model, location, feature_idx):
    def go(tensor, hook=None):
        B, L, C = tensor.shape
        
        feat_activations = model.encode(tensor.reshape(B * L, C))
        ablation_mask = torch.zeros_like(feat_activations)
        ablation_mask[:, feature_idx] = 1.0
        ablated_feat_activations = feat_activations * ablation_mask
        ablation = torch.einsum("nd,bn->bd", model.get_learned_dict(), ablated_feat_activations)

        tensor -= ablation.reshape(B, L, C)

        return tensor
    
    return go

def build_ablation_graph_non_positional(
    transformer: HookedTransformer,
    models: Dict[Location, LearnedDict],
    tokens: TensorType["_n_sentences", "_fragment_len"],
    features_to_ablate: Dict[Location, List[int]] = {},
    target_features: Dict[Location, List[int]] = {},
) -> Dict[Tuple[FeatureNoPos, FeatureNoPos], float]:
    B, L = tokens.shape

    if not features_to_ablate:
        features_to_ablate = {location: list(range(model.get_learned_dict().shape[0])) for location, model in models.items()}

    if not target_features:
        target_features = {}
    
    all_features = [(location, feature) for location, features in {**features_to_ablate, **target_features}.items() for feature in features]

    activations = cache_all_activations(transformer, models, tokens)

    graph = {}
    for location, model in models.items():
        for feature in tqdm.tqdm(features_to_ablate[location]):
            tensor_name = get_model_tensor_name(location)

            ablated_activations = cache_all_activations(
                transformer,
                models,
                tokens,
                fwd_hooks = [(
                    tensor_name,
                    ablate_feature_intervention_non_positional(model, location, feature)
                )]
            )

            # maybe later on compress into a single op
            for location_, feature_ in all_features:
                if location_ == location and feature_ == feature:
                    continue
                
                unablated = activations[location_][:, :, feature_]
                ablated = ablated_activations[location_][:, :, feature_]
                graph[(location, feature), (location_, feature_)] = torch.norm(unablated - ablated, dim=-1).mean().item()
    
    return graph

def perplexity_under_reconstruction(
    transformer: HookedTransformer,
    model: LearnedDict,
    location: Location,
    tokens: TensorType["_n_sentences", "_fragment_len"],
    **kwargs
):
    def intervention(tensor, hook=None):
        B, L, C = tensor.shape

        reshaped_tensor = tensor.reshape(B * L, C)
        prediction = model.predict(reshaped_tensor)
        reshaped_prediction = prediction.reshape(B, L, C)

        return reshaped_prediction
    
    tensor_name = get_model_tensor_name(location)

    loss = transformer.run_with_hooks(
        tokens,
        fwd_hooks = [(
            tensor_name,
            intervention
        )],
        return_type="loss",
        **kwargs
    )

    return loss

def logistic_regression_auroc(activations: TensorType["_batch_size", "_activation_size"], labels: TensorType["_batch_size"], **kwargs):
    clf = LogisticRegression(**kwargs)

    activations_, labels_ = activations.cpu().numpy(), labels.cpu().numpy()

    clf.fit(activations_, labels_)
    return metrics.roc_auc_score(labels_, clf.predict_proba(activations_)[:, 1])

def ridge_regression_auroc(activations: TensorType["_batch_size", "_activation_size"], labels: TensorType["_batch_size"], **kwargs):
    clf = RidgeClassifier(**kwargs)

    activations_, labels_ = activations.cpu().numpy(), labels.cpu().numpy()

    clf.fit(activations_, labels_)
    return metrics.roc_auc_score(labels_, clf.predict(activations_))

def mcs_duplicates(ground: LearnedDict, model: LearnedDict) -> TensorType["_n_dict_components"]:
    # get max cosine sim between each model atom and all ground atoms
    cosine_sim = torch.einsum("md,gd->mg", model.get_learned_dict(), ground.get_learned_dict())
    max_cosine_sim = cosine_sim.max(dim=-1).values
    return max_cosine_sim

def mmcs(model: LearnedDict, model2: LearnedDict):
    return mcs_duplicates(model, model2).mean()

def mcs_to_fixed(model: LearnedDict, truth: TensorType["_n_dict_components", "_activation_size"]):
    cosine_sim = torch.einsum("md,gd->mg", model.get_learned_dict(), truth)
    max_cosine_sim = cosine_sim.max(dim=-1).values
    return max_cosine_sim

def mmcs_to_fixed(model: LearnedDict, truth: TensorType["_n_dict_components", "_activation_size"]):
    return mcs_to_fixed(model, truth).mean()

def mmcs_from_list(ld_list: List[LearnedDict]) -> TensorType["_n_dicts", "_n_dicts"]:
    """
    Returns a lower triangular matrix of mmcs between all pairs of dicts in the list.
    """
    n_dicts = len(ld_list)
    mmcs_t = torch.eye(n_dicts)
    for i in range(n_dicts):
        for j in range(i):
            mmcs_t[i, j] = mmcs(ld_list[i], ld_list[j])
            mmcs_t[j, i] = mmcs_t[i, j]
    return mmcs_t

def representedness(features: TensorType["_n_dict_components", "_activation_size"], model: LearnedDict):
    # mmcs but other way around
    cosine_sim = torch.einsum("gd,md->gm", features, model.get_learned_dict())
    max_cosine_sim = cosine_sim.max(dim=-1).values
    return max_cosine_sim

def mean_nonzero_activations(model: LearnedDict, batch: TensorType["_batch_size", "_activation_size"]):
    batch_centered = model.center(batch)
    c = model.encode(batch_centered)
    return (c != 0).float().mean(dim=0)

def fraction_variance_unexplained(model: LearnedDict, batch: TensorType["_batch_size", "_activation_size"]):
    x_hat = model.predict(batch)
    residuals = (batch - x_hat).pow(2).mean()
    total = (batch - batch.mean(dim=0)).pow(2).mean()
    return residuals / total

def fraction_variance_unexplained_top_activating(model: LearnedDict, batch: TensorType["_batch_size", "_activation_size"], n_top = 2):
    # get the fvu of the top-activating neurons for each datapoint,
    # and the fvu for the rest of the neurons

    c = model.encode(model.center(batch))

    # calculate the mean activation for each neuron
    mean_activation = c.mean(dim=0)
    idxs = torch.argsort(mean_activation, descending=True)
    top_n_idxs = idxs[:n_top]
    rest_idxs = idxs[n_top:]

    c_top = torch.zeros_like(c)
    c_top[:, top_n_idxs] = c[:, top_n_idxs]

    c_rest = torch.zeros_like(c)
    c_rest[:, rest_idxs] = c[:, rest_idxs]

    x_hat_top = model.center(model.decode(c_top))
    x_hat_rest = model.center(model.decode(c_rest))

    residuals_top = (batch - x_hat_top).pow(2).mean()
    residuals_rest = (batch - x_hat_rest).pow(2).mean()

    variance = (batch - batch.mean(dim=0)).pow(2).mean()

    return residuals_top / variance, residuals_rest / variance

def r_squared(model: LearnedDict, batch: TensorType["_batch_size", "_activation_size"]):
    return 1.0 - fraction_variance_unexplained(model, batch)

def neurons_per_feature(model: LearnedDict) -> float:
    """ Gets the average numbrer of neurons attended to per learned feature, as measured by the Simpson diversity index."""
    c: TensorType["_n_dict_components", "_activation_size"] = model.get_learned_dict()
    c = c / c.abs().sum(dim=-1, keepdim=True)
    c = c.pow(2).sum(dim=-1)
    return (1.0 / c).mean()

# calculating the capacity metric from Scherlis et al 2022 
# https://arxiv.org/pdf/2210.01892.pdf
def capacity_per_feature(model: LearnedDict) -> TensorType["_n_dict_components"]:
    learned_dict: TensorType["_n_dict_components", "_activation_size"] = model.get_learned_dict()

    squared_dot_products = torch.einsum("md,nd->mn", learned_dict, learned_dict).pow(2)
    sum_of_sq_dot = squared_dot_products.sum(dim=-1)
    capacities = torch.diag(squared_dot_products) / sum_of_sq_dot
    return capacities

def plot_capacities(dicts: List[Tuple[LearnedDict, Dict[str, Any]]], show: bool =False, save_name: str = "capacities") -> None:
    max_capacity = dicts[0][0].activation_size
    capacity_sums = [sum(capacity_per_feature(d[0])) for d in dicts]
    l1_values = [d[1]["l1_alpha"] for d in dicts]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(l1_values, capacity_sums)
    ax.set_xlabel("L1 alpha")
    ax.set_ylabel("Sum of capacities")
    ax.set_xscale("log")
    ax.axhline(max_capacity, color="red", linestyle="--")
    ax.set_ylim(0, max_capacity * 1.1)
    ax.set_title(f"Sum of capacities vs L1 alpha - {save_name}")
    if show:
        plt.show()
    plt.savefig(save_name + ".png")

def plot_capacity_scatter(dicts: list[Tuple[LearnedDict, Dict[str, Any]]], show: bool = False, save_name: str = "capacity_scatter") -> None:
    all_capacities = []
    for i, (dict, hparams) in enumerate(dicts):
        capacities = capacity_per_feature(dict)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(range(len(capacities)), capacities)
        ax.set_xlabel("Learned feature")
        ax.set_ylabel("Capacity")
        ax.set_title(f"Capacity per feature - {save_name}")
        if show:
            plt.show()
        plt.savefig(save_name + "_" + str(i) + ".png")
        all_capacities.append(capacities)
    
    # plot histogram of capacities
    fig = plt.figure()
    ax = fig.add_subplot(111)
    all_capacities_flat = torch.cat(all_capacities).flatten()
    print(all_capacities_flat.shape)
    ax.hist(all_capacities_flat, bins=80)
    ax.set_xlabel("Capacity")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Capacity histogram - {save_name}")
    if show:
        plt.show()
    plt.savefig(save_name + "_hist.png")


def plot_hist(scores: TensorType["_n_dict_components"], x_label, y_label, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(scores, **kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return Image.fromarray(data, mode="RGB")

def plot_scatter(scores_x: TensorType["_n_dict_components"], scores_y: TensorType["_n_dict_components"], x_label, y_label, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(scores_x, scores_y, **kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return Image.fromarray(data, mode="RGB")

def calc_feature_n_active(batch):
    # batch: [batch_size, n_features]
    n_active = torch.sum(batch != 0, dim=0)
    return n_active

def batched_calc_feature_n_ever_active(learned_dict: LearnedDict, activations: torch.Tensor, batch_size: int = 1000, threshold: int = 10) -> int:
    n_active_count = torch.zeros(learned_dict.n_feats, device=activations.device)
    for i in range(0, len(activations), batch_size):
        batch = activations[i:i+batch_size]
        feat_activations = learned_dict.encode(batch)
        n_active_count += calc_feature_n_active(feat_activations)

    n_active_total = int((n_active_count > threshold).sum().item())
    return n_active_total

def calc_feature_mean(batch):
    # batch: [batch_size, n_features]
    mean = torch.mean(batch, dim=0)
    return mean

def calc_feature_variance(batch):
    # batch: [batch_size, n_features]
    variance = torch.var(batch, dim=0)
    return variance

# weird asymmetric kurtosis/skew with center at 0
def calc_feature_skew(batch):
    # batch: [batch_size, n_features]
    variance = torch.var(batch, dim=0)
    asymm_skew = torch.mean(batch**3, dim=0) / torch.clamp(variance**1.5, min=1e-8)

    return asymm_skew

def calc_feature_kurtosis(batch):
    # batch: [batch_size, n_features]
    variance = torch.var(batch, dim=0)
    asymm_kurtosis = torch.mean(batch**4, dim=0) / torch.clamp(variance**2, min=1e-8)

    return asymm_kurtosis


def calc_moments_streaming(learned_dict, activations, batch_size=1000):
    times_active = torch.zeros(learned_dict.n_feats, device=activations.device)
    mean = torch.zeros(learned_dict.n_feats, device=activations.device)
    m2 = torch.zeros(learned_dict.n_feats, device=activations.device)
    m3 = torch.zeros(learned_dict.n_feats, device=activations.device)
    m4 = torch.zeros(learned_dict.n_feats, device=activations.device)
    
    n = 0
    for i in range(0, len(activations), batch_size):
        batch = activations[i:i+batch_size]
        feature_activations = learned_dict.encode(batch)
        batch_mean = calc_feature_mean(feature_activations)
        batch_m2 = (feature_activations ** 2).mean(dim=0)
        batch_m3 = (feature_activations ** 3).mean(dim=0)
        batch_m4 = (feature_activations ** 4).mean(dim=0)
        
        times_active += (batch_mean != 0).float()
        
        # update
        mean = (n * mean + batch_size * batch_mean) / (n + batch_size)
        m2 = (n * m2 + batch_size * batch_m2) / (n + batch_size)
        m3 = (n * m3 + batch_size * batch_m3) / (n + batch_size)
        m4 = (n * m4 + batch_size * batch_m4) / (n + batch_size)
        
        n += batch_size
    
    var = m2 - mean**2
    skew = m3 / torch.clamp(var**1.5, min=1e-8)
    kurtosis = m4 / torch.clamp(var**2, min=1e-8)
    return times_active, mean, var, skew, kurtosis, m4
    

def plot_grid(scores: np.ndarray, first_tick_labels, second_tick_labels, first_label, second_label, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(scores, **kwargs)
    ax.set_xticks(np.arange(len(first_tick_labels)))
    ax.set_yticks(np.arange(len(second_tick_labels)))
    ax.set_xticklabels(first_tick_labels)
    ax.set_yticklabels(second_tick_labels)
    ax.set_xlabel(first_label)
    ax.set_ylabel(second_label)
    
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return Image.fromarray(data, mode="RGB")


def cluster_vectors(model: LearnedDict, n_clusters: int = 1000, top_clusters: int = 10, save_loc: str = "outputs/top_clusters.txt"):
    # take the direction vectors and cluster them
    # get the direction vectors
    direction_vectors: TensorType["_n_dict_components", "_activation_size"] = model.get_learned_dict()

    # first apply t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, random_state=0)
    direction_vectors_tsne = tsne.fit_transform(direction_vectors)

    # now we're going to cluster the direction vectors
    # first, we'll try k-means
    print("Clustering vectors using kmeans")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(direction_vectors_tsne)
    # now get the clusters which have the most points in them and get the ids of the points in those clusters
    cluster_ids, cluster_counts = np.unique(kmeans.labels_, return_counts=True)
    cluster_ids = cluster_ids[np.argsort(cluster_counts)[::-1]]
    cluster_counts = cluster_counts[np.argsort(cluster_counts)[::-1]]   
    # now get the ids of the points in the top 10 clusters
    top_cluster_ids = cluster_ids[:top_clusters]
    top_cluster_points = []
    for cluster_id in top_cluster_ids:
        top_cluster_points.append(np.where(kmeans.labels_ == cluster_id)[0])

    # save clusters as separate lines on a text file
    with open(save_loc, "w") as f:
        for cluster in top_cluster_points:
            f.write(f"{list(cluster)}\n")

    # now want to take a selection of points, and find the nearest neighbours to them
    # first, take a random selection of points
    # n_points = 10
    # random_points = np.random.choice(direction_vectors_tsne.shape[0], n_points, replace=False)
    # # now find the nearest neighbours to these points
    # nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(direction_vectors_tsne)


def hierarchical_cluster_vectors(vectors: TensorType["_n_dict_components", "_activation_size"], n_clusters=100, show = True):
    from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree 
    linkage_matrix = linkage(vectors, 'average', metric='cosine') # computes the distance matrix
    dendrogram(linkage_matrix, labels=list(range(vectors.shape[0])), leaf_rotation=90, leaf_font_size=8)
    if show:
        # set backend not to be agg so that we can see the dendrogram
        plt.switch_backend("TkAgg")
        plt.show()
    clusters = cut_tree(linkage_matrix, n_clusters=n_clusters)
    return clusters


def make_one_chunk_per_layer() -> None:
    device = torch.device("cuda:1")
    model_name = "EleutherAI/pythia-70m-deduped"
    model = HookedTransformer.from_pretrained(model_name, device=device)
    tokenizer = model.tokenizer

    for layer_loc in ["residual", "mlp", "mlpout", "attn"]:
        for layer in range(6):
            setup_data(
                tokenizer,
                model,
                dataset_name="EleutherAI/pile",
                dataset_folder=f"/mnt/ssd-cluster/single_chunks/l{layer}_{layer_loc}",
                layer=layer,
                layer_loc=layer_loc,
                n_chunks=1,
                device=device,
                start_line=1_000_000,
            )

def make_one_chunk_per_layer_gpt2sm() -> None:    
    device = torch.device("cuda:4")
    model_name = "gpt2"
    model = HookedTransformer.from_pretrained(model_name, device=device)
    tokenizer = model.tokenizer
    
    for layer_loc in ["residual"]:
        for layer in range(12):
            setup_data(
                tokenizer,
                model,
                dataset_name="openwebtext",
                dataset_folder=f"/mnt/ssd-cluster/single_chunks_gpt2sm/l{layer}_{layer_loc}",
                layer=layer,
                layer_loc=layer_loc,
                n_chunks=1,
                device=device,
            )

def calculate_perplexity(
        model: HookedTransformer, 
        autoencoders: Union[Tuple[LearnedDict, Dict], List[Tuple[LearnedDict, Dict]]],
        layer: int,
        setting: str,
        dataset_name: str = "NeelNaanda/pile-10k",
        model_batch_size: int = 32,
        fragment_len: int = 256
    ) -> Tuple[float, List[float]]:
    """
    Takes an autoencoder or list of autoencoders, and calculates the perplexity of the model 
    on the dataset when the activations of the layer are replaced with the reconstruction 
    of the autoencoder. 
    Returns the original perplexity and a list containing the perplexity for each autoencoder.
    """
    if isinstance(autoencoders, tuple): # if only one autoencoder, make it a list
        autoencoders = [autoencoders]
    num_dictionaries = len(autoencoders)    

    # Define function to replace activations with reconstruction
    def replace_with_reconstruction(value, hook, autoencoder):
        # Rearrange to fit autoencoder
        int_val = rearrange(value, 'b s h -> (b s) h')
        # Run through the autoencoder
        reconstruction = autoencoder.predict(int_val)
        batch, seq_len, hidden_size = value.shape
        reconstruction = rearrange(reconstruction, '(b s) h -> b s h', b=batch, s=seq_len)
        return reconstruction

    # Load model
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = model.eval()

    assert setting in ["residual", "mlp"], "setting must be either 'residual' or 'mlp'"
    if setting == "residual":
        cache_name = f"blocks.{layer}.hook_resid_post"
    elif setting == "mlp":
        cache_name = f"blocks.{layer}.mlp.hook_post"
    else:
        raise NotImplementedError

    dataset = load_dataset(dataset_name, split="train").map(
        lambda x: model.tokenizer(x['text']),
        batched=True,
    ).filter(
        lambda x: len(x['input_ids']) > fragment_len
    ).map(
        lambda x: {'input_ids': x['input_ids'][:fragment_len]}
    )

    with torch.no_grad(), dataset.formatted_as("pt"):
        dl = DataLoader(dataset["input_ids"], batch_size=model_batch_size, shuffle=False)
        # Calculate Original Perplexity ie no intervention/no dictionary
        total_loss = 0
        for i, batch in enumerate(dl):
            loss = model(batch.to(device), return_type="loss")
            total_loss += loss.item()
        # Average
        avg_neg_log_likelihood_orig = torch.tensor(total_loss / len(dl)).to(device)
        # Exponentiate to compute perplexity
        original_perplexity = torch.exp(avg_neg_log_likelihood_orig)
        print("Perplexity for original model: ", original_perplexity.item())

        # Compute perplexity for each dictionary
        all_perplexities = np.zeros(num_dictionaries)
        # Calculate Perplexity for each dictionary
        for dict_index in range(num_dictionaries):
            autoencoder, hparams = autoencoders[dict_index]
            autoencoder.to_device(device)
            total_loss = 0
            for i, batch in enumerate(dl):
                # Perplexity with reconstructed activations
                loss = model.run_with_hooks(batch.to(device), 
                    return_type="loss",
                    fwd_hooks=[(
                        cache_name, # intermediate activation that we intervene on
                        partial(replace_with_reconstruction, autoencoder=autoencoder), # function to apply to cache_name
                        )]
                    )
                total_loss += loss.item()

            # Average
            avg_neg_log_likelihood_recon = torch.tensor(total_loss / len(dl)).to(device)
            # Exponentiate to compute perplexity
            recon_perplexity = torch.exp(avg_neg_log_likelihood_recon)
            print(f"Perplexity for hparams {hparams}: {recon_perplexity.item():.2f}")
            all_perplexities[dict_index] = recon_perplexity.item()

    return original_perplexity.item(), all_perplexities.tolist()

def calc_for_layer(args) -> Tuple[int, List[Tuple[int, List[Tuple[float, float]]]]]:
    layer: int
    layer_loc: str
    ratios: List[int]
    device: torch.device
    base_dir: str
    layer, layer_loc, ratios, device, base_dir = args

    chunk_loc = f"/mnt/ssd-cluster/single_chunks/l{layer}_{'residual' if layer_loc == 'resid' else 'mlp'}/0.pt"
    activations = torch.load(chunk_loc, map_location=device).to(torch.float32)
    dead_feats_data: List[Tuple[int, List[Tuple[float, float]]]] = []
    with torch.no_grad():
        for ratio in ratios:
            dicts_loc = f"output_hoagy_dense_sweep_tied_{layer_loc}_l{layer}_r{ratio}"
            all_dicts = torch.load(os.path.join(base_dir, dicts_loc, "_9", "learned_dicts.pt"))
            dead_feats_data_series: List[Tuple[float, float]] = []
            batch_size = int(1e6 // (ratio + 1))
            for learned_dict, hparams in all_dicts:
                learned_dict.to_device(device)
                n_active = torch.zeros(learned_dict.n_feats, dtype=torch.int64, device=device)
                for i in range(0, activations.shape[0], batch_size):
                    feat_activations = learned_dict.encode(activations[i:i+batch_size])
                    n_active += calc_feature_n_active(feat_activations)
                
                active_feats = (n_active > 10).sum().item()
                print(layer, layer_loc, ratio, hparams["l1_alpha"], active_feats, active_feats / learned_dict.n_feats)
                dead_feats_data_series.append((hparams["l1_alpha"], active_feats / learned_dict.n_feats))
            dead_feats_data.append((ratio, dead_feats_data_series))

    return layer, dead_feats_data

def calc_all_activities():
    base_dir = "/home/mchorse/sparse_coding_hoagy"
    layer_loc = "resid"
    layers = [0,1,2,3,4,5]
    ratios = [0, 1, 2, 4, 8, 16, 32]

    assert torch.cuda.is_available()

    devices = [torch.device(f"cuda:{i}") for i in [0,1,2,3,4,6]] # 5 is busy
    tasks = [(layer, layer_loc, ratios, devices[i], base_dir) for i, layer in enumerate(layers)]
    
    with mp.Pool(6) as p:
        results = p.map(calc_for_layer, tasks)

    pickle.dump(results, open("n_active_data.pkl", "wb"))

def calc_kurtosis_for_layer(args) -> Tuple[int, List[Tuple[int, List[Tuple[float, float, float]]]]]:
    layer: int
    layer_loc: str
    ratios: List[int]
    device: torch.device
    base_dir: str
    layer, layer_loc, ratios, device, base_dir = args

    chunk_loc = f"/mnt/ssd-cluster/single_chunks/l{layer}_{'residual' if layer_loc == 'resid' else 'mlp'}/0.pt"
    activations = torch.load(chunk_loc, map_location=device).to(torch.float32)
    dead_feats_data: List[Tuple[int, List[Tuple[float, float, float]]]] = []
    with torch.no_grad():
        for ratio in ratios:
            dicts_loc = f"output_hoagy_dense_sweep_tied_{layer_loc}_l{layer}_r{ratio}"
            all_dicts = torch.load(os.path.join(base_dir, dicts_loc, "_9", "learned_dicts.pt"))
            dead_feats_data_series: List[Tuple[float, float, float]] = []
            batch_size = int(1e6 // (ratio + 1))
            for learned_dict, hparams in all_dicts:
                learned_dict.to_device(device)
                n_active = torch.zeros(learned_dict.n_feats, dtype=torch.int64, device=device)
                kurtoses = torch.zeros(learned_dict.n_feats, dtype=torch.float32, device=device)
                for i in range(0, activations.shape[0], batch_size):
                    feat_activations = learned_dict.encode(activations[i:i+batch_size])
                    n_active += calc_feature_n_active(feat_activations)
                    kurtoses += calc_feature_kurtosis(feat_activations)
                
                active_feats = (n_active > 10)
                kurtoses = kurtoses / (activations.shape[0] // batch_size)
                av_kurtosis_all = kurtoses.mean().item()
                av_kurtosis_active = kurtoses[active_feats].mean().item()
                print(layer, layer_loc, ratio, hparams["l1_alpha"], av_kurtosis_all, av_kurtosis_active)
                dead_feats_data_series.append((hparams["l1_alpha"], av_kurtosis_all, av_kurtosis_active))
            dead_feats_data.append((ratio, dead_feats_data_series))

    return layer, dead_feats_data

def calc_all_kurtosis():
    base_dir = "/home/mchorse/sparse_coding_hoagy"
    layer_loc = "resid"
    layers = [0,1,2,3,4,5]
    ratios = [0, 1, 2, 4, 8, 16, 32]

    assert torch.cuda.is_available()

    devices = [torch.device(f"cuda:{i}") for i in [0,1,2,3,4,6]] # 5 is busy
    tasks = [(layer, layer_loc, ratios, devices[i], base_dir) for i, layer in enumerate(layers)]
    
    with mp.Pool(6) as p:
        results = p.map(calc_kurtosis_for_layer, tasks)

    pickle.dump(results, open("kurtosis_data.pkl", "wb"))


def run_mmcs_with_larger(learned_dicts, threshold=0.9, device: Union[str, torch.device] = "cpu"):
    n_l1_coefs, n_dict_sizes = len(learned_dicts), len(learned_dicts[0])
    av_mmcs_with_larger_dicts = np.zeros((n_l1_coefs, n_dict_sizes))
    feats_above_threshold = np.zeros((n_l1_coefs, n_dict_sizes))
    full_max_cosine_sim_for_histograms = np.empty((n_l1_coefs, n_dict_sizes-1), dtype=object)


    for l1_ndx, dict_size_ndx in tqdm(list(product(range(n_l1_coefs), range(n_dict_sizes)))):
        if dict_size_ndx == n_dict_sizes - 1:
            continue
        smaller_dict = learned_dicts[l1_ndx][dict_size_ndx]
        # Clone the larger dict, because we're going to zero it out to do replacements
        larger_dict_clone = learned_dicts[l1_ndx][dict_size_ndx + 1].clone().to(device)
        smaller_dict_features, _ = smaller_dict.shape
        larger_dict_features, _ = larger_dict_clone.shape
        # Hungary algorithm
        from scipy.optimize import linear_sum_assignment
        # Calculate all cosine similarities and store in a 2D array
        cos_sims = np.zeros((smaller_dict_features, larger_dict_features))
        for idx, vector in enumerate(smaller_dict):
            cos_sims[idx] = torch.nn.functional.cosine_similarity(vector.to(device), larger_dict_clone, dim=1).cpu().numpy()
        # Convert to a minimization problem
        cos_sims = 1 - cos_sims
        # Use the Hungarian algorithm to solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(cos_sims)
        # Retrieve the max cosine similarities and corresponding indices
        max_cosine_similarities = 1 - cos_sims[row_ind, col_ind]
        av_mmcs_with_larger_dicts[l1_ndx, dict_size_ndx] = max_cosine_similarities.mean().item()
        threshold = 0.9
        feats_above_threshold[l1_ndx, dict_size_ndx] = (max_cosine_similarities > threshold).sum().item() / smaller_dict_features * 100
        full_max_cosine_sim_for_histograms[l1_ndx][dict_size_ndx] = max_cosine_similarities
    return av_mmcs_with_larger_dicts, feats_above_threshold, full_max_cosine_sim_for_histograms

if __name__ == "__main__":
    make_one_chunk_per_layer_gpt2sm()

    # dicts = torch.load("/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r4/_9/learned_dicts.pt")
    # plot_capacity_scatter(dicts, save_name="outputs/capacity_scatter_l2_r4")

    # mp.set_start_method('spawn')
    # calc_all_kurtosis()

    # ld_loc = "output_hoagy_dense_sweep_tied_resid_l2_r4/_38/learned_dicts.pt"
    # learned_dicts: List[Tuple[LearnedDict, Dict[str, Any]]] = torch.load(ld_loc)
    # activations_loc = "pilechunks_l2_resid/0.pt"
    # activations = torch.load(activations_loc).to(torch.float32)

    # for learned_dict, hparams in learned_dicts:
    #     feat_activations = learned_dict.encode(activations)
    #     means = calc_feature_mean(activations)



    # for learned_dict, hparams in learned_dicts:
    #     neurons_per_feat = neurons_per_feature(learned_dict)
    #     l1_value = hparams["l1_alpha"]
    #     print(f"l1: {l1_value}, neurons per feat: {neurons_per_feat}")