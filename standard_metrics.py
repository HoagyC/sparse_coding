import asyncio
import json
import os
import pickle
from typing import List, Tuple, Union, Optional, Dict, Any

from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import torch
from torchtyping import TensorType

from transformer_lens import HookedTransformer

# set OPENAI_API_KEY environment variable from secrets.json['openai_key']
# needs to be done before importing openai interp bits
with open("secrets.json") as f:
    secrets = json.load(f)
    os.environ["OPENAI_API_KEY"] = secrets["openai_key"]

from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
from neuron_explainer.explanations.explanations import ScoredSimulation
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.activations.activations import ActivationRecord

from autoencoders.learned_dict import LearnedDict

import utils
from activation_dataset import setup_data

from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn import metrics

matplotlib.use('Agg')

SIMULATOR_MODEL_NAME = "text-davinci-003" 
PROMPT_FORMAT = PromptFormat.INSTRUCTION_FOLLOWING
MAX_CONCURRENT = 5
REPLACEMENT_CHAR = "�"
TRUE_THRESHOLD = 2

_batch_size, _activation_size, _n_dict_components, _fragment_len, _n_sentences, _n_dicts = None, None, None, None, None, None

async def get_synthetic_dataset(
        explanation: str, 
        sentences: List[str], 
        tokenizer, 
        str_len: int = 64
    ) -> Tuple[List[List[str]], List[List[float]], List[List[int]]]:
    token_strs_list = []
    sim_activations_list = []
    tokens_list = []

    simulator = UncalibratedNeuronSimulator(
        ExplanationNeuronSimulator(
            SIMULATOR_MODEL_NAME,
            explanation,
            max_concurrent=MAX_CONCURRENT,
            prompt_format=PROMPT_FORMAT, # INSTRUCTIONFOLLIWING
        )
    )

    for sentence in sentences:
        # Getting activations - using HookedTransformer to avoid tokenization artefacts like 'Ġand'
        tokens = tokenizer.to_tokens(sentence, prepend_bos=False)[:, :str_len]
        print(tokens.shape)
        token_strs = tokenizer.to_str_tokens(tokens)
        if REPLACEMENT_CHAR in token_strs:
            continue

        simulation = await simulator.simulate(token_strs)

        assert len(simulation.expected_activations) == len(tokens[0]), f"Expected activations length {len(simulation.expected_activations)} does not match tokens length {len(tokens[0])}"

        token_strs_list.append(token_strs)
        sim_activations_list.append(simulation.expected_activations)
        tokens_list.append(tokens[0].tolist())
        breakpoint()

    return token_strs_list, sim_activations_list, tokens_list

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

def run_ablating_model_directions(transformer: HookedTransformer, model: LearnedDict, tensor_name, features_to_ablate, tokens, other_hooks=[], **kwargs):
    def intervention(tensor, hook=None):
        B, L, C = tensor.shape # batch_size, sequence_length, activation_size
        reshaped_tensor = tensor.reshape(B * L, C)

        # to ablate the features, we set them to zero, and then subtract the resulting activations from the original activations
        # then multiply back by the learned dictionary to get the effect on the original input
        feature_activations = model.encode(reshaped_tensor)
        not_ablated = feature_activations.clone()
        not_ablated[:, features_to_ablate] = 0.0 # 
        to_ablate = feature_activations - not_ablated
        ablation = torch.einsum("nd,bn->bd", model.get_learned_dict(), to_ablate)
        reshaped_ablation = ablation.reshape(B, L, C)

        print("norm of new tensor", torch.norm(tensor - reshaped_ablation), "norm of tensor", torch.norm(tensor), "norm of ablation", torch.norm(reshaped_ablation))

        return tensor - reshaped_ablation

    return transformer.run_with_hooks(
        tokens,
        fwd_hooks = other_hooks + [(
            tensor_name,
            intervention
        )],
        **kwargs
    )

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

def measure_concept_erasure(
        transformer: HookedTransformer, 
        model: LearnedDict, 
        ablation_t_name: str, 
        features_to_ablate: TensorType["_n_dict_components"], 
        read_t_name: str, 
        tokens: TensorType["_n_sentences", "_fragment_len"], 
        labels: TensorType["_n_sentences", "_fragment_len"], 
        **kwargs
        ) -> Tuple[float, float, float]:
    """ Testing whether a concept which is believed to be represented by a learned direction 
    is actually represented by that direction. We do this by ablation of the direction, and
    then measuring the effect on the model's predictions/.
    """
    assert tokens.shape == labels.shape, f"tokens shape {tokens.shape} does not match labels shape {labels.shape}"
    assert features_to_ablate.shape == (model.n_feats, ), f"to ablate shape {features_to_ablate.shape} is not {model.n_feats}" # features_to_ablate is a binary vector of length n_feats indicating which features to ablate
    B, L = tokens.shape  # tokens: [batch_size, sequence_length]


    activation_cache = torch.empty(0)
    def read_tensor_hook(tensor, hook=None):
        nonlocal activation_cache
        activation_cache = tensor.clone()
        return tensor
    
    with torch.no_grad():
        transformer.run_with_hooks(
            tokens,
            fwd_hooks = [(read_t_name, read_tensor_hook)],
            **kwargs
        )
    y_true = activation_cache.reshape(B * L, -1)

    with torch.no_grad():
        run_ablating_model_directions(transformer, model, ablation_t_name, features_to_ablate, tokens, other_hooks=[(read_t_name, read_tensor_hook)], **kwargs)
    y_ablated = activation_cache.reshape(B * L, -1)

    labels_reshaped = labels.reshape(B * L)
    labels_thresholded = labels_reshaped.cpu().numpy() > TRUE_THRESHOLD

    # should use different classifiers for each task, we want to compare linear separability
    # just using this for sanity checks atm
    regression_model = RidgeClassifier()
    regression_model.fit(y_true.cpu().numpy(), labels_thresholded)

    y_true_pred = regression_model.predict(y_true.cpu().numpy())
    y_ablated_pred = regression_model.predict(y_ablated.cpu().numpy())

    auroc_true = metrics.roc_auc_score(labels_thresholded, y_true_pred)
    auroc_ablated = metrics.roc_auc_score(labels_thresholded, y_ablated_pred)

    # not sure what summary to use here?
    return 1 - (auroc_ablated - 0.5) / (auroc_true - 0.5), auroc_true, auroc_ablated

def mcs_duplicates(ground: LearnedDict, model: LearnedDict) -> TensorType["_n_dict_components"]:
    # get max cosine sim between each model atom and all ground atoms
    cosine_sim = torch.einsum("md,gd->mg", model.get_learned_dict(), ground.get_learned_dict())
    max_cosine_sim = cosine_sim.max(dim=-1).values
    return max_cosine_sim

def mmcs(model: LearnedDict, model2: LearnedDict) -> float:
    return mcs_duplicates(model, model2).mean().item()

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

def mean_nonzero_activations(model: LearnedDict, batch: TensorType["_batch_size", "_activation_size"]):
    c = model.encode(batch)
    return (c > 0.0).float().mean(dim=0)

def fraction_variance_unexplained(model: LearnedDict, batch: TensorType["_batch_size", "_activation_size"]):
    x_hat = model.predict(batch)
    residuals = (batch - x_hat).pow(2).mean()
    total = (batch - batch.mean(dim=0)).pow(2).mean()
    return residuals / total

def r_squared(model: LearnedDict, batch: TensorType["_batch_size", "_activation_size"]):
    return 1.0 - fraction_variance_unexplained(model, batch)

def neurons_per_feature(model: LearnedDict) -> float:
    """ Gets the average numbrer of neurons attended to per learned feature, as measured by the Simpson diversity index."""
    c: TensorType["_n_dict_components", "_activation_size"] = model.get_learned_dict()
    c = c / c.abs().sum(dim=-1, keepdim=True)
    c = c.pow(2).sum(dim=-1)
    return (1.0 / c).mean()

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

def process_scored_simulation(simulation: ScoredSimulation, tokenizer: HookedTransformer) -> Tuple[List[List[str]], List[List[float]], List[List[int]]]:
    token_strs = [x.simulation.tokens for x in simulation.scored_sequence_simulations]
    sim_activations = [x.simulation.expected_activations for x in simulation.scored_sequence_simulations]

    tokens = [tokenizer.to_tokens("".join(x), prepend_bos=False)[0].tolist() for x in token_strs]
    breakpoint()
    for i in range(len(tokens)):
        assert len(tokens[i]) == len(sim_activations[i]) 
    return token_strs, sim_activations, tokens


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

def measure_ablation_score() -> None:    
    from argparser import parse_args
    cfg = parse_args()

    cfg.layer = 1
    cfg.use_residual = False
    cfg.model_name = "EleutherAI/pythia-70m-deduped"
    cfg.fresh_synth_data = False

    device = "cuda:0"
    dicts = torch.load("output/_0/learned_dicts.pt")
    pile_10k = load_dataset("NeelNanda/pile-10k")
    transformer = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device=device)

    sentences = pile_10k["train"]["text"][:10]

    feature_id = 11

    if cfg.fresh_synth_data:
        token_strs, sim_activations, tokens = asyncio.run(
            get_synthetic_dataset(
                explanation=" mathematical expressions and sequences.", 
                sentences=sentences,
                tokenizer=transformer,
            )
        )
    else:
        data_loc = f"auto_interp_results/pythia-70m-deduped_layer1_postnonlin/ld_mlp/feature_{feature_id}/scored_simulation.pkl"
        with open(data_loc, "rb") as file:
            scored_simulation = pickle.load(file)
        token_strs, sim_activations, tokens = process_scored_simulation(scored_simulation, tokenizer=transformer)

    model: LearnedDict = torch.load(cfg.load_interpret_autoencoder)
    model.to_device(device)

    features_to_ablate = torch.zeros(model.n_feats).to(dtype=torch.bool)
    features_to_ablate[feature_id] = 1

    tensor_to_ablate = utils.make_tensor_name(cfg.layer, cfg.use_residual, cfg.model_name)
    tensor_to_read = utils.make_tensor_name(cfg.layer+1, use_residual=True, model_name=cfg.model_name)
    
    print(f"num nonzero activations: {(torch.Tensor(sim_activations) > 2).sum()}")
    breakpoint()

    erasure_score, true, ablated = measure_concept_erasure(
        transformer,
        model,
        tensor_to_ablate,
        features_to_ablate,
        tensor_to_read,
        torch.Tensor(tokens).to(dtype=torch.int32),
        torch.Tensor(sim_activations),
    )

    activations: TensorType["_batch_size", "_activation_size"] = None
    def copy_to_activations(tensor, hook=None):
        global activations
        activations = tensor.clone()
        return tensor

    with torch.no_grad():
        transformer.run_with_hooks(
            torch.Tensor(tokens).to(dtype=torch.int32),
            fwd_hooks = [(tensor_to_read, copy_to_activations)]
        )

    # r_sq = fraction_variance_unexplained(model, activations.reshape(-1, activations.shape[-1])).item()

    print(erasure_score, true, ablated)

def make_one_chunk_per_layer() -> None:
    device = torch.device("cuda:1")
    model_name = "EleutherAI/pythia-70m-deduped"
    model = HookedTransformer.from_pretrained(model_name, device=device)
    tokenizer = model.tokenizer

    for use_residual in [False, True]:
        activation_width = 512 if use_residual else 2048
        for layer in range(6):
            setup_data(
                tokenizer,
                model,
                model_name=model_name,
                activation_width=activation_width,
                dataset_name="EleutherAI/pile",
                dataset_folder=f"single_chunks/l{layer}_{'resid' if use_residual else 'mlp'}",
                layer=layer,
                use_residual=use_residual,
                use_baukit=False,
                n_chunks=1,
                device=device,
            )

if __name__ == "__main__":
    make_one_chunk_per_layer()

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
