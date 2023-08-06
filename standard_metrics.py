import asyncio
import json
import os
import pickle
from typing import List, Tuple, Union, Optional, Dict, Any, Literal

from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
import torch
from torchtyping import TensorType

import tqdm

from transformer_lens import HookedTransformer

"""
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
"""

from autoencoders.learned_dict import LearnedDict

import utils

from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn import metrics

matplotlib.use('Agg')

"""
SIMULATOR_MODEL_NAME = "text-davinci-003" 
PROMPT_FORMAT = PromptFormat.INSTRUCTION_FOLLOWING
MAX_CONCURRENT = 5
REPLACEMENT_CHAR = "�"
TRUE_THRESHOLD = 2
"""

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

def run_ablating_model_directions(transformer: HookedTransformer, model: LearnedDict, tensor_name, features_to_ablate, tokens, other_hooks=[], ablation_positions=[], run_with_cache=False, **kwargs):
    def intervention(tensor, hook=None):
        nonlocal ablation_positions

        B, L, C = tensor.shape # batch_size, sequence_length, activation_size

        if ablation_positions == None:
            ablation_positions = list(range(L))
        
        L_ = len(ablation_positions)

        activations_at_positions = tensor[:, ablation_positions, :]
        reshaped_activations = activations_at_positions.reshape(B * L_, C)
        feat_activations = model.encode(reshaped_activations)
        ablation_mask = torch.zeros_like(feat_activations)
        ablation_mask[:, features_to_ablate] = 1.0
        ablated_feat_activations = feat_activations * ablation_mask
        reshaped_ablation = torch.einsum("nd,bn->bd", model.get_learned_dict(), ablated_feat_activations).reshape(B, L_, C)
        tensor[:, ablation_positions, :] -= reshaped_ablation

        return tensor
    
    if run_with_cache:
        return transformer.run_with_cache(
            tokens,
            fwd_hooks = other_hooks + [(
                tensor_name,
                intervention
            )],
            **kwargs
        )
    else:
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

def build_ablation_graph(
    transformer: HookedTransformer,
    models: Dict[Location, LearnedDict],
    tokens: TensorType["_n_sentences", "_fragment_len"],
    features_to_ablate: Optional[Dict[Location, List[Feature]]] = None,
    root_features: Optional[Dict[Location, List[Feature]]] = None,
) -> Dict[Tuple[Feature, Feature], float]:
    
    B, L = tokens.shape

    if features_to_ablate == None:
        features_to_ablate = {location: list(zip(range(L), range(model.get_learned_dict().shape[0]))) for location, model in models.items()}

    if root_features == None:
        root_features = {}
    
    all_features = [(location, feature) for location, features in {**features_to_ablate, **root_features}.items() for feature in features]

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
    features_to_ablate: Optional[Dict[Location, List[FeatureIdx]]] = None,
    root_features: Optional[Dict[Location, List[FeatureIdx]]] = None,
) -> Dict[Tuple[FeatureIdx, FeatureIdx], float]:
        
        B, L = tokens.shape
    
        if features_to_ablate == None:
            features_to_ablate = {location: list(range(model.get_learned_dict().shape[0])) for location, model in models.items()}
    
        if root_features == None:
            root_features = {}
        
        all_features = [(location, feature) for location, features in {**features_to_ablate, **root_features}.items() for feature in features]
    
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

def mmcs(model: LearnedDict, model2: LearnedDict):
    return mcs_duplicates(model, model2).mean()

def mcs_to_fixed(model: LearnedDict, truth: TensorType["_n_dict_components", "_n_dict_components"]):
    cosine_sim = torch.einsum("md,gd->mg", model.get_learned_dict(), truth)
    max_cosine_sim = cosine_sim.max(dim=-1).values
    return max_cosine_sim

def mmcs_to_fixed(model: LearnedDict, truth: TensorType["_n_dict_components", "_n_dict_components"]):
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

def mean_nonzero_activations(model: LearnedDict, batch: TensorType["_batch_size", "_activation_size"]):
    c = model.encode(batch)
    return (c != 0).float().mean(dim=0)

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

"""
def process_scored_simulation(simulation: ScoredSimulation, tokenizer: HookedTransformer) -> Tuple[List[List[str]], List[List[float]], List[List[int]]]:
    token_strs = [x.simulation.tokens for x in simulation.scored_sequence_simulations]
    sim_activations = [x.simulation.expected_activations for x in simulation.scored_sequence_simulations]

    tokens = [tokenizer.to_tokens("".join(x), prepend_bos=False)[0].tolist() for x in token_strs]
    breakpoint()
    for i in range(len(tokens)):
        assert len(tokens[i]) == len(sim_activations[i]) 
    return token_strs, sim_activations, tokens

def measure_ablation_score():    
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

if __name__ == "__main__":
    ld_loc = "output_hoagy_dense_sweep_tied_resid_l2_r4/_38/learned_dicts.pt"
    learned_dicts: List[Tuple[LearnedDict, Dict[str, Any]]] = torch.load(ld_loc)
    activations_loc = "pilechunks_l2_resid/0.pt"
    activations = torch.load(activations_loc).to(torch.float32)

    for learned_dict, hparams in learned_dicts:
        feat_activations = learned_dict.encode(activations)
        means = calc_feature_mean(activations)



    # for learned_dict, hparams in learned_dicts:
    #     neurons_per_feat = neurons_per_feature(learned_dict)
    #     l1_value = hparams["l1_alpha"]
    #     print(f"l1: {l1_value}, neurons per feat: {neurons_per_feat}")
"""