import asyncio
import json
import multiprocessing as mp
import os
import pickle
import sys
from typing import List, Tuple

import torch
from datasets import load_dataset
from sklearn import metrics
from sklearn.linear_model import RidgeClassifier
from torchtyping import TensorType
from transformer_lens import HookedTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from activation_dataset import make_tensor_name
from autoencoders.learned_dict import LearnedDict

# set OPENAI_API_KEY environment variable from secrets.json['openai_key']
# needs to be done before importing openai interp bits
with open("secrets.json") as f:
    secrets = json.load(f)
    os.environ["OPENAI_API_KEY"] = secrets["openai_key"]

mp.set_start_method("spawn", force=True)

from neuron_explainer.explanations.explanations import ScoredSimulation
from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator
from neuron_explainer.explanations.prompt_builder import PromptFormat


REPLACEMENT_CHAR = "�"
MAX_CONCURRENT = None
EXPLAINER_MODEL_NAME = "gpt-4" # "gpt-3.5-turbo"
SIMULATOR_MODEL_NAME = "text-davinci-003"

_n_dict_components, _n_sentences, _fragment_len = None, None, None


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

def measure_concept_erasure(
        transformer: HookedTransformer, 
        model: LearnedDict, 
        ablation_t_name: str, 
        features_to_ablate: TensorType["_n_dict_components"], 
        read_t_name: str, 
        tokens: TensorType["_n_sentences", "_fragment_len"], 
        labels: TensorType["_n_sentences", "_fragment_len"], 
        true_threshold: float = 2.0, # threshold for converting [0, 10] to binary
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
    labels_thresholded = labels_reshaped.cpu().numpy() > true_threshold

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
            prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
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


def process_scored_simulation(simulation: ScoredSimulation, tokenizer: HookedTransformer) -> Tuple[List[List[str]], List[List[float]], List[List[int]]]:
    token_strs = [x.simulation.tokens for x in simulation.scored_sequence_simulations]
    sim_activations = [x.simulation.expected_activations for x in simulation.scored_sequence_simulations]

    tokens = [tokenizer.to_tokens("".join(x), prepend_bos=False)[0].tolist() for x in token_strs]
    breakpoint()
    for i in range(len(tokens)):
        assert len(tokens[i]) == len(sim_activations[i]) 
    return token_strs, sim_activations, tokens


def measure_ablation_score() -> None:    
    from argparser import parse_args
    cfg = parse_args()

    cfg.layer = 1
    cfg.layer_loc = "mlp"
    cfg.model_name = "pythia-70m-deduped"
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

    tensor_to_ablate = make_tensor_name(cfg.layer, cfg.layer_loc, cfg.model_name)
    tensor_to_read = make_tensor_name(cfg.layer + 1, layer_loc="residual", model_name=cfg.model_name)
    
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
