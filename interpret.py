import argparse
import asyncio
from datetime import datetime
import importlib
import json
import multiprocessing as mp
import os
import pickle
import requests
import sys
from typing import Any, Dict, Union, List, Callable

from baukit import Trace
from datasets import load_dataset, ReadInstruction
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, PCA, NMF
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer, AutoTokenizer

from argparser import parse_args
from comparisons import NoCredentialsError
from utils import dotdict, make_tensor_name, upload_to_aws
from nanoGPT_model import GPT
from run import setup_data


# set OPENAI_API_KEY environment variable from secrets.json['openai_key']
# needs to be done before importing openai interp bits
with open("secrets.json") as f:
    secrets = json.load(f)
    os.environ["OPENAI_API_KEY"] = secrets["openai_key"]

mp.set_start_method("spawn", force=True)

from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.activations.activations import ActivationRecordSliceParams, ActivationRecord, NeuronRecord, NeuronId
from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
from neuron_explainer.explanations.explainer import TokenActivationPairExplainer
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.scoring import simulate_and_score, aggregate_scored_sequence_simulations
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator
from neuron_explainer.fast_dataclasses import loads

EXPLAINER_MODEL_NAME = "gpt-4" # "gpt-3.5-turbo"
SIMULATOR_MODEL_NAME = "text-davinci-003" # "text-davinci-003"

OPENAI_MAX_FRAGMENTS = 50000
OPENAI_FRAGMENT_LEN = 64
OPENAI_EXAMPLES_PER_SPLIT = 5
N_SPLITS = 4
TOTAL_EXAMPLES = OPENAI_EXAMPLES_PER_SPLIT * N_SPLITS
REPLACEMENT_CHAR = "�"


# Replaces the load_neuron function in neuron_explainer.activations.activations because couldn't get blobfile to work
def load_neuron(
    layer_index: Union[str, int], neuron_index: Union[str, int],
    dataset_path: str = "https://openaipublic.blob.core.windows.net/neuron-explainer/data/collated-activations",
) -> NeuronRecord:
    """Load the NeuronRecord for the specified neuron."""
    url = os.path.join(dataset_path, str(layer_index), f"{neuron_index}.json")
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Neuron record not found at {url}.")
    neuron_record = loads(response.content)

    if not isinstance(neuron_record, NeuronRecord):
        raise ValueError(
            f"Stored data incompatible with current version of NeuronRecord dataclass."
        )
    return neuron_record

def make_activation_dataset(cfg, model, activation_dim: int,  total_activation_size: int = 512 * 1024 * 1024):
    if cfg.model_name in ["gpt2", "nanoGPT"]:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    elif cfg.model_name == "EleutherAI/pythia-70m-deduped":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    else:
        raise NotImplementedError
    cfg.n_chunks = 1
    dataset_name = cfg.dataset_name.split("/")[-
    1] + "-" + cfg.model_name.split("/")[-1] + "-" + str(cfg.layer)
    cfg.dataset_folder = os.path.join(cfg.datasets_folder, dataset_name)
    if not os.path.exists(cfg.dataset_folder) or len(os.listdir(cfg.dataset_folder)) == 0:
        cfg.n_chunks = 1
        cfg.activation_dim = activation_dim
        setup_data(cfg, tokenizer, model, use_baukit=True)
    chunk_loc = os.path.join(cfg.dataset_folder, f"0.pkl")

    elem_size = 4
    n_activations = total_activation_size // (elem_size * cfg.activation_dim)

    dataset = DataLoader(pickle.load(open(chunk_loc, "rb")), batch_size=n_activations, shuffle=True)
    return dataset, n_activations


def activation_ICA(dataset, n_activations):
    """
    Takes a tensor of activations and returns the ICA of the activations
    """
    ica = FastICA()
    print(f"Fitting ICA on {n_activations} activations")
    ica_start = datetime.now()
    ica.fit(next(iter(dataset))[0].cpu().numpy()) # 1GB of activations takes about 15m
    print(f"ICA fit in {datetime.now() - ica_start}")
    return ica

def activation_PCA(dataset, n_activations):
    pca = PCA()
    print(f"Fitting PCA on {n_activations} activations")
    pca_start = datetime.now()
    pca.fit(next(iter(dataset))[0].cpu().numpy()) # 1GB of activations takes about 40s
    print(f"PCA fit in {datetime.now() - pca_start}")
    return pca

def activation_NMF(dataset, n_activations):
    nmf = NMF()
    print(f"Fitting NMF on {n_activations} activations")
    nmf_start = datetime.now()
    data = next(iter(dataset))[0].cpu().numpy() # 1GB of activations takes an unknown but long time
    # NMF doesn't support negative values, so shift the data to be positive
    data -= data.min()
    nmf.fit(data)
    print(f"NMF fit in {datetime.now() - nmf_start}")
    return nmf

def flex_activation_function(input, activation_str, **kwargs):
    if activation_str == "ica":
        assert "ica" in kwargs
        return torch.tensor(kwargs["ica"].transform(input.cpu()))
    elif activation_str == "pca":
        assert "pca" in kwargs
        return torch.tensor(kwargs["pca"].transform(input.cpu()))
    elif activation_str == "nmf":
        assert "nmf" in kwargs
        return torch.tensor(kwargs["nmf"].transform(input.cpu()))
    elif activation_str == "random":
        assert "random_matrix" in kwargs
        return input @ kwargs["random_matrix"].to(kwargs["device"])
    elif activation_str == "random_bias":
        assert all([k in kwargs for k in ["random_matrix", "bias", "weight"]]), f"Missing keys {set(['random_matrix', 'bias', 'weight']) - set(kwargs.keys())}"
        assert len(kwargs["bias"]) <= input.shape[1]
        # going to divide the biases by the norm of the weight vector
        norms = torch.norm(kwargs["weight"], dim=1)
        assert len(norms) == len(kwargs["bias"])
        adjusted_bias = kwargs["bias"] / norms
        return torch.relu(input @ kwargs["random_matrix"].to(kwargs["device"]) + adjusted_bias[:input.shape[1]])
    elif activation_str == "neuron_basis":
        return input
    elif activation_str == "neuron_relu":
        return torch.relu(input)
    elif activation_str == "neuron_basis_bias":
        # here we take the neuron basis, add a (probably negative) bias term, and then apply a relu
        assert all([k in kwargs for k in ["bias", "weight"]]), f"Missing keys {set(['bias', 'weight']) - set(kwargs.keys())}"
        assert len(kwargs["bias"]) <= input.shape[1]
        norms = torch.norm(kwargs["weight"], dim=1)
        assert len(norms) == len(kwargs["bias"])
        adjusted_bias = kwargs["bias"] / norms
        return torch.relu(input + adjusted_bias[:input.shape[1]])
    elif activation_str == "feature_dict":
        assert "autoencoder" in kwargs
        reconstruction, dict_activations = kwargs["autoencoder"](input)
        return dict_activations
    else:
        raise ValueError(f"Unknown activation function {activation_str}")


def process_fragment(cfg, fragment_id, fragment_tokens, activation_data, tokenizer_model, activation_fn_kwargs: Dict):
    # Project the activations into the feature space
    # Need to define the activation function as a top-level function for mp to be able to pickle it
    feature_activation_data = flex_activation_function(activation_data.detach(), cfg.activation_transform, **activation_fn_kwargs)
    if not isinstance(feature_activation_data, torch.Tensor):
        feature_activation_data = torch.tensor(feature_activation_data)
    
    # Get average activation for each feature
    feature_activation_means = torch.mean(feature_activation_data, dim=0)

    fragment_dict: Dict[str, Any] = {}
    fragment_dict["fragment_id"] = fragment_id
    fragment_dict["fragment_token_ids"] = fragment_tokens[0].tolist()
    fragment_dict["fragment_token_strs"] = tokenizer_model.to_str_tokens(fragment_tokens[0])

    # if there are any question marks in the fragment, throw it away (caused by byte pair encoding)

    if REPLACEMENT_CHAR in fragment_dict["fragment_token_strs"]:
        # throw away the fragment
        return None
    
    # for j in range(feature_activation_means.shape[0]):
    #     fragment_dict[f"feature_{j}_mean"] = feature_activation_means[j].item()
    #     fragment_dict[f"feature_{j}_activations"] = feature_activation_data[:, j].tolist()
    #     assert len(fragment_dict[f"feature_{j}_activations"]) == len(fragment_dict["fragment_token_strs"]), f"Feature {j} has {len(fragment_dict[f'feature_{j}_activations'])} activations but {len(fragment_dict['fragment_token_strs'])} tokens"

    # rebuilding the above loop to be more efficient, using numpy
    fragment_dict["feature_means"] = feature_activation_means.tolist()
    fragment_dict["feature_activations"] = feature_activation_data.cpu().numpy()

    return fragment_dict

def make_feature_activation_dataset(
        model_name: str,
        model: HookedTransformer, 
        layer: int,
        use_residual: bool,
        activation_fn_name: str,
        activation_fn_kwargs: Dict, 
        activation_dim: int,
        use_baukit: bool = False,
        device: str = "cpu",
        n_fragments = OPENAI_MAX_FRAGMENTS,
        random_fragment = True, # used for debugging
    ):
    """
    Takes a specified point of a model, and a dataset. 
    Returns a dataset which contains the activations of the model at that point, 
    for each fragment in the dataset, transformed into the feature space
    """
    sentence_dataset = load_dataset("openwebtext", split="train", streaming=True)

    if model_name == "nanoGPT":
        tokenizer_model = HookedTransformer.from_pretrained("gpt2")
    else:
        tokenizer_model = model
    
    tensor_name = make_tensor_name(layer, use_residual, model_name)
    # make list of sentence, tokenization pairs
    
    iter_dataset = iter(sentence_dataset)

    # Make dataframe with columns for each feature, and rows for each sentence fragment
    # each row should also have the full sentence, the current tokens and the previous tokens

    n_thrown = 0
    n_added = 0
    batch_size = min(20, n_fragments)

    fragment_token_ids_list = []
    fragment_token_strs_list = []

    activation_means_table = np.zeros((n_fragments, activation_dim), dtype=np.float16)
    activation_data_table = np.zeros((n_fragments, activation_dim * OPENAI_FRAGMENT_LEN), dtype=np.float16)
    with torch.no_grad():
        while n_added < n_fragments:
            fragments: List[torch.Tensor] = []
            fragment_strs: List[str] = []
            while len(fragments) < batch_size:
                print(f"Added {n_added} fragments, thrown {n_thrown} fragments\t\t\t\t\t\t", end="\r")
                sentence = next(iter_dataset)
                # split the sentence into fragments
                sentence_tokens = tokenizer_model.to_tokens(sentence["text"], prepend_bos=False)
                n_tokens = sentence_tokens.shape[1]
                # get a random fragment from the sentence - only taking one fragment per sentence so examples aren't correlated]
                if random_fragment:
                    token_start = np.random.randint(0, n_tokens - OPENAI_FRAGMENT_LEN)
                else:
                    token_start = 0
                fragment_tokens = sentence_tokens[:, token_start:token_start + OPENAI_FRAGMENT_LEN]
                token_strs = tokenizer_model.to_str_tokens(fragment_tokens[0])
                if REPLACEMENT_CHAR in token_strs:
                    n_thrown += 1
                    continue

                fragment_strs.append(token_strs)
                fragments.append(fragment_tokens)
            
            tokens = torch.cat(fragments, dim=0)
            assert tokens.shape == (batch_size, OPENAI_FRAGMENT_LEN), tokens.shape

            if use_baukit:
                with Trace(model, tensor_name) as ret:
                    _ = model(tokens)
                    mlp_activation_data = ret.output.to(device)
                    mlp_activation_data = nn.functional.gelu(mlp_activation_data)
            else:
                _, cache = model.run_with_cache(tokens)
                mlp_activation_data = cache[tensor_name].to(device)

            for i in range(batch_size):
                fragment_tokens = tokens[i:i+1, :]
                activation_data = mlp_activation_data[i:i+1, :].squeeze(0)
                token_ids = fragment_tokens[0].tolist()
                    
                feature_activation_data = flex_activation_function(activation_data, activation_fn_name, **activation_fn_kwargs)
                feature_activation_means = torch.mean(feature_activation_data, dim=0)

                activation_means_table[n_added, :] = feature_activation_means.cpu().numpy()
                feature_activation_data = feature_activation_data.cpu().numpy()

                activation_data_table[n_added, :] = feature_activation_data.flatten()
    
                fragment_token_ids_list.append(token_ids)
                fragment_token_strs_list.append(fragment_strs[i])
                
                n_added += 1

                if n_added >= n_fragments:
                    break
            
    print(f"Added {n_added} fragments, thrown {n_thrown} fragments")
    # Now we build the dataframe from the numpy arrays and the lists
    print(f"Making dataframe from {n_added} fragments")
    df = pd.DataFrame()
    df["fragment_token_ids"] = fragment_token_ids_list
    df["fragment_token_strs"] = fragment_token_strs_list
    means_column_names = [f"feature_{i}_mean" for i in range(activation_dim)]
    activations_column_names = [f"feature_{i}_activation_{j}" for j in range(OPENAI_FRAGMENT_LEN) for i in range(activation_dim)] # nested for loops are read left to right
    
    assert feature_activation_data.shape == (OPENAI_FRAGMENT_LEN, activation_dim)
    df = pd.concat([df, pd.DataFrame(activation_means_table, columns=means_column_names)], axis=1)
    df = pd.concat([df, pd.DataFrame(activation_data_table, columns=activations_column_names)], axis=1)
    print(f"Threw away {n_thrown} fragments, made {len(df)} fragments")
    return df

async def main(cfg: dotdict) -> None:
    # Load model
    if cfg.model_name in ["gpt2", "EleutherAI/pythia-70m-deduped"]:
        model = HookedTransformer.from_pretrained(cfg.model_name, device=cfg.device)
        use_baukit = False
        if cfg.model_name == "gpt2":
            resid_width = 768
        elif cfg.model_name == "EleutherAI/pythia-70m-deduped":
            resid_width = 512
    elif cfg.model_name == "nanoGPT":
        model_dict = torch.load(open(cfg.model_path, "rb"), map_location="cpu")["model"]
        model_dict = {k.replace("_orig_mod.", ""): v for k, v in model_dict.items()}
        cfg_loc = cfg.model_path[:-3] + "cfg"  # cfg loc is same as model_loc but with .pt replaced with cfg.py
        cfg_loc = cfg_loc.replace("/", ".")
        model_cfg = importlib.import_module(cfg_loc).model_cfg
        model = GPT(model_cfg).to(cfg.device)
        model.load_state_dict(model_dict)
        use_baukit = True
        resid_width = 32
    else:
        raise ValueError("Model name not recognised")
    if cfg.use_residual:
        activation_width = resid_width
    else:
        activation_width = resid_width * 4
    
    # Load feature dict
    if cfg.activation_transform in ["feature_dict", "neuron_basis_bias", "random_bias"]:
        assert cfg.load_interpret_autoencoder is not None
        if cfg.tied_ae:
            AutoEncoder: Any
            from autoencoders.tied_ae import AutoEncoder
            autoencoder = pickle.load(open(cfg.load_interpret_autoencoder, "rb")).to(cfg.device)
        else:
            from run import AutoEncoder
            autoencoder = pickle.load(open(cfg.load_interpret_autoencoder, "rb")).to(cfg.device)
    
    if cfg.activation_transform == "feature_dict":
        feature_size = autoencoder.decoder.weight.shape[1]
    else:
        feature_size = activation_width
    
    if cfg.activation_transform in ["ica", "pca", "nmf"]:
        activation_dataset, n_activations = make_activation_dataset(cfg, model, activation_dim=feature_size)


    point_name = "resid" if cfg.use_residual else "postnonlin"
    activations_name = f"{cfg.model_name.split('/')[-1]}_layer{cfg.layer}_{point_name}"
    activation_fn_kwargs = {"device": cfg.device}
    if cfg.activation_transform == "neuron_basis":
        print("Using neuron basis activation transform")
    elif cfg.activation_transform == "neuron_relu":
        print("Using neuron relu activation transform")
    elif cfg.activation_transform == "neuron_basis_bias":
        print("Using neuron basis bias activation transform")
        activation_fn_kwargs.update({"bias": autoencoder.encoder[0].bias})
        activation_fn_kwargs.update({"weight": autoencoder.encoder[0].weight})
    elif cfg.activation_transform == "ica":
        print("Using ICA activation transform")
        ica_path = os.path.join("auto_interp_results", activations_name, "ica_1gb.pkl")
        if os.path.exists(ica_path):
            print("Loading ICA")
            ica = pickle.load(open(ica_path, "rb"))
        else:
            ica = activation_ICA(activation_dataset, n_activations)
            os.makedirs(os.path.dirname(ica_path), exist_ok=True)
            pickle.dump(ica, open(ica_path, "wb"))
        
        activation_fn_kwargs.update({"ica": ica})

    elif cfg.activation_transform == "pca":
        print("Using PCA activation transform")
        pca_path = os.path.join("auto_interp_results", activations_name, "pca_1gb.pkl")
        if os.path.exists(pca_path):
            print("Loading PCA")
            pca = pickle.load(open(pca_path, "rb"))
        else:
            pca = activation_PCA(activation_dataset, n_activations)
            os.makedirs(os.path.dirname(pca_path), exist_ok=True)
            pickle.dump(pca, open(pca_path, "wb"))
        activation_fn_kwargs.update({"pca": pca})

    elif cfg.activation_transform == "nmf":
        print("Using NMF activation transform")
        nmf_path = os.path.join("auto_interp_results", activations_name, "nmf_1gb.pkl")
        if os.path.exists(nmf_path):
            print("Loading NMF")
            nmf = pickle.load(open(nmf_path, "rb"))
        else:
            nmf = activation_NMF(activation_dataset, n_activations)
            os.makedirs(os.path.dirname(nmf_path), exist_ok=True)
            pickle.dump(nmf, open(nmf_path, "wb"))
        activation_fn_kwargs.update({"nmf": nmf})

    elif cfg.activation_transform == "feature_dict":
        print("Using feature dict activation transform")
        activation_fn_kwargs.update({"autoencoder": autoencoder})

    elif cfg.activation_transform == "random":
        print("Using random activation transform")
        random_path = os.path.join("auto_interp_results", activations_name, "random_dirs.pkl")
        if os.path.exists(random_path):
            print("Loading random directions")
            random_direction_matrix = pickle.load(open(random_path, "rb"))
        else:
            random_direction_matrix = torch.randn(activation_width, activation_width)
            os.makedirs(os.path.dirname(random_path), exist_ok=True)
            pickle.dump(random_direction_matrix, open(random_path, "wb"))
        
        activation_fn_kwargs.update({"random_matrix": random_direction_matrix})
    elif cfg.activation_transform == "random_bias":
        print("Using random activation transform")
        random_path = os.path.join("auto_interp_results", activations_name, "random_dirs.pkl")
        if os.path.exists(random_path):
            print("Loading random directions")
            random_direction_matrix = pickle.load(open(random_path, "rb"))
        else:
            random_direction_matrix = torch.randn(activation_width, activation_width)
            os.makedirs(os.path.dirname(random_path), exist_ok=True)
            pickle.dump(random_direction_matrix, open(random_path, "wb"))
        
        activation_fn_kwargs.update({"random_matrix": random_direction_matrix})
        activation_fn_kwargs.update({"bias": autoencoder.encoder[0].bias})

    else:
        raise ValueError(f"Activation transform {cfg.activation_transform} not recognised")

    if cfg.activation_transform == "feature_dict":
        transform_name = cfg.load_interpret_autoencoder.split("/")[-1][:-4]
    else:
        transform_name = cfg.activation_transform

    if cfg.interp_name:
        transform_folder = os.path.join("auto_interp_results", activations_name, cfg.interp_name)
    else:
        transform_folder = os.path.join("auto_interp_results", activations_name, transform_name)
    df_loc = os.path.join(transform_folder, f"activation_df.hdf")

    if not (cfg.load_activation_dataset and os.path.exists(df_loc)) or cfg.refresh_data:
        base_df = make_feature_activation_dataset(
            cfg.model_name,
            model,
            layer=cfg.layer,
            use_residual=cfg.use_residual,
            activation_fn_name=cfg.activation_transform,
            activation_fn_kwargs=activation_fn_kwargs,
            activation_dim=feature_size,
            device=cfg.device,
            use_baukit=use_baukit,
        )
        # save the dataset, saving each column separately so that we can retrive just the columns we want later
        print(f"Saving dataset to {df_loc}")
        os.makedirs(transform_folder, exist_ok=True)
        base_df.to_hdf(df_loc, key="df", mode="w")
    else:
        start_time = datetime.now()
        base_df = pd.read_hdf(df_loc)
        print(f"Loaded dataset in {datetime.now() - start_time}")


    # save the autoencoder being investigated
    os.makedirs(transform_folder, exist_ok=True)
    if cfg.activation_transform == "feature_dict":
        torch.save(autoencoder, os.path.join(transform_folder, "autoencoder.pt"))
        
    for feat_n in range(0, cfg.n_feats_explain):
        if os.path.exists(os.path.join(transform_folder, f"feature_{feat_n}")):
            print(f"Feature {feat_n} already exists, skipping")
            continue

        # logan_ae_top_mcs = [1910, 1597, 1991, 1672,  781,  239, 1375, 1048,  232, 1943,  885]
        # if feat_n not in logan_ae_top_mcs:
        #     continue

        # pyth2resid_top_mcs_list = [232, 1891, 1178, 1958,  389, 1461, 1248, 1582,  961, 1014,  465,
        #     1899,  406,  814,  744, 1644,  215, 1831,  965, 1722,  778, 1054,
        #     677, 1185, 1519,   17,  986,  113, 1087, 1237,  989,  831, 1477,
        #     1979,  647,  191, 1026, 2021, 1532, 1687, 1309,  435,  618,  442,
        #     1858, 1205, 1048, 1169,   25,  476, 1600,  720,  586, 1602,   69,
        #     1141, 1854,  261, 1622, 1101,  658,  836, 1774, 1679,  888,  534,
        #     1937, 1353,  622, 1863,  573, 1423, 1573,  200,  667,  104, 1252,
        #     1954, 1614, 1244,  826, 1724, 2006,  318,  552, 1574, 1253, 1415,
        #     979,  558, 1148, 1405,  768,  555, 1804, 1276, 1430, 2031, 1770,
        #     1758]
        # if feat_n not in pyth2resid_top_mcs_list:
        #     continue

        activation_col_names = [f"feature_{feat_n}_activation_{i}" for i in range(OPENAI_FRAGMENT_LEN)]
        read_fields = ["fragment_token_strs", f"feature_{feat_n}_mean", *activation_col_names]
        df = base_df[read_fields].copy()
        sorted_df = df.sort_values(by=f"feature_{feat_n}_mean", ascending=False)
        sorted_df = sorted_df.head(TOTAL_EXAMPLES)
        top_activation_records = []
        for i, row in sorted_df.iterrows():
            top_activation_records.append(ActivationRecord(row["fragment_token_strs"], [row[f"feature_{feat_n}_activation_{j}"] for j in range(OPENAI_FRAGMENT_LEN)]))
        
        random_activation_records: List[ActivationRecord] = []
        # Adding random fragments
        # random_df = df.sample(n=TOTAL_EXAMPLES)
        # for i, row in random_df.iterrows():
        #     random_activation_records.append(ActivationRecord(row["fragment_token_strs"], [row[f"feature_{feat_n}_activation_{j}"] for j in range(OPENAI_FRAGMENT_LEN)]))
        
        # making sure that the have some variation in each of the features, though need to be careful that this doesn't bias the results
        random_ordering = torch.randperm(len(df)).tolist()
        skip_feature = False
        while len(random_activation_records) < TOTAL_EXAMPLES:
            try:
                i = random_ordering.pop()
            except IndexError:  
                skip_feature = True
                break
            # if there are no activations for this fragment, skip it
            if df.iloc[i][f"feature_{feat_n}_mean"] == 0:
                continue
            random_activation_records.append(ActivationRecord(df.iloc[i]["fragment_token_strs"], [df.iloc[i][f"feature_{feat_n}_activation_{j}"] for j in range(OPENAI_FRAGMENT_LEN)]))
        if skip_feature:
            print(f"Skipping feature {feat_n} due to lack of activating examples")
            continue

        neuron_id = NeuronId(layer_index=2, neuron_index=feat_n)

        neuron_record = NeuronRecord(neuron_id=neuron_id, random_sample=random_activation_records, most_positive_activation_records=top_activation_records)
        slice_params = ActivationRecordSliceParams(n_examples_per_split=OPENAI_EXAMPLES_PER_SPLIT)
        train_activation_records = neuron_record.train_activation_records(slice_params)
        valid_activation_records = neuron_record.valid_activation_records(slice_params)

        explainer = TokenActivationPairExplainer(
            model_name=EXPLAINER_MODEL_NAME,
            prompt_format=PromptFormat.HARMONY_V4,
            max_concurrent=1,
        )
        explanations = await explainer.generate_explanations(
            all_activation_records=train_activation_records,
            max_activation=calculate_max_activation(train_activation_records),
            num_samples=1
        )
        assert len(explanations) == 1
        explanation = explanations[0]
        print(f"Feature {feat_n}, {explanation=}")

        # Simulate and score the explanation.
        format = PromptFormat.HARMONY_V4 if SIMULATOR_MODEL_NAME == "gpt-3.5-turbo" else PromptFormat.INSTRUCTION_FOLLOWING
        simulator = UncalibratedNeuronSimulator(
            ExplanationNeuronSimulator(
                SIMULATOR_MODEL_NAME,
                explanation,
                max_concurrent=1,
                prompt_format=format,
            )
        )
        scored_simulation = await simulate_and_score(simulator, valid_activation_records)

        score = scored_simulation.get_preferred_score()
        breakpoint()
        assert len(scored_simulation.scored_sequence_simulations) == 10
        top_only_score = aggregate_scored_sequence_simulations(scored_simulation.scored_sequence_simulations[:5]).get_preferred_score()
        random_only_score = aggregate_scored_sequence_simulations(scored_simulation.scored_sequence_simulations[5:]).get_preferred_score()
        print(f"Feature {feat_n}, score={score:.2f}, top_only_score={top_only_score:.2f}, random_only_score={random_only_score:.2f}")

        feature_name = f"feature_{feat_n}"
        feature_folder = os.path.join(transform_folder, feature_name)
        os.makedirs(feature_folder, exist_ok=True)
        pickle.dump(scored_simulation, open(os.path.join(feature_folder, "scored_simulation.pkl"), "wb"))
        pickle.dump(neuron_record, open(os.path.join(feature_folder, "neuron_record.pkl"), "wb"))
        # write a file with the explanation and the score
        with open(os.path.join(feature_folder, "explanation.txt"), "w") as f:
            f.write(f"{explanation}\nScore: {score:.2f}\nExplainer model: {EXPLAINER_MODEL_NAME}\nSimulator model: {SIMULATOR_MODEL_NAME}\n")
            f.write(f"Top only score: {top_only_score:.2f}\n")
            f.write(f"Random only score: {random_only_score:.2f}\n")
        
    
    if cfg.upload_to_aws:
        upload_to_aws(transform_folder)


async def run_openai_example():
    neuron_record = load_neuron(9, 10)

    # Grab the activation records we'll need.
    slice_params = ActivationRecordSliceParams(n_examples_per_split=5)
    train_activation_records = neuron_record.train_activation_records(
        activation_record_slice_params=slice_params
    )
    valid_activation_records = neuron_record.valid_activation_records(
        activation_record_slice_params=slice_params
    )

    # Generate an explanation for the neuron.
    explainer = TokenActivationPairExplainer(
        model_name=EXPLAINER_MODEL_NAME,
        prompt_format=PromptFormat.HARMONY_V4,
        max_concurrent=1,
    )
    explanations = await explainer.generate_explanations(
        all_activation_records=train_activation_records,
        max_activation=calculate_max_activation(train_activation_records),
        num_samples=1,
    )
    assert len(explanations) == 1
    explanation = explanations[0]
    print(f"{explanation=}")

    # Simulate and score the explanation.
    format = PromptFormat.HARMONY_V4 if SIMULATOR_MODEL_NAME == "gpt-3.5-turbo" else PromptFormat.INSTRUCTION_FOLLOWING
    simulator = UncalibratedNeuronSimulator(
        ExplanationNeuronSimulator(
            SIMULATOR_MODEL_NAME,
            explanation,
            max_concurrent=1,
            prompt_format=format, # INSTRUCTIONFOLLIWING
        )
    )
    scored_simulation = await simulate_and_score(simulator, valid_activation_records)
    print(f"score={scored_simulation.get_preferred_score():.2f}")


def get_score(lines: List[str], mode: str):
    if mode == "top":
        return float(lines[-3].split(" ")[-1])
    elif mode == "random":
        return float(lines[-2].split(" ")[-1])
    elif mode == "both":
        score_line = [line for line in lines if "Score: " in line][0]
        return float(score_line.split(" ")[1])
    else:
        raise ValueError(f"Unknown mode: {mode}")

def read_results(cfg):
    point_name = "resid" if cfg.use_residual else "postnonlin"
    activations_name = f"{cfg.model_name.split('/')[-1]}_layer{cfg.layer}_{point_name}"
    results_folder = os.path.join("auto_interp_results", activations_name)
    transforms = os.listdir(results_folder)
    transforms = [transform for transform in transforms if os.path.isdir(os.path.join(results_folder, transform))]
    scores = {}
    for transform in transforms:
        scores[transform] = []
        # list all the features by looking for folders
        feat_folders = [x for x in os.listdir(os.path.join(results_folder, transform)) if x.startswith("feature_")]
        print(f"{transform=}, {len(feat_folders)=}")
        for feature_folder in feat_folders:
            feature_ndx = int(feature_folder.split("_")[1])
            folder = os.path.join(results_folder, transform, feature_folder)
            if not os.path.exists(folder):
                continue
            explanation_text = open(os.path.join(folder, "explanation.txt")).read()
            # score should be on the second line but if explanation had newlines could be on the third or below
            # score = float(explanation_text.split("\n")[1].split(" ")[1])
            lines = explanation_text.split("\n")
            score = get_score(lines, cfg.score_mode)

            print(f"{feature_ndx=}, {transform=}, {score=}")
            scores[transform].append(score)
                
    
    # plot the scores as a histogram
    colors = ["red", "blue", "green", "orange", "purple", "pink", "black", "yellow", "brown", "cyan", "magenta", "grey"]
    for i, transform in enumerate(transforms):
        plt.hist(scores[transform], bins=20, alpha=0.5, label=transform, color=colors[i])
        plt.axvline(x=np.mean(scores[transform]), linestyle="-", color=colors[i])
        # also want to plot confidence intervals
        sd = np.std(scores[transform])
        n = len(scores[transform])
        ci = 1.96 * sd / np.sqrt(n)
        plt.axvline(x=np.mean(scores[transform]) + ci, linestyle="--", color=colors[i])
        plt.axvline(x=np.mean(scores[transform]) - ci, linestyle="--", color=colors[i])
        print(f"{transform=}, mean={np.mean(scores[transform])}")

    plt.legend(loc='upper right')
    # plot means on that graph 

    # add title and axis labels
    plt.title(f"{cfg.model_name}. Layer {cfg.layer}. {point_name} auto-interp scores")
    plt.xlabel("GPT-4-based interpretability score")
    plt.ylabel("Feature Count")


    # save
    save_path = os.path.join(results_folder, f"{cfg.score_mode}_scores.png")
    print(f"Saving to {save_path}")
    plt.savefig(save_path)



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "openai":
        asyncio.run(run_openai_example())
    elif len(sys.argv) > 1 and sys.argv[1] == "read_results":
        # parse --layer and --model_name from command line using custom parser
        argparser = argparse.ArgumentParser()
        argparser.add_argument("--layer", type=int, required=True)
        argparser.add_argument("--model_name", type=str, required=True)
        argparser.add_argument("--use_residual", action="store_true")
        argparser.add_argument("--score_mode", type=str, default="top_random") # can be "top", "random", "top_random"
        cfg = argparser.parse_args(sys.argv[2:])
        read_results(cfg)
    else:
        default_cfg = parse_args()
        default_cfg.chunk_size_gb = 10
        asyncio.run(main(default_cfg))