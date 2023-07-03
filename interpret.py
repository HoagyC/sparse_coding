import asyncio
from datetime import datetime
import importlib
import json
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
from transformers import GPT2Tokenizer

from argparser import parse_args
from comparisons import NoCredentialsError
from utils import dotdict, make_tensor_name, upload_to_aws
from nanoGPT_model import GPT
from run import AutoEncoder, setup_data


# set OPENAI_API_KEY environment variable from secrets.json['openai_key']
# needs to be done before importing openai interp bits
with open("secrets.json") as f:
    secrets = json.load(f)
    os.environ["OPENAI_API_KEY"] = secrets["openai_key"]

from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.activations.activations import ActivationRecordSliceParams, ActivationRecord, NeuronRecord, NeuronId
from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
from neuron_explainer.explanations.explainer import TokenActivationPairExplainer
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.scoring import simulate_and_score
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator
from neuron_explainer.fast_dataclasses import loads

EXPLAINER_MODEL_NAME = "gpt-4" # "gpt-3.5-turbo"
SIMULATOR_MODEL_NAME = "text-davinci-003" # "text-davinci-003"

OPENAI_MAX_FRAGMENTS = 50000
OPENAI_FRAGMENT_LEN = 64
OPENAI_EXAMPLES_PER_SPLIT = 5
N_SPLITS = 4
TOTAL_EXAMPLES = OPENAI_EXAMPLES_PER_SPLIT * N_SPLITS

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

def make_activation_dataset(cfg, model):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    cfg.n_chunks = 1
    dataset_name = cfg.dataset_name.split("/")[-1] + "-" + cfg.model_name.split("/")[-1] + "-" + str(cfg.layer)
    cfg.dataset_folder = os.path.join(cfg.datasets_folder, dataset_name)
    if not os.path.exists(cfg.dataset_folder) or len(os.listdir(cfg.dataset_folder)) == 0:
        setup_data(cfg, tokenizer, model, use_baukit=True, chunk_size_gb=10)
    chunk_loc = os.path.join(cfg.dataset_folder, f"0.pkl")

    total_activation_size = 1024 * 1024 * 1024
    elem_size = 4
    activation_width = cfg.mlp_width
    n_activations = total_activation_size // (elem_size * activation_width)

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


def make_feature_activation_dataset(cfg, model: HookedTransformer, activation_transform: Callable[[torch.Tensor], Any], use_baukit: bool = False):
    """
    Takes a dict of features and returns the top k activations for each feature in pile10k
    """
    max_sentences = 10000
    sentence_dataset = load_dataset("openwebtext", split="train[:1%](pct1_dropremainder)")
    if max_sentences is not None and max_sentences < len(sentence_dataset):
        sentence_dataset = sentence_dataset.select(range(max_sentences))

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tensor_name = make_tensor_name(cfg)
    # make list of sentence, tokenization pairs

    print(f"Computing internals for all {len(sentence_dataset)} sentences")
    tokenizer_model = HookedTransformer.from_pretrained("gpt2", device=cfg.device) # as copilot likes to say, this is a hack

    # Make dataframe with columns for each feature, and rows for each sentence fragment
    # each row should also have the full sentence, the current tokens and the previous tokens

    sentence_fragment_dicts: List[Dict[str, Any]] = []
    n_thrown = 0
    for sentence_id, sentence in tqdm(enumerate(sentence_dataset)):
        # split the sentence into fragments
        tokens = tokenizer_model.to_tokens(sentence["text"], prepend_bos=False)
        num_fragments = tokens.shape[1] // OPENAI_FRAGMENT_LEN
        for fragment_id in range(num_fragments):
            start_idx = fragment_id * OPENAI_FRAGMENT_LEN
            end_idx = (fragment_id + 1) * OPENAI_FRAGMENT_LEN
            fragment_tokens = tokens[:, start_idx:end_idx]
            if use_baukit:
                with Trace(model, tensor_name) as ret:
                    _ = model(fragment_tokens)
                    mlp_activation_data = ret.output
                    mlp_activation_data = rearrange(mlp_activation_data, "b s n -> (b s) n").to(cfg.device)
                    mlp_activation_data = nn.functional.gelu(mlp_activation_data)
            else:
                _, cache = model.run_with_cache(fragment_tokens)
                mlp_activation_data = cache[tensor_name].to(cfg.device)  # NOTE: could do all layers at once, but currently just doing 1 layer
                mlp_activation_data = rearrange(mlp_activation_data, "b s n -> (b s) n")

            # Project the activations into the feature space
            feature_activation_data = activation_transform(mlp_activation_data.detach().cpu())
            if not isinstance(feature_activation_data, torch.Tensor):
                feature_activation_data = torch.tensor(feature_activation_data)

            # Get average activation for each feature
            feature_activation_means = torch.mean(feature_activation_data, dim=0)

            fragment_dict: Dict[str, Any] = {}
            fragment_dict["fragment_id"] = len(sentence_fragment_dicts)
            fragment_dict["fragment_token_ids"] = fragment_tokens[0].tolist()
            fragment_dict["fragment_token_strs"] = tokenizer_model.to_str_tokens(fragment_tokens[0])

            # if there are any question marks in the fragment, throw it away (caused by byte pair encoding)
            replacement_char = "�"
            if replacement_char in fragment_dict["fragment_token_strs"]:
                # throw away the fragment
                n_thrown += 1
                continue

            for j in range(feature_activation_means.shape[0]):
                fragment_dict[f"feature_{j}_mean"] = feature_activation_means[j].item()
                fragment_dict[f"feature_{j}_activations"] = feature_activation_data[:, j].tolist()
                assert len(fragment_dict[f"feature_{j}_activations"]) == len(fragment_dict["fragment_token_strs"])
                # TODO: figure out why gettnig those question marks in tokenization
                # just printing errors
            

            sentence_fragment_dicts.append(fragment_dict)
            if len(sentence_fragment_dicts) > OPENAI_MAX_FRAGMENTS:
                break
        if len(sentence_fragment_dicts) > OPENAI_MAX_FRAGMENTS:
            break

    df = pd.DataFrame(sentence_fragment_dicts)
    print(f"Threw away {n_thrown} fragments, made {len(df)} fragments")
    return df

async def main(cfg: dotdict) -> None:
    # Load model
    if cfg.model_name in ["gpt2", "EleutherAI/pythia-70m-deduped"]:
        model = HookedTransformer.from_pretrained(cfg.model_name, device=cfg.device)
        use_baukit = False
        if cfg.model_name == "gpt2":
            activation_width = 3072
        elif cfg.model_name == "EleutherAI/pythia-70m-deduped":
            activation_width = 2048
    elif cfg.model_name == "nanoGPT":
        model_dict = torch.load(open(cfg.model_path, "rb"), map_location="cpu")["model"]
        model_dict = {k.replace("_orig_mod.", ""): v for k, v in model_dict.items()}
        cfg_loc = cfg.model_path[:-3] + "cfg"  # cfg loc is same as model_loc but with .pt replaced with cfg.py
        cfg_loc = cfg_loc.replace("/", ".")
        model_cfg = importlib.import_module(cfg_loc).model_cfg
        model = GPT(model_cfg).to(cfg.device)
        model.load_state_dict(model_dict)
        use_baukit = True
        activation_width = 128
    else:
        raise ValueError("Model name not recognised")
    
    # Load feature dict
    if cfg.activation_transform == "feature_dict":
        assert cfg.load_interpret_autoencoder is not None
        autoencoder = pickle.load(open(cfg.load_interpret_autoencoder, "rb"))
        feature_dict = autoencoder.encoder[0].weight.detach().t()
    
    dataset, n_activations = make_activation_dataset(cfg, model)
    activations_name = f"{cfg.model_name.split('/')[-1]}_layer{cfg.layer}_postnonlin"

    if cfg.activation_transform == "neuron_basis":
        print("Using neuron basis activation transform")
        activation_transform = lambda x: x

    elif cfg.activation_transform == "ica":
        print("Using ICA activation transform")
        ica_path = os.path.join("auto_interp_results", activations_name, "ica_1gb.pkl")
        if os.path.exists(ica_path):
            print("Loading ICA")
            ica = pickle.load(open(ica_path, "rb"))
        else:
            ica = activation_ICA(dataset, n_activations)
            pickle.dump(ica, open(ica_path, "wb"))
        activation_transform = lambda x: torch.tensor(ica.transform(x))

    elif cfg.activation_transform == "pca":
        print("Using PCA activation transform")
        pca_path = os.path.join("auto_interp_results", activations_name, "pca_1gb.pkl")
        if os.path.exists(pca_path):
            print("Loading PCA")
            pca = pickle.load(open(pca_path, "rb"))
        else:
            pca = activation_PCA(dataset, n_activations)
            pickle.dump(pca, open(pca_path, "wb"))
        activation_transform = lambda x: torch.tensor(pca.transform(x))

    elif cfg.activation_transform == "nmf":
        print("Using NMF activation transform")
        nmf_path = os.path.join("auto_interp_results", activations_name, "nmf_1gb.pkl")
        if os.path.exists(nmf_path):
            print("Loading NMF")
            nmf = pickle.load(open(nmf_path, "rb"))
        else:
            nmf = activation_NMF(dataset, n_activations)
            pickle.dump(nmf, open(nmf_path, "wb"))
        activation_transform = lambda x: torch.tensor(nmf.transform(x))

    elif cfg.activation_transform == "feature_dict":
        print("Using feature dict activation transform")
        activation_transform = lambda x: torch.matmul(x.to(cfg.device), feature_dict.to(cfg.device)).to("cpu")

    elif cfg.activation_transform == "random":
        print("Using random activation transform")
        random_path = os.path.join("auto_interp_results", activations_name, "random_dirs.pkl")
        if os.path.exists(random_path):
            print("Loading random directions")
            random_direction_matrix = pickle.load(open(random_path, "rb"))
        else:
            random_direction_matrix = torch.randn(activation_width, activation_width)
            pickle.dump(random_direction_matrix, open(random_path, "wb"))
        activation_transform = lambda x: torch.matmul(x, random_direction_matrix)

    else:
        raise ValueError(f"Activation transform {cfg.activation_transform} not recognised")

    if cfg.activation_transform == "feature_dict":
        transform_name = cfg.load_interpret_autoencoder.split("/")[-1][:-4]
    else:
        transform_name = cfg.activation_transform

    transform_folder = os.path.join("auto_interp_results", activations_name, transform_name)
    df_loc = os.path.join(transform_folder, f"activation_df.pkl")

    if not (cfg.load_activation_dataset and os.path.exists(df_loc)):
        base_df = make_feature_activation_dataset(cfg, model, activation_transform, use_baukit=use_baukit)
        # save the dataset, saving each column separately so that we can retrive just the columns we want later
        print(f"Saving dataset to {df_loc}")
        os.makedirs(transform_folder, exist_ok=True)
        assert len(base_df.columns) % 2 == 1, "Number of columns in base_df should be odd"
        n_features_in_df = int((len(base_df.columns) - 3) / 2)
        # assert f"feature_{n_features_in_df - 1}_activations" in base_df.columns, "Wrong number of features in base_df"
        # for feat_n in range(0, n_features_in_df):
        #     print(f"Saving feature {feat_n} of layer {layer}")
        #     feat_df_loc = os.path.join(activations_folder, f"activation_df_{cfg.activation_transform}_feature_{feat_n}_layer_{layer}.csv")
        #     base_df[["fragment_id", f"feature_{feat_n}_mean", f"feature_{feat_n}_activations"]].to_csv(feat_df_loc, index=False)
        # base_df[["fragment_id", "fragment_token_strs", "fragment_token_ids"]].to_csv(df_loc, index=False)
        # for feat_n in range(0, n_features_in_df):
        #     print(f"Saving feature {feat_n}")
        #     base_df[f"feature_{feat_n}_activations"] = base_df[f"feature_{feat_n}_activations"].astype(str)
        
        pickle.dump(base_df, open(df_loc, "wb"))
    else:
        print(f"Loading dataset from {df_loc} (may take a while)")
        base_df = pickle.load(open(df_loc, "rb"))
    #     base_df["fragment_token_strs"] = base_df["fragment_token_strs"].apply(lambda x: eval(x))
    #     base_df["fragment_token_ids"] = base_df["fragment_token_ids"].apply(lambda x: eval(x))

    # for feat_n in range(0, cfg.n_feats_explain):
    #     base_df[f"feature_{feat_n}_activations"] = base_df[f"feature_{feat_n}_activations"].apply(lambda x: eval(x))

    # start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # save_folder = os.path.join("auto_interp_results", start_time)

    # save the autoencoder being investigated
    os.makedirs(transform_folder, exist_ok=True)
    if cfg.activation_transform == "feature_dict":
        torch.save(autoencoder, os.path.join(transform_folder, "autoencoder.pt"))
    for feat_n in range(0, cfg.n_feats_explain):
        if os.path.exists(os.path.join(transform_folder, f"feature_{feat_n}")):
            print(f"Feature {feat_n} already exists, skipping")
            continue
        df = base_df.copy()[["fragment_token_strs", f"feature_{feat_n}_activations", f"feature_{feat_n}_mean"]]
        sorted_df = df.sort_values(by=f"feature_{feat_n}_mean", ascending=False)
        sorted_df = sorted_df.head(TOTAL_EXAMPLES)
        top_activation_records = []
        for i, row in sorted_df.iterrows():
            top_activation_records.append(ActivationRecord(row["fragment_token_strs"], row[f"feature_{feat_n}_activations"]))
        
        # Adding random fragments
        random_df = df.sample(n=TOTAL_EXAMPLES)
        random_activation_records = []
        for i, row in random_df.iterrows():
            random_activation_records.append(ActivationRecord(row["fragment_token_strs"], row[f"feature_{feat_n}_activations"]))
        
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
                prompt_format=format,
            )
        )
        scored_simulation = await simulate_and_score(simulator, valid_activation_records)
        score = scored_simulation.get_preferred_score()
        print(f"score={score:.2f}")

        feature_name = f"feature_{feat_n}"
        feature_folder = os.path.join(transform_folder, feature_name)
        os.makedirs(feature_folder, exist_ok=True)
        pickle.dump(scored_simulation, open(os.path.join(feature_folder, "scored_simulation.pkl"), "wb"))
        pickle.dump(neuron_record, open(os.path.join(feature_folder, "neuron_record.pkl"), "wb"))
        # write a file with the explanation and the score
        with open(os.path.join(feature_folder, "explanation.txt"), "w") as f:
            f.write(f"{explanation}\nScore: {score:.2f}\nExplainer model: {EXPLAINER_MODEL_NAME}\nSimulator model: {SIMULATOR_MODEL_NAME}\n")
    
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

def read_results():
    results_folder = "auto_interp_results/pythia-70m-deduped_layer2_postnonlin"
    transforms = os.listdir(results_folder)
    transforms = [transform for transform in transforms if os.path.isdir(os.path.join(results_folder, transform))]
    scores = {}
    for transform in transforms:
        scores[transform] = []
        feature_ndx = 0
        while os.path.exists(os.path.join(results_folder, transform, f"feature_{feature_ndx}")):
            feature_folder = os.path.join(results_folder, transform, f"feature_{feature_ndx}")
            if not os.path.exists(feature_folder):
                continue
            explanation_text = open(os.path.join(feature_folder, "explanation.txt")).read()
            # score is on the second line
            score = float(explanation_text.split("\n")[1].split(" ")[1])
            print(f"{feature_ndx=}, {transform=}, {score=}")
            scores[transform].append(score)
            feature_ndx += 1
    
    # plot the scores as a histogram
    colors = ["red", "blue", "green", "orange", "purple", "pink"]
    for i, transform in enumerate(transforms):
        plt.hist(scores[transform], bins=20, alpha=0.5, label=transform, color=colors[i])
        plt.axvline(x=np.mean(scores[transform]), linestyle="--", color=colors[i])
        print(f"{transform=}, mean={np.mean(scores[transform])}")

    plt.legend(loc='upper right')
    # plot means on that graph 

    # add title and axis labels
    plt.title("Pythia 70M Deduped Layer 2 Postnonli Auto-Interp Scores")
    plt.xlabel("Score")
    plt.ylabel("Count")


    # save
    save_path = os.path.join(results_folder, "scores.png")
    print(f"Saving to {save_path}")
    plt.savefig(save_path)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "openai":
        asyncio.run(run_openai_example())
    elif len(sys.argv) > 1 and sys.argv[1] == "read_results":
        read_results()
    else:
        cfg = parse_args()
        asyncio.run(main(cfg))