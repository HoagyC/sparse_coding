import asyncio
import importlib
import json
import os
import pickle
import requests
import sys
from typing import Any, Dict, Union, List

from baukit import Trace
from datasets import load_dataset, ReadInstruction
from einops import rearrange
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer

from argparser import parse_args
from utils import dotdict, make_tensor_name
from nanoGPT_model import GPT
from run import AutoEncoder


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

EXPLAINER_MODEL_NAME = "gpt-3.5-turbo"
SIMULATOR_MODEL_NAME = "text-davinci-003"

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


def make_feature_activation_dataset(cfg, model: HookedTransformer, feature_dict: torch.Tensor, use_baukit: bool = False):
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
            feature_activation_data = torch.matmul(mlp_activation_data, feature_dict.to(torch.float32))

            # Get average activation for each feature
            assert feature_activation_data.shape == (OPENAI_FRAGMENT_LEN, feature_dict.shape[1])
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

            for j in range(feature_dict.shape[1]):
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


async def interpret_neuron(neuron_record, n_examples_per_split: int= 5):
    slice_params = ActivationRecordSliceParams(n_examples_per_split=n_examples_per_split)
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
    simulator = UncalibratedNeuronSimulator(
        ExplanationNeuronSimulator(
            SIMULATOR_MODEL_NAME,
            explanation,
            max_concurrent=1,
            prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
        )
    )
    scored_simulation = await simulate_and_score(simulator, valid_activation_records)
    return scored_simulation


async def main(cfg: dotdict):
    # Load model
    if cfg.model_name in ["gpt2", "EleutherAI/pythia-70m-deduped"]:
        model = HookedTransformer.from_pretrained(cfg.model_name, device=cfg.device)
        use_baukit = False
    elif cfg.model_name == "nanoGPT":
        model_dict = torch.load(open(cfg.model_path, "rb"), map_location="cpu")["model"]
        model_dict = {k.replace("_orig_mod.", ""): v for k, v in model_dict.items()}
        cfg_loc = cfg.model_path[:-3] + "cfg"  # cfg loc is same as model_loc but with .pt replaced with cfg.py
        cfg_loc = cfg_loc.replace("/", ".")
        model_cfg = importlib.import_module(cfg_loc).model_cfg
        model = GPT(model_cfg).to(cfg.device)
        model.load_state_dict(model_dict)
        use_baukit = True
    else:
        raise ValueError("Model name not recognised")
    
    # Load feature dict
    assert cfg.load_autoencoders is not None
    loaded_autoencoders = pickle.load(open(cfg.load_autoencoders, "rb"))
    autoencoder = loaded_autoencoders[0][0]
    feature_dict = autoencoder.encoder[0].weight.detach().t().to(cfg.device)

    if not (cfg.load_activation_dataset and os.path.exists(cfg.load_activation_dataset)):
        base_df = make_feature_activation_dataset(cfg, model, feature_dict, use_baukit=use_baukit)
        # save the dataset, saving each column separately so that we can retrive just the columns we want later
        # base_df.to_csv(cfg.save_activation_dataset, index=False)
        layer = 2
        for i in range(0, cfg.n_feats_explain):
            print(f"Saving feature {i} of layer {layer}")
            base_df[["fragment_id", f"feature_{i}_mean", f"feature_{i}_activations"]].to_csv(cfg.save_activation_dataset.replace(".csv", f"_layer_{layer}_feature_{i}.csv"), index=False)
        base_df[["fragment_id", "fragment_token_strs", "fragment_token_ids"]].to_csv(cfg.save_activation_dataset, index=False)

    base_df = pd.read_csv(cfg.load_activation_dataset)
    for feature_num in range(0, cfg.n_feats_explain):
        # Load the dataset
        neuron_df = pd.read_csv(cfg.load_activation_dataset.replace(".csv", f"_layer_2_feature_{feature_num}.csv"))
        # need to convert the activations and list of tokens from strings to lists
        df = base_df.merge(neuron_df, on="fragment_id")
        df[f"feature_{feature_num}_activations"] = df[f"feature_{feature_num}_activations"].apply(lambda x: eval(x))
        df["fragment_token_strs"] = df["fragment_token_strs"].apply(lambda x: eval(x))
        df["fragment_token_ids"] = df["fragment_token_ids"].apply(lambda x: eval(x))

        sorted_df = df.sort_values(by=f"feature_{feature_num}_mean", ascending=False)
        sorted_df = sorted_df.head(TOTAL_EXAMPLES)
        top_activation_records = []
        for i, row in sorted_df.iterrows():
            top_activation_records.append(ActivationRecord(row["fragment_token_strs"], row[f"feature_{feature_num}_activations"]))
        
        # Adding random fragments
        random_df = df.sample(n=TOTAL_EXAMPLES)
        random_activation_records = []
        for i, row in random_df.iterrows():
            random_activation_records.append(ActivationRecord(row["fragment_token_strs"], row[f"feature_{feature_num}_activations"]))
        
        neuron_id = NeuronId(layer_index=2, neuron_index=feature_num)

        neuron_record = NeuronRecord(neuron_id=neuron_id, random_sample=random_activation_records, most_positive_activation_records=top_activation_records)
        breakpoint()

        scored_simulation = await interpret_neuron(neuron_record, OPENAI_EXAMPLES_PER_SPLIT)
        print(f"score={scored_simulation.get_preferred_score():.2f}")

async def run_openai_example():
    neuron_record = load_neuron(9, 10)
    breakpoint()

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
    simulator = UncalibratedNeuronSimulator(
        ExplanationNeuronSimulator(
            SIMULATOR_MODEL_NAME,
            explanation,
            max_concurrent=1,
            prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
        )
    )
    scored_simulation = await simulate_and_score(simulator, valid_activation_records)
    print(f"score={scored_simulation.get_preferred_score():.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "openai":
        asyncio.run(run_openai_example())

    else:
        cfg = parse_args()
        asyncio.run(main(cfg))