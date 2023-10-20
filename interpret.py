import argparse
import asyncio
import copy
import importlib
import json
import multiprocessing as mp
import os
import pickle
import sys
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from baukit import Trace
from datasets import load_dataset
from transformer_lens import HookedTransformer

from activation_dataset import check_use_baukit, make_tensor_name
from config import BaseArgs, InterpArgs, InterpGraphArgs
from autoencoders.learned_dict import LearnedDict

# set OPENAI_API_KEY environment variable from secrets.json['openai_key']
# needs to be done before importing openai interp bits
with open("secrets.json") as f:
    secrets = json.load(f)
    os.environ["OPENAI_API_KEY"] = secrets["openai_key"]

mp.set_start_method("spawn", force=True)

from neuron_explainer.activations.activation_records import \
    calculate_max_activation
from neuron_explainer.activations.activations import (
    ActivationRecord, ActivationRecordSliceParams, NeuronId, NeuronRecord)
from neuron_explainer.explanations.calibrated_simulator import \
    UncalibratedNeuronSimulator
from neuron_explainer.explanations.explainer import \
    TokenActivationPairExplainer
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.scoring import (
    aggregate_scored_sequence_simulations, simulate_and_score)
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator
from neuron_explainer.fast_dataclasses import loads

EXPLAINER_MODEL_NAME = "gpt-4"  # "gpt-3.5-turbo"
SIMULATOR_MODEL_NAME = "text-davinci-003"

OPENAI_MAX_FRAGMENTS = 50000
OPENAI_FRAGMENT_LEN = 64
OPENAI_EXAMPLES_PER_SPLIT = 5
N_SPLITS = 4
TOTAL_EXAMPLES = OPENAI_EXAMPLES_PER_SPLIT * N_SPLITS
REPLACEMENT_CHAR = "�"
MAX_CONCURRENT: Any = None

BASE_FOLDER = "/mnt/ssd-cluster/sweep_interp"


# Replaces the load_neuron function in neuron_explainer.activations.activations because couldn't get blobfile to work
def load_neuron(
    layer_index: Union[str, int],
    neuron_index: Union[str, int],
    dataset_path: str = "https://openaipublic.blob.core.windows.net/neuron-explainer/data/collated-activations",
) -> NeuronRecord:
    """Load the NeuronRecord for the specified neuron from OpenAI's original work with GPT-2."""
    url = os.path.join(dataset_path, str(layer_index), f"{neuron_index}.json")
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Neuron record not found at {url}.")
    neuron_record = loads(response.content)

    if not isinstance(neuron_record, NeuronRecord):
        raise ValueError(f"Stored data incompatible with current version of NeuronRecord dataclass.")
    return neuron_record


def make_feature_activation_dataset(
    model: HookedTransformer,
    learned_dict: LearnedDict,
    layer: int,
    layer_loc: str,
    device: str = "cpu",
    n_fragments=OPENAI_MAX_FRAGMENTS,
    max_features: int = 0,  # number of features to store activations for, 0 for all
    random_fragment=True,  # used for debugging
):
    """
    Takes a specified point of a model, and a dataset.
    Returns a dataset which contains the activations of the model at that point,
    for each fragment in the dataset, transformed into the feature space
    """
    model.to(device)
    model.eval()
    learned_dict.to_device(device)

    use_baukit = check_use_baukit(model.cfg.model_name)

    if max_features:
        feat_dim = min(max_features, learned_dict.n_feats)
    else:
        feat_dim = learned_dict.n_feats

    sentence_dataset = load_dataset("openwebtext", split="train", streaming=True)

    if model.cfg.model_name == "nanoGPT":
        tokenizer_model = HookedTransformer.from_pretrained("gpt2", device=device)
    else:
        tokenizer_model = model

    tensor_name = make_tensor_name(layer, layer_loc, model.cfg.model_name)
    # make list of sentence, tokenization pairs

    iter_dataset = iter(sentence_dataset)

    # Make dataframe with columns for each feature, and rows for each sentence fragment
    # each row should also have the full sentence, the current tokens and the previous tokens

    n_thrown = 0
    n_added = 0
    batch_size = min(20, n_fragments)

    fragment_token_ids_list = []
    fragment_token_strs_list = []

    activation_maxes_table = np.zeros((n_fragments, feat_dim), dtype=np.float16)
    activation_data_table = np.zeros((n_fragments, feat_dim * OPENAI_FRAGMENT_LEN), dtype=np.float16)
    with torch.no_grad():
        while n_added < n_fragments:
            fragments: List[torch.Tensor] = []
            fragment_strs: List[str] = []
            while len(fragments) < batch_size:
                print(
                    f"Added {n_added} fragments, thrown {n_thrown} fragments\t\t\t\t\t\t",
                    end="\r",
                )
                sentence = next(iter_dataset)
                # split the sentence into fragments
                sentence_tokens = tokenizer_model.to_tokens(sentence["text"], prepend_bos=False).to(device)
                n_tokens = sentence_tokens.shape[1]
                # get a random fragment from the sentence - only taking one fragment per sentence so examples aren't correlated]
                if random_fragment:
                    token_start = np.random.randint(0, n_tokens - OPENAI_FRAGMENT_LEN)
                else:
                    token_start = 0
                fragment_tokens = sentence_tokens[:, token_start : token_start + OPENAI_FRAGMENT_LEN]
                token_strs = tokenizer_model.to_str_tokens(fragment_tokens[0])
                if REPLACEMENT_CHAR in token_strs:
                    n_thrown += 1
                    continue

                fragment_strs.append(token_strs)
                fragments.append(fragment_tokens)

            tokens = torch.cat(fragments, dim=0)
            assert tokens.shape == (batch_size, OPENAI_FRAGMENT_LEN), tokens.shape

            # breakpoint()
            if use_baukit:
                with Trace(model, tensor_name) as ret:
                    _ = model(tokens)
                    mlp_activation_data = ret.output.to(device)
                    mlp_activation_data = nn.functional.gelu(mlp_activation_data)
            else:
                _, cache = model.run_with_cache(tokens)
                mlp_activation_data = cache[tensor_name].to(device)

            for i in range(batch_size):
                fragment_tokens = tokens[i : i + 1, :]
                activation_data = mlp_activation_data[i : i + 1, :].squeeze(0)
                token_ids = fragment_tokens[0].tolist()

                feature_activation_data = learned_dict.encode(activation_data)
                feature_activation_maxes = torch.max(feature_activation_data, dim=0)[0]

                activation_maxes_table[n_added, :] = feature_activation_maxes.cpu().numpy()[:feat_dim]

                feature_activation_data = feature_activation_data.cpu().numpy()[:, :feat_dim]

                activation_data_table[n_added, :] = feature_activation_data.flatten()

                fragment_token_ids_list.append(token_ids)
                fragment_token_strs_list.append(fragment_strs[i])

                n_added += 1

                if n_added >= n_fragments:
                    break

    print(f"Added {n_added} fragments, thrown {n_thrown} fragments")
    # Now we build the dataframe from the numpy arrays and the lists
    print(f"Making dataframe from {n_added} fragments")
    df = pd.DataFrame()
    df["fragment_token_ids"] = fragment_token_ids_list
    df["fragment_token_strs"] = fragment_token_strs_list
    maxes_column_names = [f"feature_{i}_max" for i in range(feat_dim)]
    activations_column_names = [
        f"feature_{i}_activation_{j}" for j in range(OPENAI_FRAGMENT_LEN) for i in range(feat_dim)
    ]  # nested for loops are read left to right

    assert feature_activation_data.shape == (OPENAI_FRAGMENT_LEN, feat_dim)
    df = pd.concat([df, pd.DataFrame(activation_maxes_table, columns=maxes_column_names)], axis=1)
    df = pd.concat(
        [df, pd.DataFrame(activation_data_table, columns=activations_column_names)],
        axis=1,
    )
    print(f"Threw away {n_thrown} fragments, made {len(df)} fragments")
    return df


def get_df(
    feature_dict: LearnedDict,
    model_name: str,
    layer: int,
    layer_loc: str,
    n_feats: int,
    save_loc: str,
    device: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    # Load feature dict
    feature_dict.to_device(device)

    df_loc = os.path.join(save_loc, f"activation_df.hdf")

    reload_data = True
    if os.path.exists(df_loc) and not force_refresh:
        start_time = datetime.now()
        base_df = pd.read_hdf(df_loc)
        print(f"Loaded dataset in {datetime.now() - start_time}")

        # Check that the dataset has enough features saved
        if f"feature_{n_feats - 1}_activation_0" in base_df.keys():
            reload_data = False
        else:
            print("Dataset does not have enough features, remaking")

    if reload_data:
        model = HookedTransformer.from_pretrained(model_name, device=device)

        base_df = make_feature_activation_dataset(
            model,
            learned_dict=feature_dict,
            layer=layer,
            layer_loc=layer_loc,
            device=device,
            max_features=n_feats,
        )
        # save the dataset, saving each column separately so that we can retrive just the columns we want later
        print(f"Saving dataset to {df_loc}")
        os.makedirs(save_loc, exist_ok=True)
        base_df.to_hdf(df_loc, key="df", mode="w")

    # save the autoencoder being investigated
    os.makedirs(save_loc, exist_ok=True)
    torch.save(feature_dict, os.path.join(save_loc, "autoencoder.pt"))

    return base_df


async def interpret(base_df: pd.DataFrame, save_folder: str, n_feats_to_explain: int) -> None:
    for feat_n in range(0, n_feats_to_explain):
        if os.path.exists(os.path.join(save_folder, f"feature_{feat_n}")):
            print(f"Feature {feat_n} already exists, skipping")
            continue

        activation_col_names = [f"feature_{feat_n}_activation_{i}" for i in range(OPENAI_FRAGMENT_LEN)]
        read_fields = [
            "fragment_token_strs",
            f"feature_{feat_n}_max",
            *activation_col_names,
        ]
        # check that the dataset has the required columns
        if not all([field in base_df.columns for field in read_fields]):
            print(f"Dataset does not have all required columns for feature {feat_n}, skipping")
            continue
        df = base_df[read_fields].copy()
        sorted_df = df.sort_values(by=f"feature_{feat_n}_max", ascending=False)
        sorted_df = sorted_df.head(TOTAL_EXAMPLES)
        top_activation_records = []
        for i, row in sorted_df.iterrows():
            top_activation_records.append(
                ActivationRecord(
                    row["fragment_token_strs"],
                    [row[f"feature_{feat_n}_activation_{j}"] for j in range(OPENAI_FRAGMENT_LEN)],
                )
            )

        random_activation_records: List[ActivationRecord] = []
        # Adding random fragments
        # random_df = df.sample(n=TOTAL_EXAMPLES)
        # for i, row in random_df.iterrows():
        #     random_activation_records.append(ActivationRecord(row["fragment_token_strs"], [row[f"feature_{feat_n}_activation_{j}"] for j in range(OPENAI_FRAGMENT_LEN)]))

        # making sure that the have some variation in each of the features, though need to be careful that this doesn't bias the results
        random_ordering = torch.randperm(len(df)).tolist()
        skip_feature = False
        while len(random_activation_records) < TOTAL_EXAMPLES:
            try:
                i = random_ordering.pop()
            except IndexError:
                skip_feature = True
                break
            # if there are no activations for this fragment, skip it
            if df.iloc[i][f"feature_{feat_n}_max"] == 0:
                continue
            random_activation_records.append(
                ActivationRecord(
                    df.iloc[i]["fragment_token_strs"],
                    [df.iloc[i][f"feature_{feat_n}_activation_{j}"] for j in range(OPENAI_FRAGMENT_LEN)],
                )
            )
        if skip_feature:
            # Add placeholder folder so that we don't try to recompute this feature
            os.makedirs(os.path.join(save_folder, f"feature_{feat_n}"), exist_ok=True)
            print(f"Skipping feature {feat_n} due to lack of activating examples")
            continue

        neuron_id = NeuronId(layer_index=2, neuron_index=feat_n)

        neuron_record = NeuronRecord(
            neuron_id=neuron_id,
            random_sample=random_activation_records,
            most_positive_activation_records=top_activation_records,
        )
        slice_params = ActivationRecordSliceParams(n_examples_per_split=OPENAI_EXAMPLES_PER_SPLIT)
        train_activation_records = neuron_record.train_activation_records(slice_params)
        valid_activation_records = neuron_record.valid_activation_records(slice_params)

        explainer = TokenActivationPairExplainer(
            model_name=EXPLAINER_MODEL_NAME,
            prompt_format=PromptFormat.HARMONY_V4,
            max_concurrent=MAX_CONCURRENT,
        )
        explanations = await explainer.generate_explanations(
            all_activation_records=train_activation_records,
            max_activation=calculate_max_activation(train_activation_records),
            num_samples=1,
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
                max_concurrent=MAX_CONCURRENT,
                prompt_format=format,
            )
        )
        scored_simulation = await simulate_and_score(simulator, valid_activation_records)
        score = scored_simulation.get_preferred_score()
        assert len(scored_simulation.scored_sequence_simulations) == 10
        top_only_score = aggregate_scored_sequence_simulations(
            scored_simulation.scored_sequence_simulations[:5]
        ).get_preferred_score()
        random_only_score = aggregate_scored_sequence_simulations(
            scored_simulation.scored_sequence_simulations[5:]
        ).get_preferred_score()
        print(
            f"Feature {feat_n}, score={score:.2f}, top_only_score={top_only_score:.2f}, random_only_score={random_only_score:.2f}"
        )

        feature_name = f"feature_{feat_n}"
        feature_folder = os.path.join(save_folder, feature_name)
        os.makedirs(feature_folder, exist_ok=True)
        pickle.dump(
            scored_simulation,
            open(os.path.join(feature_folder, "scored_simulation.pkl"), "wb"),
        )
        pickle.dump(neuron_record, open(os.path.join(feature_folder, "neuron_record.pkl"), "wb"))
        # write a file with the explanation and the score
        with open(os.path.join(feature_folder, "explanation.txt"), "w") as f:
            f.write(
                f"{explanation}\nScore: {score:.2f}\nExplainer model: {EXPLAINER_MODEL_NAME}\nSimulator model: {SIMULATOR_MODEL_NAME}\n"
            )
            f.write(f"Top only score: {top_only_score:.2f}\n")
            f.write(f"Random only score: {random_only_score:.2f}\n")


def run(dict: LearnedDict, cfg: InterpArgs):
    assert cfg.df_n_feats >= cfg.n_feats_explain
    df = get_df(
        feature_dict=dict,
        model_name=cfg.model_name,
        layer=cfg.layer,
        layer_loc=cfg.layer_loc,
        n_feats=cfg.df_n_feats,
        save_loc=cfg.save_loc,
        device=cfg.device,
    )
    asyncio.run(interpret(df, cfg.save_loc, n_feats_to_explain=cfg.n_feats_explain))


def get_score(lines: List[str], mode: str):
    if mode == "top":
        return float(lines[-3].split(" ")[-1])
    elif mode == "random":
        return float(lines[-2].split(" ")[-1])
    elif mode == "top_random":
        score_line = [line for line in lines if "Score: " in line][0]
        return float(score_line.split(" ")[1])
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_folder(cfg: InterpArgs):
    base_folder = cfg.load_interpret_autoencoder
    all_encoders = os.listdir(cfg.load_interpret_autoencoder)
    all_encoders = [x for x in all_encoders if (x.endswith(".pt") or x.endswith(".pkl"))]
    print(f"Found {len(all_encoders)} encoders in {cfg.load_interpret_autoencoder}")
    for i, encoder in enumerate(all_encoders):
        print(f"Running encoder {i} of {len(all_encoders)}: {encoder}")
        learned_dict = torch.load(os.path.join(base_folder, encoder), map_location=torch.device(cfg.device))
        cfg.save_loc = os.path.join(BASE_FOLDER, encoder)
        run(learned_dict, cfg)


def make_tag_name(hparams: Dict) -> str:
    tag = ""
    if "tied" in hparams.keys():
        tag += f"tied_{hparams['tied']}"
    if "dict_size" in hparams.keys():
        tag += f"dict_size_{hparams['dict_size']}"
    if "l1_alpha" in hparams.keys():
        tag += f"l1_alpha_{hparams['l1_alpha']:.2}"
    if "bias_decay" in hparams.keys():
        tag += "0.0" if hparams["bias_decay"] == 0 else f"{hparams['bias_decay']:.1}"
    return tag


def run_from_grouped(cfg: InterpArgs, results_loc: str):
    """
    Run autointerpretation across a file of learned dicts as outputted by big_sweep.py or similar.
    Expects results_loc to a .pt file containing a list of tuples of (learned_dict, hparams_dict)
    """
    # First, read in the results file
    results = torch.load(results_loc)
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(os.path.join("auto_interp_results", time_str), exist_ok=True)
    # Now split the results out into separate files
    for learned_dict, hparams_dict in results:
        filename = make_tag_name(hparams_dict) + ".pt"
        torch.save(learned_dict, os.path.join("auto_interp_results", time_str, filename))

    cfg.load_interpret_autoencoder = os.path.join("auto_interp_results", time_str)
    run_folder(cfg)
    
def read_transform_scores(transform_loc: str, score_mode: str, verbose: bool = False) -> Tuple[List[int], List[float]]:
    transform_scores = []
    transform_ndxs = []
    # list all the features by looking for folders
    feat_folders = [x for x in os.listdir(transform_loc) if x.startswith("feature_")]
    if len(feat_folders) == 0:
        return [], []
    
    transform = transform_loc.split('/')[-1]
    print(f"{transform=} {len(feat_folders)=}")
    for feature_folder in feat_folders:
        feature_ndx = int(feature_folder.split("_")[1])
        folder = os.path.join(transform_loc, feature_folder)
        if not os.path.exists(folder):
            continue
        if not os.path.exists(os.path.join(folder, "explanation.txt")):
            continue
        explanation_text = open(os.path.join(folder, "explanation.txt")).read()
        # score should be on the second line but if explanation had newlines could be on the third or below
        # score = float(explanation_text.split("\n")[1].split(" ")[1])
        lines = explanation_text.split("\n")
        score = get_score(lines, score_mode)

        if verbose:
            print(f"{feature_ndx=}, {transform=}, {score=}")
        transform_scores.append(score)
        transform_ndxs.append(feature_ndx)
    
    return transform_ndxs, transform_scores


def read_scores(results_folder: str, score_mode: str = "top") -> Dict[str, Tuple[List[int], List[float]]]:
    assert score_mode in ["top", "random", "top_random"]
    scores: Dict[str, Tuple[List[int], List[float]]] = {}
    transforms = os.listdir(results_folder)
    transforms = [transform for transform in transforms if os.path.isdir(os.path.join(results_folder, transform))]
    if "sparse_coding" in transforms:
        transforms.remove("sparse_coding")
        transforms = ["sparse_coding"] + transforms

    for transform in transforms:
        transform_ndxs, transform_scores = read_transform_scores(os.path.join(results_folder, transform), score_mode)
        if len(transform_ndxs) > 0:
            scores[transform] = (transform_ndxs, transform_scores)

    return scores


def parse_folder_name(folder_name: str) -> Tuple[str, str, int, float, str]:
    """
    Parse the folder name to get the hparams
    """
    # examples: tied_mlpout_l1_r2, tied_residual_l5_r8
    tied, layer_loc, layer_str, ratio_str, *extras = folder_name.split("_")
    if extras:
        extra_str = "_".join(extras)
    else:
        extra_str = ""
    layer = int(layer_str[1:])
    ratio = float(ratio_str[1:])
    if ratio == 0:
        ratio = 0.5

    return tied, layer_loc, layer, ratio, extra_str


def run_list_of_learned_dicts(dicts: List[Tuple[str, LearnedDict]], cfg):
    """
    Run autointerpretation across a folder of learned dicts as outputted by big_sweep.py or similar, where the layer/layer_loc are the same.
    """
    for name, dict in dicts:
        print(f"Running {name}")
        run(dict, cfg)


def worker(queue, device_id):
    device = f"cuda:{device_id}"
    while not queue.empty():
        learned_dict, cfg = queue.get()
        print(f"Running {cfg.save_loc}")
        cfg.device = device
        learned_dict.to_device(device)
        run(learned_dict, cfg)


def interpret_across_baselines(n_gpus: int = 3):
    baselines_dir = "/mnt/ssd-cluster/baselines"
    save_dir = "/mnt/ssd-cluster/auto_interp_results/"
    os.makedirs(save_dir, exist_ok=True)
    base_cfg = InterpArgs()

    if n_gpus > 1:
        job_queue: mp.Queue = mp.Queue()

    all_folders = os.listdir(baselines_dir)
    for folder in all_folders:
        layer_str, layer_loc = folder.split("_")
        layer = int(layer_str[1:])
        layer_baselines = os.listdir(os.path.join(baselines_dir, folder))
        for baseline_file in layer_baselines:
            cfg = copy.deepcopy(base_cfg)
            cfg.layer = layer
            cfg.layer_loc = layer_loc
            cfg.save_loc = os.path.join(save_dir, folder, baseline_file[:-3])
            cfg.n_feats_explain = 150
            if not cfg.layer_loc == "residual":
                continue
            if "nmf" in baseline_file:
                continue
            learned_dict = torch.load(
                os.path.join(baselines_dir, folder, baseline_file),
                map_location=cfg.device,
            )
            print(f"{layer=}, {layer_loc=}, {baseline_file=}")
            if n_gpus == 1:
                run(learned_dict, cfg)
            else:
                job_queue.put((learned_dict, cfg))

    if n_gpus > 1:
        processes = [mp.Process(target=worker, args=(job_queue, i)) for i in range(n_gpus)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()


def interpret_across_big_sweep(l1_val: float, n_gpus: int = 1):
    base_cfg = InterpArgs()
    base_dir = "/mnt/ssd-cluster/bigrun0308"
    save_dir = "/mnt/ssd-cluster/auto_interp_results/"
    
    n_chunks_training = 10
    os.makedirs(save_dir, exist_ok=True)

    all_folders = os.listdir(base_dir)
    if n_gpus != 1:
        job_queue: List[Tuple[Callable, InterpArgs]] = []

    for folder in all_folders:
        try:
            tied, layer_loc, layer, ratio, extra_str = parse_folder_name(folder)
        except:
            continue
        print(f"{tied}, {layer_loc=}, {layer=}, {ratio=}")
        if layer_loc != "residual":
            continue
        if tied != "tied":
            continue
        if ratio != 2:
            continue
        if extra_str != "":
            continue

        cfg = copy.deepcopy(base_cfg)
        autoencoders = torch.load(
            os.path.join(base_dir, folder, f"_{n_chunks_training - 1}", "learned_dicts.pt"),
            map_location=cfg.device,
        )
        # find ae with matching l1_val
        matching_encoders = [ae for ae in autoencoders if abs(ae[1]["l1_alpha"] - l1_val) < 1e-4]
        if not len(matching_encoders) == 1:
            print(f"Found {len(matching_encoders)} matching encoders for {folder}")
        matching_encoder = matching_encoders[0][0]

        # save the learned dict
        save_str = f"l{layer}_{layer_loc}/{tied}_r{ratio}_l1a{l1_val:.2}"
        # os.makedirs(os.path.join(save_dir, save_str), exist_ok=True)
        # torch.save(matching_encoder, os.path.join(save_dir, save_str, "learned_dict.pt"))

        # run the interpretation
        cfg.load_interpret_autoencoder = os.path.join(save_dir, save_str, "learned_dict.pt")
        cfg.layer = layer
        cfg.layer_loc = layer_loc
        cfg.save_loc = os.path.join(save_dir, save_str)
        cfg.n_feats_explain = 150
        if n_gpus == 1:
            run(matching_encoder, cfg)
        else:
            cfg.device = f"cuda:{len(job_queue) % n_gpus}"
            job_queue.append((matching_encoder, cfg))

    if n_gpus > 1:
        with mp.Pool(n_gpus) as p:
            p.starmap(run, job_queue)


def interpret_across_chunks(l1_val: float, n_gpus: int = 1):
    base_cfg = InterpArgs()
    base_dir = "/mnt/ssd-cluster/longrun2408"
    save_dir = "/mnt/ssd-cluster/auto_interp_results_overtime/"
    os.makedirs(save_dir, exist_ok=True)

    all_folders = os.listdir(base_dir)
    if n_gpus != 1:
        job_queue: List[Tuple[Callable, InterpArgs]] = []

    for folder in all_folders:
        for n_chunks in [1, 4, 16, 32]:
            tied, layer_loc, layer, ratio, extra_str = parse_folder_name(folder)
            if layer != base_cfg.layer:
                continue
            cfg = copy.deepcopy(base_cfg)
            autoencoders = torch.load(
                os.path.join(base_dir, folder, f"_{n_chunks - 1}", "learned_dicts.pt"),
                map_location=cfg.device,
            )
            # find ae with matching l1_val
            matching_encoders = [ae for ae in autoencoders if abs(ae[1]["l1_alpha"] - l1_val) < 1e-4]
            if not len(matching_encoders) == 1:
                print(f"Found {len(matching_encoders)} matching encoders for {folder}")
            matching_encoder = matching_encoders[0][0]

            # save the learned dict
            save_str = f"l{layer}_{layer_loc}/{tied}_r{ratio}_nc{n_chunks}_l1a{l1_val:.2}"
            os.makedirs(os.path.join(save_dir, save_str), exist_ok=True)
            torch.save(matching_encoder, os.path.join(save_dir, save_str, "learned_dict.pt"))

            # run the interpretation
            cfg.load_interpret_autoencoder = os.path.join(save_dir, save_str, "learned_dict.pt")
            cfg.layer = layer
            cfg.layer_loc = layer_loc
            cfg.save_loc = os.path.join(save_dir, save_str)
            cfg.n_feats_explain = 100
            if n_gpus == 1:
                run(matching_encoder, cfg)
            else:
                cfg.device = f"cuda:{len(job_queue) % n_gpus}"
                job_queue.append((matching_encoder, cfg))

    if n_gpus > 1:
        with mp.Pool(n_gpus) as p:
            p.starmap(run, job_queue)


def read_results(activation_name: str, score_mode: str) -> None:
    results_folder = os.path.join("/mnt/ssd-cluster/auto_interp_results", activation_name)

    scores = read_scores(
        results_folder, score_mode
    )  # Dict[str, Tuple[List[int], List[float]]], where the tuple is (feature_ndxs, scores)
    if len(scores) == 0:
        print(f"No scores found for {activation_name}")
        return
    transforms = scores.keys()

    plt.clf()  # clear the plot

    # plot the scores as a violin plot
    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "pink",
        "black",
        "brown",
        "cyan",
        "magenta",
        "grey",
    ]

    # fix yrange from -0.2 to 0.6
    plt.ylim(-0.2, 0.6)
    # add horizontal grid lines every 0.1
    plt.yticks(np.arange(-0.2, 0.6, 0.1))
    plt.grid(axis="y", color="grey", linestyle="-", linewidth=0.5, alpha=0.3)
    # first we need to get the scores into a list of lists
    scores_list = [scores[transform][1] for transform in transforms]
    # remove any transforms that have no scores
    scores_list = [scores for scores in scores_list if len(scores) > 0]
    violin_parts = plt.violinplot(scores_list, showmeans=False, showextrema=False)
    for i, pc in enumerate(violin_parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_edgecolor(colors[i % len(colors)])
        pc.set_alpha(0.3)

    # add x labels
    plt.xticks(np.arange(1, len(transforms) + 1), transforms, rotation=90)

    # add standard errors around the means but don't plot the means
    cis = [1.96 * np.std(scores[transform][1], ddof=1) / np.sqrt(len(scores[transform][1])) for transform in transforms]
    for i, transform in enumerate(transforms):
        plt.errorbar(
            i + 1,
            np.mean(scores[transform][1]),
            yerr=cis[i],
            fmt="o",
            color=colors[i % len(colors)],
            elinewidth=2,
            capsize=20,
        )

    plt.title(f"{activation_name} {score_mode}")
    plt.xlabel("Transform")
    plt.ylabel("GPT-4-based interpretability score")
    plt.xticks(rotation=90)

    # and a thicker line at 0
    plt.axhline(y=0, linestyle="-", color="black", linewidth=1)

    plt.tight_layout()
    save_path = os.path.join(results_folder, f"{score_mode}_means_and_violin.png")
    print(f"Saving means and violin graph to {save_path}")
    plt.savefig(save_path)


if __name__ == "__main__":
    cfg: BaseArgs
    if len(sys.argv) > 1 and sys.argv[1] == "read_results":
        cfg = InterpGraphArgs()
        if cfg.score_mode == "all":
            score_modes = ["top", "random", "top_random"]
        else:
            score_modes = [cfg.score_mode]
 
        base_path = "/mnt/ssd-cluster/auto_interp_results"

        if cfg.run_all:
            activation_names = [x for x in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, x))]
        else:
            activation_names = [f"{cfg.model_name.split('/')[-1]}_layer{cfg.layer}_{cfg.layer_loc}"]

        for activation_name in activation_names:
            for score_mode in score_modes:
                read_results(activation_name, score_mode)

    elif len(sys.argv) > 1 and sys.argv[1] == "run_group":
        cfg = InterpArgs()
        run_from_grouped(cfg, cfg.load_interpret_autoencoder)

    elif len(sys.argv) > 1 and sys.argv[1] == "big_sweep":
        sys.argv.pop(1)
        # l1_val = 0.00018478
        l1_val = 0.0008577 # 8e-4 in logspace(-4, -2, 16)
        # l1_val = 0.00083768 # 8e-4 in logspace(-4, -2, 14)
        # l1_val = 0.0007197 # 8e-4 in logspace(-4, -2, 8)
        # l1_val = 1e-3
        # l1_val = 0.000316  # early one for mlp??
        interpret_across_big_sweep(l1_val)

    elif len(sys.argv) > 1 and sys.argv[1] == "all_baselines":
        sys.argv.pop(1)
        interpret_across_baselines()

    elif len(sys.argv) > 1 and sys.argv[1] == "chunks":
        l1_val = 0.0007197  # 8e-4 in logspace(-4, -2, 8)
        sys.argv.pop(1)
        interpret_across_chunks(l1_val)

    else:
        cfg = InterpArgs()
        if os.path.isdir(cfg.load_interpret_autoencoder):
            run_folder(cfg)
        else:
            learned_dict = torch.load(cfg.load_interpret_autoencoder, map_location=cfg.device)
            save_folder = f"/mnt/ssd-cluster/auto_interp_results/l{cfg.layer}_{cfg.layer_loc}"
            cfg.save_loc = os.path.join(save_folder, cfg.load_interpret_autoencoder)
            run(learned_dict, cfg)
