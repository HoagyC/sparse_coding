import argparse
from dataclasses import dataclass
import os
from typing import List

import torch

import activation_dataset
from config import BaseArgs
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

@dataclass
class GenTestArgs(BaseArgs):
    model: str = "EleutherAI/pythia-70m-deduped"
    n_chunks: int = 1
    skip_chunks: int = 0
    chunk_size_gb: float = 2.0
    dataset: str = "NeelNanda/pile-10k"
    layers: List[int] = [2]
    location: str = "residual"
    dataset_folder: str = "activation_data"
    layer_folder_fmt: str = "layer_{layer}"
    device: str = "cuda:0"

if __name__ == "__main__":
    args = GenTestArgs()

    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    layer_folders = [args.layer_folder_fmt.format(layer=layer) for layer in args.layers]

    os.makedirs(args.dataset_folder, exist_ok=True)
    for layer_folder in layer_folders:
        os.makedirs(os.path.join(args.dataset_folder, layer_folder), exist_ok=True)
    
    dataset_folders = [os.path.join(args.dataset_folder, layer_folder) for layer_folder in layer_folders]
    activation_dataset.setup_data(
        tokenizer,
        model,
        args.dataset,
        dataset_folders,
        layer=args.layers,
        layer_loc=args.location,
        n_chunks=args.n_chunks,
        chunk_size_gb=args.chunk_size_gb,
        device=torch.device(args.device),
        skip_chunks=args.skip_chunks,
    )