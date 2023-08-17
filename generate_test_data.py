import activation_dataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

import torch

import argparse

import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--n_chunks", type=int, default=1)
    parser.add_argument("--skip_chunks", type=int, default=0)
    parser.add_argument("--chunk_size_gb", type=float, default=2)
    parser.add_argument("--dataset", type=str, default="NeelNanda/pile-10k")
    parser.add_argument("--layers", type=int, nargs="+", default=[2])
    parser.add_argument("--location", type=str, default="residual")
    parser.add_argument("--dataset_folder", type=str, default="activation_data")
    parser.add_argument("--layer_folder_fmt", type=str, default="layer_{layer}")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

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
        device=args.device,
        skip_chunks=args.skip_chunks,
    )