import activation_dataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

import torch

import argparse

import os

def setup_data_new(
    model_name: str,
    dataset_name: str,
    dataset_folder: str,
    tensor_names: List[str],
    chunk_size: int,
    n_chunks: int,
    skip_chunks: int = 0,
    device: Optional[torch.device] = torch.device("cuda:0"),
    max_length: int = 2048,
    model_batch_size: int = 4,
    precision: Literal["float16", "float32"] = "float16",
    shuffle_seed: Optional[int] = None,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--n_chunks", type=int, default=1)
    parser.add_argument("--skip_chunks", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=4000000)
    parser.add_argument("--dataset", type=str, default="NeelNanda/pile-10k")
    parser.add_argument("--tensors", type=str, nargs="+", default=["layer.{layer}"])
    parser.add_argument("--layers", type=int, nargs="+", default=[2])
    parser.add_argument("--location", type=str, default="residual")
    parser.add_argument("--dataset_folder", type=str, default="activation_data")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    tensor_names = [name.fmt(layer=l) for name in args.tensors for layer in args.layers]

    os.makedirs(args.dataset_folder, exist_ok=True)
    for layer_folder in tensor_names:
        os.makedirs(os.path.join(args.dataset_folder, layer_folder), exist_ok=True)

    activation_dataset.setup_data_new(
        args.model,
        args.dataset,
        args.dataset_folder,
        tensor_names,
        args.chunk_size,
        args.n_chunks,
        skip_chunks=args.skip_chunks,
        device=args.device,
    )