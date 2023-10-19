import os
import shutil

import torch
import tqdm
from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer

import standard_metrics
from activation_dataset import setup_data
from argparser import parse_args
from autoencoders.learned_dict import LearnedDict

TARGET_SPARSITY = 180
DICT_RATIO = 4
BASE_FOLDER = "~/sparse_coding_aidan"

device = torch.device("cuda:0")

layer_filenames = [
    f"/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l{layer}_r4/_9/learned_dicts.pt" for layer in range(6)
]
dicts = [torch.load(filename) for filename in layer_filenames]

all_scores = []

cfg = parse_args()
cfg.model_name = "EleutherAI/pythia-70m-deduped"


def get_model(cfg):
    if cfg.model_name in [
        "gpt2",
        "EleutherAI/pythia-70m-deduped",
        "EleutherAI/pythia-160m-deduped",
    ]:
        model = HookedTransformer.from_pretrained(cfg.model_name, device=cfg.device)
    else:
        raise ValueError("Model name not recognised")

    if hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer
    else:
        print("Using default tokenizer from gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    return model, tokenizer


def init_model_dataset(cfg):
    if cfg.use_residual:
        if cfg.model_name == "EleutherAI/pythia-160m-deduped":
            cfg.activation_width = 768
        else:
            cfg.activation_width = 512
    else:
        cfg.activation_width = 2048  # mlp_width is 4x the residual width

    if len(os.listdir(cfg.dataset_folder)) == 0:
        print(f"Activations in {cfg.dataset_folder} do not exist, creating them")
        transformer, tokenizer = get_model(cfg)
        setup_data(cfg, tokenizer, transformer)
        del transformer, tokenizer
    else:
        print(f"Activations in {cfg.dataset_folder} already exist, loading them")


def init_test_data(cfg, n_layers):
    os.makedirs(os.path.join(BASE_FOLDER, "activation_data_layers"), exist_ok=True)

    for layer in range(6):
        cfg.layer = layer
        cfg.use_residual = True
        cfg.n_chunks = 1
        cfg.dataset_folder = os.path.join(BASE_FOLDER, "activation_data_layers", f"layer_{layer}")

        os.makedirs(cfg.dataset_folder, exist_ok=True)

        init_model_dataset(cfg)


init_test_data(cfg, 6)

for layer, l_dicts in tqdm.tqdm(enumerate(dicts)):
    best_score = None
    best_idx = -1
    scores = []

    chunk = torch.load(os.path.join(BASE_FOLDER, "activation_data", f"layer_{layer}", "0.pt")).to(device, dtype=torch.float32)[
        :100000
    ]

    for idx, (l_dict, hyperparams) in enumerate(l_dicts):
        l_dict.to_device(device)
        sparsity = standard_metrics.mean_nonzero_activations(l_dict, chunk).sum().item()
        fvu = standard_metrics.fraction_variance_unexplained(l_dict, chunk).item()
        scores.append((hyperparams["l1_alpha"], sparsity, fvu))

        score = (sparsity - TARGET_SPARSITY) ** 2
        if best_score == -1 or score < best_score:
            best_score = score
            best_idx = idx

    all_scores.append((best_idx, scores))

best_dicts = [dicts[layer][idx][1] for layer, (idx, _) in enumerate(all_scores)]

torch.save(best_dicts, "best_dicts.pt")

import matplotlib.pyplot as plt

colors = ["blue", "green", "orange", "purple", "brown", "pink"]

for layer, (best_idx, scores) in enumerate(all_scores):
    plt.scatter(
        [score[1] for score in scores],
        [score[2] for score in scores],
        label=f"Layer {layer}",
        color=colors[layer],
    )
    plt.plot([scores[best_idx][1]], [scores[best_idx][2]], marker="x", color="red")
plt.legend()
plt.xlabel("Sparsity")
plt.ylabel("FVU")
plt.savefig(os.path.join(BASE_FOLDER, "sparsity_fvu.png"))
plt.close()
