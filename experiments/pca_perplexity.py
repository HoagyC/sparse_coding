import itertools
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from datasets import Dataset, load_dataset
from transformer_lens import HookedTransformer

import standard_metrics
from autoencoders.learned_dict import AddedNoise, LearnedDict
from autoencoders.pca import BatchedPCA, PCAEncoder

BASE_FOLDER = "~/sparse_coding_aidan"


def train_pca(dataset):
    pca = BatchedPCA(dataset.shape[1], device)

    print("Training PCA")

    batch_size = 5000
    for i in tqdm.tqdm(range(0, len(dataset), batch_size)):
        j = min(i + batch_size, len(dataset))
        batch = dataset[i:j]
        pca.train_batch(batch)

    return pca


if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device="cuda:7")

    dataset_name = "NeelNanda/pile-10k"
    token_amount = 256
    token_dataset = (
        load_dataset(dataset_name, split="train")
        .map(
            lambda x: model.tokenizer(x["text"]),
            batched=True,
        )
        .filter(lambda x: len(x["input_ids"]) > token_amount)
        .map(lambda x: {"input_ids": x["input_ids"][:token_amount]})
    )

    N_SENTENCES = 256

    device = torch.device("cuda:7")

    tokens = torch.stack(
        [torch.tensor(token_dataset[i]["input_ids"], dtype=torch.long, device="cuda:7") for i in range(N_SENTENCES)],
        dim=0,
    )

    print(tokens.shape)

    dataset = torch.load(os.path.join(BASE_FOLDER, "activation_data_layers/layer_2/0.pt")).to(dtype=torch.float32, device=device)
    pca = train_pca(dataset)

    sample_idxs = np.random.choice(len(dataset), 10000, replace=False)
    sample = dataset[sample_idxs]

    dict_files = [
        # "output_topk/_27/learned_dicts.pt",
        "/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r0/_9/learned_dicts.pt",
        "/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r2/_9/learned_dicts.pt",
        "/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r8/_9/learned_dicts.pt",
        "/mnt/ssd-cluster/bigrun0308/output_hoagy_dense_sweep_tied_resid_l2_r16/_9/learned_dicts.pt",
        os.path.join(BASE_FOLDER, "output_topk/_39/learned_dicts.pt"),
    ]

    file_labels = [
        "Linear",
        "Linear",
        "Linear",
        "Linear",
        "TopK",
    ]

    d_activation = 512

    d_activation = 512

    learned_dict_sets: Dict[str, List[Tuple[LearnedDict, Dict[str, Any]]]] = {}
    for label, learned_dict_file in zip(file_labels, dict_files):
        learned_dicts = torch.load(learned_dict_file)
        dict_sizes = list(set([hyperparams["dict_size"] for _, hyperparams in learned_dicts]))
        for learned_dict, hyperparams in learned_dicts:
            name = label + " " + str(hyperparams["dict_size"])

            if name not in learned_dict_sets:
                learned_dict_sets[name] = []
            learned_dict_sets[name].append((learned_dict, hyperparams))

    # baselines
    learned_dict_sets["Added Noise"] = [
        (AddedNoise(mag, 512, device="cuda:7"), {"dict_size": 512}) for mag in np.linspace(0.0, 0.5, 32)
    ]
    learned_dict_sets["PCA (dynamic)"] = [
        (pca.to_learned_dict(k), {"dict_size": 512, "k": k}) for k in range(1, d_activation // 2, 8)
    ]
    learned_dict_sets["PCA (static)"] = [
        (pca.to_rotation_dict(n), {"dict_size": 512, "n": n}) for n in range(1, d_activation // 2, 8)
    ]

    scores: Dict[str, List[Tuple[float, float]]] = {}
    for label, learned_dict_set in learned_dict_sets.items():
        scores[label] = []
        for learned_dict, hyperparams in tqdm.tqdm(learned_dict_set):
            learned_dict.to_device(device)
            fvu = standard_metrics.fraction_variance_unexplained(learned_dict, sample).item()
            perplexity = []
            for i in range(0, tokens.shape[0], 16):
                j = min(i + 16, tokens.shape[0])
                with torch.no_grad():
                    perplexity.append(
                        standard_metrics.perplexity_under_reconstruction(model, learned_dict, (2, "residual"), tokens[i:j]).item()
                    )
            scores[label].append((fvu, np.mean(perplexity)))

    """
    scores["PCA (static)"] = []
    eigenvals, eigenvecs = pca.get_pca()

    # reverse order
    eigenvals = eigenvals.flip(dims=(0,))
    eigenvecs = eigenvecs.flip(dims=(0,))

    r_sq = torch.cumsum(eigenvals, dim=0) / torch.sum(eigenvals)
    fvus = 1 - r_sq

    for n_eig in tqdm.tqdm(range(1, eigenvecs.shape[0], 16)):
        def intervention(tensor, hook=None):
            compressed = torch.einsum("ij,blj->bli", eigenvecs[:n_eig], tensor)
            reconstructed = torch.einsum("ij,bli->blj", eigenvecs[:n_eig], compressed)
            return reconstructed
        
        perplexity = []
        no_intervention_loss = 0
        for i in range(0, tokens.shape[0], 16):
            j = min(i + 16, tokens.shape[0])
            with torch.no_grad():
                perplexity.append(model.run_with_hooks(
                    tokens[i:j],
                    fwd_hooks=[(
                        standard_metrics.get_model_tensor_name((2, "residual")),
                        intervention,
                    )],
                    return_type="loss"
                ).item())
        scores["PCA (static)"].append((fvus[n_eig].item(), np.mean(perplexity)))
    """

    colors = ["red", "blue", "green", "orange", "purple", "black"]
    markers = ["o", "x", "s", "v", "D", "P"]

    settings = itertools.product(markers, colors)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for (marker, color), (label, score) in zip(settings, scores.items()):
        x, y = zip(*score)
        ax.scatter(x, y, label=label, color=color, marker=marker)
    ax.legend()
    ax.set_ylabel("Loss")
    ax.set_xlabel("Fraction Variance Unexplained")
    plt.savefig(os.path.join(BASE_FOLDER, "pca_perplexity.png"))
