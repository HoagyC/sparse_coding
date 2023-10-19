"""
Running and saving a suite of baseline models to compare against.
"""
import multiprocessing as mp
import os

import torch
import tqdm

from autoencoders.ica import ICAEncoder
from autoencoders.learned_dict import IdentityReLU, RandomDict
from autoencoders.nmf import NMFEncoder
from autoencoders.pca import BatchedPCA
from standard_metrics import mean_nonzero_activations

def run_ica(chunk, output_file):
    chunk = torch.load(chunk, map_location="cpu")

    activation_dim = chunk.shape[1]

    ica = ICAEncoder(activation_size=activation_dim)
    print("Training ICA")
    ica.train(chunk)

    torch.save(ica, output_file)

def run_layer_baselines(args) -> None:
    layer: int
    layer_locs: list[str]
    chunks_folder: str
    output_folder: str
    sparsity: int
    device: torch.device
    remake: bool = False
    layer, layer_locs, chunks_folder, output_folder, sparsity, device = args

    for layer_loc in layer_locs:
        print(f"Layer {layer}, {layer_loc}")
        folder_name = f"l{layer}_{layer_loc}"

        os.makedirs(os.path.join(output_folder, folder_name), exist_ok=True)
        full_chunk_path = os.path.join(chunks_folder, folder_name, "0.pt")
        full_chunk = torch.load(full_chunk_path, map_location=device)
        activation_dim = full_chunk.shape[1]

        # Load the learned dict with l1_alpha of 8e-4
        if layer_loc == "residual":
            learned_dicts = torch.load(f"/mnt/ssd-cluster/bigrun0308/tied_{layer_loc}_l{layer}_r1/_9/learned_dicts.pt")
            l1_vals = [hparams["l1_alpha"] for _, hparams in learned_dicts]
            print("l1 vals", list(enumerate(l1_vals)))
            learned_dict = learned_dicts[7][0]  # 7th is 8.5e-4
            learned_dict.to_device(device)
            sparsity = mean_nonzero_activations(learned_dict, full_chunk.to(torch.float32)).sum().item()
            print(f"new sparsity for layer {layer}:", sparsity)

        if os.path.exists(os.path.join(output_folder, folder_name, "pca.pt")) and not remake:
            print("Skipping PCA")
        else:
            # Run batched PCA on the layer
            pca = BatchedPCA(n_dims=activation_dim, device=device)
            print("Training PCA")
            pca_batch_size = 500
            with torch.no_grad():
                for i in tqdm.tqdm(range(0, len(full_chunk), pca_batch_size)):
                    j = min(i + pca_batch_size, len(full_chunk))
                    batch = full_chunk[i:j]
                    pca.train_batch(batch)

            pca_ld = pca.to_learned_dict(sparsity=activation_dim)  # No sparsity, use topK for that
            torch.save(pca_ld, os.path.join(output_folder, folder_name, "pca.pt"))

            pca_top_k = pca.to_topk_dict(sparsity)
            torch.save(pca_top_k, os.path.join(output_folder, folder_name, "pca_topk.pt"))

        if os.path.exists(os.path.join(output_folder, folder_name, "ica.pt")) and not remake:
            print("Skipping ICA")
        else:
            # Run ICA
            ica = ICAEncoder(activation_size=activation_dim)
            print("Training ICA")
            ica.train(full_chunk)
            torch.save(ica, os.path.join(output_folder, folder_name, "ica.pt"))

            ica_top_k = ica.to_topk_dict(sparsity)
            torch.save(ica_top_k, os.path.join(output_folder, folder_name, "ica_topk.pt"))

        # if os.path.exists(os.path.join(output_folder, folder_name, "nmf.pt")) and not remake:
        #     print("Skipping NMF")
        # else:
        #     # Run NMF
        #     nmf = NMFEncoder(activation_size=activation_dim)
        #     print("Training NMF")
        #     nmf.train(full_chunk)
        #     torch.save(nmf, os.path.join(output_folder, folder_name, "nmf.pt"))

        #     nmf_top_k = nmf.to_topk_dict(sparsity)
        #     torch.save(nmf_top_k, os.path.join(output_folder, folder_name, "nmf_topk.pt"))

        if os.path.exists(os.path.join(output_folder, folder_name, "random.pt")) and not remake:
            print("Skipping random")
        else:
            # Run random dict
            random_dict = RandomDict(activation_size=activation_dim)
            torch.save(random_dict, os.path.join(output_folder, folder_name, "random.pt"))

        if os.path.exists(os.path.join(output_folder, folder_name, "identity_relu.pt")) and not remake:
            print("Skipping identity relu")
        else:
            # Run identity relu
            identity_relu = IdentityReLU(activation_size=activation_dim)
            torch.save(
                identity_relu,
                os.path.join(output_folder, folder_name, "identity_relu.pt"),
            )


def resave_change_sparsity() -> None:
    layer_loc = "residual"
    device = torch.device("cuda:0")
    chunks_folder = "/mnt/ssd-cluster/single_chunks"

    for layer in range(6):
        folder_name = f"l{layer}_{layer_loc}"
        full_chunk_path = os.path.join(chunks_folder, folder_name, "0.pt")
        full_chunk = torch.load(full_chunk_path, map_location=device)

        # Load the learned dict with l1_alpha of 8e-4
        learned_dicts = torch.load(f"/mnt/ssd-cluster/bigrun0308/tied_{layer_loc}_l{layer}_r1/_9/learned_dicts.pt")
        l1_vals = [hparams["l1_alpha"] for _, hparams in learned_dicts]
        print("l1 vals", list(enumerate(l1_vals)))
        learned_dict = learned_dicts[7][0]
        learned_dict.to_device(device)
        sparsity = int(mean_nonzero_activations(learned_dict, full_chunk.to(torch.float32)).sum().item())

        print("new sparsity", sparsity)

        # load ica and pca, and resave top_k with new sparsity
        ica = torch.load(f"/mnt/ssd-cluster/baselines/{folder_name}/ica.pt")
        ica_top_k = ica.to_topk_dict(sparsity)
        torch.save(ica_top_k, f"/mnt/ssd-cluster/baselines/{folder_name}/ica_topk.pt")

        activation_dim = full_chunk.shape[1]
        pca = BatchedPCA(n_dims=activation_dim, device=device)
        print("Training PCA")
        pca_batch_size = 500
        with torch.no_grad():
            for i in tqdm.tqdm(range(0, len(full_chunk), pca_batch_size)):
                j = min(i + pca_batch_size, len(full_chunk))
                batch = full_chunk[i:j]
                pca.train_batch(batch)
        pca_full = pca.to_learned_dict(sparsity=activation_dim)
        torch.save(pca_full, f"/mnt/ssd-cluster/baselines/{folder_name}/pca.pt")

        pca_top_k = pca.to_topk_dict(sparsity)
        torch.save(pca_top_k, f"/mnt/ssd-cluster/baselines/{folder_name}/pca_topk.pt")


def run_all() -> None:
    chunks_folder = "/mnt/ssd-cluster/single_chunks"
    output_folder = "/mnt/ssd-cluster/baselines"
    os.makedirs(output_folder, exist_ok=True)

    sparsity = 50

    layers = list(range(6))

    layer_locs = ["mlp"]
    devices = [f"cuda:{i}" for i in [1, 2, 3, 4, 6, 7]]
    args_list = [(layer, layer_locs, chunks_folder, output_folder, sparsity, devices[i]) for i, layer in enumerate(layers)]

    with mp.Pool(processes=len(layers)) as pool:
        pool.map(run_layer_baselines, args_list)

if __name__ == "__main__":
    run_ica("activation_data/layer_12/0.pt", "ica.pt")