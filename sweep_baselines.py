"""
Running and saving a suite of baseline models to compare against.
"""
import multiprocessing as mp
import os

import torch
import tqdm

from autoencoders.pca import BatchedPCA
from autoencoders.ica import ICAEncoder
from autoencoders.nmf import NMFEncoder

def run_layer_baselines(args) -> None:
    layer: int
    layer_locs: list[str]
    chunks_folder: str
    output_folder: str 
    sparsity: int
    device: torch.device
    layer, layer_locs, chunks_folder, output_folder, sparsity, device = args

    for layer_loc in layer_locs:
        print(f"Layer {layer}, {layer_loc}")
        folder_name = f"l{layer}_{layer_loc}"
        os.makedirs(os.path.join(output_folder, folder_name), exist_ok=True)
        full_chunk_path = os.path.join(chunks_folder, folder_name, "0.pt")
        full_chunk = torch.load(full_chunk_path, map_location=device)
        activation_dim = full_chunk.shape[1]

        # Run batched PCA on the layer
        pca = BatchedPCA(n_dims=activation_dim, device=device)
        print("Training PCA")
        pca_batch_size = 500
        with torch.no_grad():
            for i in tqdm.tqdm(range(0, len(full_chunk), pca_batch_size)):
                j = min(i + pca_batch_size, len(full_chunk))
                batch = full_chunk[i:j]
                pca.train_batch(batch)
        
        pca_ld = pca.to_learned_dict(sparsity=activation_dim) # No sparsity, use topK for that
        torch.save(pca_ld, os.path.join(output_folder, folder_name, "pca.pt"))

        pca_top_k = pca.to_topk_dict(sparsity)
        torch.save(pca_top_k, os.path.join(output_folder, folder_name, "pca_topk.pt"))

        # Run ICA
        ica = ICAEncoder(activation_size=activation_dim)
        print("Training ICA")
        ica.train(full_chunk)
        torch.save(ica, os.path.join(output_folder, folder_name, "ica.pt"))

        ica_top_k = ica.to_topk_dict(sparsity)
        torch.save(ica_top_k, os.path.join(output_folder, folder_name, "ica_topk.pt"))

        # Run NMF
        nmf = NMFEncoder(activation_size=activation_dim)
        print("Training NMF")
        nmf.train(full_chunk)
        torch.save(nmf, os.path.join(output_folder, folder_name, "nmf.pt"))

        nmf_top_k = nmf.to_topk_dict(sparsity)
        torch.save(nmf_top_k, os.path.join(output_folder, folder_name, "nmf_topk.pt"))

def run_all() -> None:
    chunks_folder = "/mnt/ssd-cluster/single_chunks"
    output_folder = "/mnt/ssd-cluster/baselines"
    os.makedirs(output_folder, exist_ok=True)

    sparsity = 50

    layers = list(range(6))

    layer_locs = ["mlp", "residual"]
    devices = [f"cuda:{i}" for i in [1,2,3,4,6,7]]
    args_list = [(layer, layer_locs, chunks_folder, output_folder, sparsity, devices[i]) for i, layer in enumerate(layers)]

    with mp.Pool(processes=len(layers)) as pool:        
        pool.map(run_layer_baselines, args_list)


if __name__ == "__main__":
    run_all()
        



        





    

if __name__ == "__main__":
    run_all()