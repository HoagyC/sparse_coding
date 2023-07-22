"""
Functions to help work with Autoencoder models.
"""
import argparse
from datetime import datetime
import sys
import os
import pickle
import json

from baukit import Trace
from datasets import load_dataset
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from transformer_lens import HookedTransformer
import torch
import torch.nn as nn

from run import run_mmcs_with_larger
from utils import make_tensor_name

with open("secrets.json") as f:
    secrets = json.load(f)
    os.environ["OPENAI_API_KEY"] = secrets["openai_key"]

from neuron_explainer.explanations.explanations import ScoredSimulation
from neuron_explainer.explanations.scoring import aggregate_scored_sequence_simulations

def calculate_mcs_scores(dict_base_loc: str, dict_compare_loc: str, save_loc: str = "") -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dict_base = pickle.load(open(dict_base_loc, "rb"))
    dict_compare = pickle.load(open(dict_compare_loc, "rb"))
    learned_dicts = [[
        dict_base.decoder.weight.detach().cpu().data.t(), 
        dict_compare.decoder.weight.detach().cpu().data.t()
    ]]
    _, _, full_max_cosine_sim_for_histograms = run_mmcs_with_larger(learned_dicts, threshold=0.9, device=device)
    if not save_loc:
        time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_loc = f"outputs/mcs_scores_{time}.txt"
    with open(save_loc, "w") as f:
        f.write("\n".join([str(x) for x in full_max_cosine_sim_for_histograms.cpu().numpy()]))

def calculate_activation_levels(
        dict_loc: str, 
        layer: int = 1,
        use_residual: bool = False,
        model_name: str = "EleutherAI/pythia-70m-deduped",
        dataset_name: str = "NeelNanda/pile-10k", 
        save_loc: str = "",
        use_baukit: bool = False,
        max_batches: int = 20
        ) -> None:
    dataset = load_dataset(dataset_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dict_base = pickle.load(open(dict_loc, "rb"))

    # get activations
    n_batches = len(dataset["train"])
    feature_activations = np.zeros(dict_base.decoder.weight.shape[1])

    autoencoder = pickle.load(open(dict_loc, "rb"))

    tensor_name = make_tensor_name(layer, use_residual, model_name)
    model = HookedTransformer.from_pretrained(model_name)
    batch_ndx = 0
    for batch in dataset["train"]:
        tokens = model.to_tokens(batch["text"])
        if use_baukit:
            with Trace(model, tensor_name) as ret:
                _ = model(tokens)
                mlp_activation_data = ret.output.to(device)
                mlp_activation_data = nn.functional.gelu(mlp_activation_data)
        else:
            _, cache = model.run_with_cache(tokens)
            mlp_activation_data = cache[tensor_name].to(device)
        
        mlp_activation_data = mlp_activation_data.reshape(-1, mlp_activation_data.shape[-1])

        # get activations
        reconstruction, dict_activations = autoencoder(mlp_activation_data)
        # breakpoint()
        feature_activations += dict_activations.detach().cpu().numpy().sum(axis=0)
        print(f"completed batch {batch_ndx} of {n_batches}", end="\r")
        batch_ndx += 1
        if batch_ndx >= max_batches:
            break

    print(f"completed batch {batch_ndx} of {n_batches}")

    # normalize activations
    feature_activations /= n_batches

    if not save_loc:    
        time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_loc = f"outputs/activation_levels_{time}.txt"
    
    # write activations out as decimals with 7 decimal places
    with open(save_loc, "w") as f:
        f.write("\n".join([f"{x:.7f}" for x in feature_activations]))


def get_top_and_random(folder: str):
    """
    Take the scored_simulation.pkl and get out the top-only and random-only correlation scores.
    Takes in a folder like "auto_interp_results/pythia-70m-deduped_layer2_resid"
    """
    # List the transformation folders
    transformations = [x for x in os.listdir(folder) if os.path.isdir(os.path.join(folder, x))]
    
    for transform in transformations:
        # get the feature folders 
        features = [x for x in os.listdir(os.path.join(folder, transform)) if os.path.isdir(os.path.join(folder, transform, x))]
        for feature in features:
            assert "feature_" in feature
            # check if the output in explanation.txt already contains the top and random scores 
            if not os.path.exists(os.path.join(folder, transform, feature, "explanation.txt")):
                print(f"Could not find explanation.txt for {transform} {feature}, skipping")
                continue
            with open(os.path.join(folder, transform, feature, "explanation.txt"), "r") as f:
                lines = f.readlines()
            if "Top_only" in lines[-2] and "Random_only" in lines[-1]:
                print(f"Already found top and random scores for {transform} {feature}, skipping")
                continue
                
            # load the scored_simulation.pkl
            with open(os.path.join(folder, transform, feature, "scored_simulation.pkl"), "rb") as f:
                scored_simulation = pickle.load(f)

            assert len(scored_simulation.scored_sequence_simulations) == 10
            top_only_score = aggregate_scored_sequence_simulations(scored_simulation.scored_sequence_simulations[:5]).get_preferred_score()
            random_only_score = aggregate_scored_sequence_simulations(scored_simulation.scored_sequence_simulations[5:]).get_preferred_score()

            # append top and random scores to explanation.txt
            with open(os.path.join(folder, transform, feature, "explanation.txt"), "a") as f:
                f.write(f"Top only score: {top_only_score:.2f}\n")
                f.write(f"Random only score: {random_only_score:.2f}\n")

def cluster_vectors(autoencoder):
    # take the direction vectors and cluster them
    # get the direction vectors
    direction_vectors = autoencoder.decoder.weight.detach().cpu().numpy().T

    # first apply t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, random_state=0)
    assert direction_vectors.shape[0] >= direction_vectors.shape[1] # make sure that we've got indices the right way round
    direction_vectors_tsne = tsne.fit_transform(direction_vectors)

    # now we're going to cluster the direction vectors
    # first, we'll try k-means
    print("Clustering vectors using kmeans")
    kmeans = KMeans(n_clusters=1000, random_state=0).fit(direction_vectors_tsne)
    # now get the clusters which have the most points in them and get the ids of the points in those clusters
    cluster_ids, cluster_counts = np.unique(kmeans.labels_, return_counts=True)
    cluster_ids = cluster_ids[np.argsort(cluster_counts)[::-1]]
    cluster_counts = cluster_counts[np.argsort(cluster_counts)[::-1]]   
    # now get the ids of the points in the top 10 clusters
    top_cluster_ids = cluster_ids[:10]
    top_cluster_points = []
    for cluster_id in top_cluster_ids:
        top_cluster_points.append(np.where(kmeans.labels_ == cluster_id)[0])

    # save clusters as separate lines on a text file
    with open("outputs/top_clusters.txt", "w") as f:
        for cluster in top_cluster_points:
            f.write(f"{list(cluster)}\n")

    # now want to take a selection of points, and find the nearest neighbours to them
    # first, take a random selection of points
    # n_points = 10
    # random_points = np.random.choice(direction_vectors_tsne.shape[0], n_points, replace=False)
    # # now find the nearest neighbours to these points
    # nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(direction_vectors_tsne)


        


if __name__ == "__main__":
    if sys.argv[1] == "calc_mcs":
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--dict_base', type=str, help='Path to base dictionary')
        parser.add_argument('--dict_compare', type=str, help='Path to dictionary to compare to base')
        parser.add_argument('--save', type=str, default="", help='Path to save results')
        args = parser.parse_args(sys.argv[2:])
        calculate_mcs_scores(args.dict_base, args.dict_compare, save_loc=args.save)
    elif sys.argv[1] == "calc_act":
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--dict', type=str, help='Path to dictionary')
        parser.add_argument('--dataset', type=str, default="NeelNanda/pile-10k", help='Dataset to use')
        parser.add_argument('--save', type=str, default="", help='Path to save results')
        parser.add_argument('--layer', type=int, default=1, help='Layer to use')
        parser.add_argument('--use_residual', action='store_true', help='Use residual')
        parser.add_argument('--model_name', type=str, default="EleutherAI/pythia-70m-deduped", help='Model to use')
        parser.add_argument('--use_baukit', action='store_true', help='Use baukit')
        parser.add_argument("--tied_weights", action="store_true", help="Use tied weights for autoencoder")
        args = parser.parse_args(sys.argv[2:])

        if args.tied_weights:
            from autoencoders.tied_ae import AutoEncoder
        else:
            from run import AutoEncoder # type: ignore

        calculate_activation_levels(args.dict, dataset_name=args.dataset, save_loc=args.save)
    
    elif sys.argv[1] == "get_top_and_random":
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--folder', type=str, help='Path to folder')
        args = parser.parse_args(sys.argv[2:])
        get_top_and_random(args.folder)
    
    elif sys.argv[1] == "cluster":
        from autoencoders.tied_ae import AutoEncoder
        ae = pickle.load(open("saved_autoencoders/resid2pyth-2048.pkl", "rb"))
        cluster_vectors(ae)