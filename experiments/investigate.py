"""
Looking at whether there are systematic differences between feature that have converged with larger dicts and those that have not.
"""
import os
import pickle
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import InvestigateArgs
from standard_metrics import run_mmcs_with_larger


def test_diversity_of_random_features():
    random_features = torch.randn(10000, 128)
    random_features = random_features / torch.norm(random_features, dim=1, keepdim=True)
    proportion = lambda x: (x.abs().t() / torch.sum(x.abs(), dim=1)).t()
    effective_number_of_neurons = lambda x: 1 / (proportion(x) ** 2).sum(dim=1)
    enn = effective_number_of_neurons(random_features)
    plt.hist(enn)
    print("mean:", enn.mean())
    plt.xlabel("Effective number of neurons")
    plt.ylabel("MMCS")
    plt.savefig("outputs/enn_vs_mmcs_randn.png")
    # plt.show()

    # now try with random directions, sampled from a normal distribution, not sampled from torch.randn
    gaussian = torch.distributions.normal.Normal(0, 1)
    random_features = gaussian.sample((10000, 128))
    random_features = random_features / torch.norm(random_features, dim=1, keepdim=True)
    enn_gaussian = effective_number_of_neurons(random_features)
    plt.hist(enn_gaussian)
    print("mean:", enn_gaussian.mean())
    plt.xlabel("Effective number of neurons")
    plt.ylabel("MMCS")
    plt.savefig("outputs/enn_vs_mmcs_gaussian.png")


def main(cfg):
    # load in two autoencoder
    autoencoders = torch.load("location_A.pt", map_location=torch.device("cpu"))

    print(len(autoencoders), len(autoencoders[0]))
    ae1, ae2 = autoencoders[0][2:4]
    aes = [[ae1, ae2]]
    learned_dicts = [[auto_e.decoder.weight.detach().cpu().data.t().to(torch.float32) for auto_e in l1] for l1 in aes]
    (
        mmcs_with_larger,
        feats_above_threshold,
        full_max_cosine_sim_for_histograms,
    ) = run_mmcs_with_larger(learned_dicts, threshold=cfg.threshold, device=cfg.device)

    # now we want to check what the sparsity of the features is, which we will do by looking at the entropy of the features
    # we will also look at the entropy of the features that are above the threshold and those that are below the threshold to compare

    # first we need to get the entropy of the features
    entropy = lambda x: -torch.sum(x * torch.log(x + 1e-8), dim=1)
    # get the entropy of the features for each of the learned dicts, first normalise the features
    entropy_for_ae = [
        [entropy(torch.nn.functional.normalize(learned_dict, dim=1).abs()) for learned_dict in l1] for l1 in learned_dicts
    ]

    # now we check the correlation between entropy in the features and the mmcs

    entropy = entropy_for_ae[0][0]
    mmcs = full_max_cosine_sim_for_histograms[0][0]
    assert entropy.shape == mmcs.shape
    correlation = torch.corrcoef(torch.stack([entropy, torch.tensor(mmcs)]))[0, 1]
    print("correlation between entropy and mmcs:", correlation)

    # other measure of the same concept is the effective number of neurons
    proportion = lambda x: (x.abs().t() / torch.sum(x.abs(), dim=1)).t()
    effective_number_of_neurons = lambda x: 1 / (proportion(x) ** 2).sum(dim=1)
    enn = [[effective_number_of_neurons(learned_dict) for learned_dict in l1] for l1 in learned_dicts]
    enn = enn[0][0]

    # make scatter plot of entropy and mmcs
    plt.scatter(entropy, mmcs)
    plt.xlabel("entropy")
    plt.ylabel("mmcs")
    plt.savefig("outputs/entropy_vs_mmcs.png")
    plt.close()

    # make scatter plot of enn and mmcs
    plt.scatter(enn, mmcs)
    plt.xlabel("Effective number of neurons")
    plt.ylabel("MMCS")
    plt.savefig("outputs/enn_vs_mmcs.png")
    # plt.show()

    # calculate mean entropy for features above and below threshold
    enn_above_threshold = enn[mmcs > cfg.threshold]
    enn_below_threshold = enn[mmcs < cfg.threshold]
    print("mean enn above threshold:", enn_above_threshold.mean())
    print("mean enn below threshold:", enn_below_threshold.mean())

    # now we look at the cosine similarity of the input vectors of the different neurons
    transformer_loc = "models/32d70k.pt"
    model = torch.load(open(transformer_loc, "rb"), map_location=torch.device("cpu"))
    model_dict = model["model"]
    in_matrix = model_dict[f"_orig_mod.transformer.h.{cfg.layer}.mlp.c_fc.weight"]


if __name__ == "__main__":
    cfg = InvestigateArgs()
    test_diversity_of_random_features()
