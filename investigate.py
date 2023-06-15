"""
Looking at whether there are systematic differences between feature that have converged with larger dicts and those that have not.
"""
import pickle

import torch

from argparser import parse_args
from run import run_mmcs_with_larger, AutoEncoder

def main(cfg):
    # load in two autoencoders
    with open(cfg.load_autoencoders, "rb") as f:
        autoencoders = pickle.load(f)

    ae1, ae2 = autoencoders[0][2:4]
    aes = [[ae1, ae2]]
    learned_dicts = [[auto_e.decoder.weight.detach().cpu().data.t().to(torch.float32) for auto_e in l1] for l1 in aes]
    mmcs_with_larger, feats_above_threshold, full_max_cosine_sim_for_histograms = run_mmcs_with_larger(cfg, learned_dicts, threshold=cfg.threshold)

    # now we want to check what the sparsity of the features is, which we will do by looking at the entropy of the features
    # we will also look at the entropy of the features that are above the threshold and those that are below the threshold to compare

    # first we need to get the entropy of the features
    entropy = lambda x: -torch.sum(x * torch.log(x + 1e-8), dim=1)
    # get the entropy of the features for each of the learned dicts, first normalise the features
    entropy_for_ae = [[entropy(torch.nn.functional.normalize(learned_dict, dim=1).abs()) for learned_dict in l1] for l1 in learned_dicts]
    
    # now we check the correlation between entropy in the features and the mmcs

    entropy = entropy_for_ae[0][0]
    mmcs = full_max_cosine_sim_for_histograms[0][0]
    assert entropy.shape == mmcs.shape
    correlation = torch.corrcoef(torch.stack([entropy, torch.tensor(mmcs)]))[0, 1]
    print("correlation between entropy and mmcs:", correlation)


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)