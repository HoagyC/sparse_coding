import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
import matplotlib.pyplot as plt

import torch

if __name__ == "__main__":
    scores = torch.load("kl_div_scores_layer_4.pt")

    fig, ax = plt.subplots()

    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

    for (key, score), color in zip(scores.items(), colors):
        kl_div, sparsity = zip(*score)
        ax.plot(kl_div, sparsity, label=key, color=color)
    
    ax.set_xlabel("KL Divergence")
    ax.set_ylabel("Sparsity")

    ax.legend()

    plt.savefig("graphs/sparsity_kl_div.png")