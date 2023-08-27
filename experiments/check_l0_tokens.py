"""
Checking whether the directions learned by the sparse autoencoder in the residual stream at layer 0 are just the token embeddings.
"""

import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped")

embedding_matrix = model.W_E.detach().cpu()
unembed_matrix = model.W_U.detach().cpu().t()
print(embedding_matrix.shape)
print(unembed_matrix.shape)

# normalize embedding matrix and unembedding matrix
embedding_matrix = embedding_matrix / torch.norm(embedding_matrix, dim=1, keepdim=True)
unembed_matrix = unembed_matrix / torch.norm(unembed_matrix, dim=1, keepdim=True)

data = {}
for layer in range(6):
    comp_dict_locs = [
        f"/mnt/ssd-cluster/bigrun0308/tied_residual_l{layer}_r0/_9/learned_dicts.pt",
        f"/mnt/ssd-cluster/bigrun0308/tied_residual_l{layer}_r1/_9/learned_dicts.pt",
        f"/mnt/ssd-cluster/bigrun0308/tied_residual_l{layer}_r2/_9/learned_dicts.pt",
        f"/mnt/ssd-cluster/bigrun0308/tied_residual_l{layer}_r4/_9/learned_dicts.pt",
        f"/mnt/ssd-cluster/bigrun0308/tied_residual_l{layer}_r8/_9/learned_dicts.pt",
        f"/mnt/ssd-cluster/bigrun0308/tied_residual_l{layer}_r16/_9/learned_dicts.pt",
        f"/mnt/ssd-cluster/bigrun0308/tied_residual_l{layer}_r32/_9/learned_dicts.pt",
    ]

    dicts = [torch.load(loc) for loc in comp_dict_locs]
    from standard_metrics import mcs_to_fixed

    layer_data = []
    for d in dicts:
        learned_dict = d[7][0]
        embed_mcs = mcs_to_fixed(learned_dict, embedding_matrix)
        unembed_mcs = mcs_to_fixed(learned_dict, unembed_matrix)
        print(layer, "embed:", embed_mcs.mean(), "unembed:", unembed_mcs.mean())
        layer_data.append((embed_mcs.mean(), unembed_mcs.mean()))

    data[layer] = layer_data

# plot as two graphs
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for layer in range(6):
    embed, unembed = zip(*data[layer])
    ax[0].plot(embed, label=layer)
    ax[1].plot(unembed, label=layer)
ax[0].set_title("Embedding")
ax[1].set_title("Unembedding")
ax[0].legend()
ax[1].legend()
# set xticks to [0.5, 1, 2, 4, 8, 16, 32]
ax[0].set_xticks(range(7))
ax[0].set_xticklabels([0.5, 1, 2, 4, 8, 16, 32])
ax[1].set_xticks(range(7))
ax[1].set_xticklabels([0.5, 1, 2, 4, 8, 16, 32])

ax[0].set_xlabel("Dict ratio")
ax[1].set_xlabel("Dict ratio")
ax[0].set_ylabel("Mean cosine similarity")
ax[1].set_ylabel("Mean cosine similarity")

plt.savefig("embed_unembed.png")
