import torch
from transformer_lens import HookedTransformerConfig, HookedTransformer
import transformer_lens.utils as utils

# cfg = HookedTransformerConfig(
#     n_layers=6,
#     d_model=16, 
#     n_heads=8,
#     d_head=2, # d_model/n_heads (bit silly to be 2?)
#     d_mlp=64, # 4*d_model
#     d_vocab=50256,
#     n_ctx=256,
#     act_fn="gelu",
#     normalization_type="LN", # 'LN' (use LayerNorm, including weights & biases) and 'LNPre' (use LayerNorm, but no weights & biases), not 100% certain what the difference is
# )

# model = HookedTransformer(cfg)
# loaded_model = torch.load(open("models/ckpt.pt", "rb"), map_location="cpu")
# model.load_state_dict(loaded_model)

