import torch
import transformer_lens
from transformer_lens import HookedTransformerConfig, HookedTransformer
import transformer_lens.utils as utils

loaded_model = torch.load(open("models/17kckpt.pt", "rb"), map_location="cpu")

# Example layer of model keys
# _orig_mod.transformer.h.5.ln_1.weight
# _orig_mod.transformer.h.5.attn.c_attn.weight
# _orig_mod.transformer.h.5.attn.c_proj.weight
# _orig_mod.transformer.h.5.ln_2.weight
# _orig_mod.transformer.h.5.mlp.c_fc.weight
# _orig_mod.transformer.h.5.mlp.c_proj.weight

# # Example layer of desired keys
# "blocks.2.ln1.w", # match is "_orig_mod.transformer.h.5.ln_1.weight"
# "blocks.2.ln1.b", # not used
# "blocks.2.ln2.w", # match is "_orig_mod.transformer.h.5.ln_2.weight"
# "blocks.2.ln2.b", # not used
# "blocks.2.attn.W_Q", # match is "_orig_mod.transformer.h.5.attn.c_attn.weight".split(self.n_embd)[0]
# "blocks.2.attn.W_K", # match is "_orig_mod.transformer.h.5.attn.c_attn.weight".split(self.n_embd)[1]
# "blocks.2.attn.W_V", # match is "_orig_mod.transformer.h.5.attn.c_attn.weight".split(self.n_embd)[2]
# "blocks.2.attn.W_O", # match is "_orig_mod.transformer.h.5.attn.c_proj.weight"
# "blocks.2.attn.b_Q", # not used 
# "blocks.2.attn.b_K", # not used
# "blocks.2.attn.b_V", # not used
# "blocks.2.attn.b_O", # not used
# "blocks.2.attn.mask", # ignore
# "blocks.2.attn.IGNORE", # ignore
# "blocks.2.mlp.W_in", # match is "_orig_mod.transformer.h.5.mlp.c_fc.weight"
# "blocks.2.mlp.b_in", # not used
# "blocks.2.mlp.W_out", # match is "_orig_mod.transformer.h.5.mlp.c_proj.weight"
# "blocks.2.mlp.b_out", # not used

# # Additional keys
# "embed.W_E" # match is "_orig_mod.transformer.wte.weight"
# "pos_embed.W_pos" # match is "_orig_mod.transformer.wpe.weight"
# "ln_final.w", # match is "_orig_mod.transformer.ln_f.weight"
# "ln_final.b", # not used
# "unembed.W_U", # match is "_orig_mod.lm_head.weight"
# "unembed.b_U" # not used

# Model parameters
model_dict = loaded_model["model"]
new_model_dict = {}
embed_dim=16
n_heads=8
dims_per_head=2
d_mlp=64
context_length=256
token_dict_size=50304
n_layers=6

# Create new model dict with matching keys
new_model_dict["embed.W_E"] = model_dict["_orig_mod.transformer.wte.weight"]
new_model_dict["pos_embed.W_pos"] = model_dict["_orig_mod.transformer.wpe.weight"]
new_model_dict["ln_final.w"] = model_dict["_orig_mod.transformer.ln_f.weight"]
new_model_dict["unembed.W_U"] = model_dict["_orig_mod.lm_head.weight"].T
for layer_n in range(n_layers):
    new_model_dict[f"blocks.{layer_n}.ln1.w"] = model_dict[f"_orig_mod.transformer.h.{layer_n}.ln_1.weight"]
    new_model_dict[f"blocks.{layer_n}.ln2.w"] = model_dict[f"_orig_mod.transformer.h.{layer_n}.ln_2.weight"]
    new_model_dict[f"blocks.{layer_n}.attn.W_Q"] = model_dict[f"_orig_mod.transformer.h.{layer_n}.attn.c_attn.weight"].split(16, dim=0)[0].view(n_heads, embed_dim, dims_per_head)
    new_model_dict[f"blocks.{layer_n}.attn.W_K"] = model_dict[f"_orig_mod.transformer.h.{layer_n}.attn.c_attn.weight"].split(16, dim=0)[1].view(n_heads, embed_dim, dims_per_head)
    new_model_dict[f"blocks.{layer_n}.attn.W_V"] = model_dict[f"_orig_mod.transformer.h.{layer_n}.attn.c_attn.weight"].split(16, dim=0)[2].view(n_heads, embed_dim, dims_per_head)
    new_model_dict[f"blocks.{layer_n}.attn.W_O"] = model_dict[f"_orig_mod.transformer.h.{layer_n}.attn.c_proj.weight"].view(n_heads, dims_per_head, embed_dim)
    new_model_dict[f"blocks.{layer_n}.mlp.W_in"] = model_dict[f"_orig_mod.transformer.h.{layer_n}.mlp.c_fc.weight"].T
    new_model_dict[f"blocks.{layer_n}.mlp.W_out"] = model_dict[f"_orig_mod.transformer.h.{layer_n}.mlp.c_proj.weight"].T
    new_model_dict[f"blocks.{layer_n}.attn.mask"] = torch.ones(context_length, context_length)
    new_model_dict[f"blocks.{layer_n}.attn.b_Q"] = torch.zeros(n_heads, dims_per_head)
    new_model_dict[f"blocks.{layer_n}.attn.b_K"] = torch.zeros(n_heads, dims_per_head)
    new_model_dict[f"blocks.{layer_n}.attn.b_V"] = torch.zeros(n_heads, dims_per_head)
    new_model_dict[f"blocks.{layer_n}.attn.b_O"] = torch.zeros(embed_dim)
    new_model_dict[f"blocks.{layer_n}.mlp.b_in"] = torch.zeros(d_mlp)
    new_model_dict[f"blocks.{layer_n}.mlp.b_out"] = torch.zeros(embed_dim)
    new_model_dict[f"blocks.{layer_n}.ln1.b"] = torch.zeros(embed_dim)
    new_model_dict[f"blocks.{layer_n}.ln2.b"] = torch.zeros(embed_dim)
    new_model_dict[f"blocks.{layer_n}.attn.IGNORE"] = torch.zeros(1)
new_model_dict["unembed.b_U"] = torch.zeros(token_dict_size)
new_model_dict["ln_final.b"] = torch.zeros(embed_dim)

# Load the new model
cfg = HookedTransformerConfig(
    n_layers=n_layers,
    d_model=embed_dim, 
    n_heads=n_heads,
    d_head=dims_per_head, # d_model/n_heads (bit silly to be 2?)
    d_mlp=d_mlp, # 4*d_model
    d_vocab=token_dict_size,
    n_ctx=context_length,
    act_fn="gelu",
    normalization_type="LN", # 'LN' (use LayerNorm, including weights & biases) and 'LNPre' (use LayerNorm, but no weights & biases), not 100% certain what the difference is
)

model = HookedTransformer(cfg)
model.load_state_dict(new_model_dict)

import random
from datasets import load_dataset
n=1
sentence_list = []
dataset = load_dataset("NeelNanda/pile-10k")
while len(sentence_list) < n:
    sentence = dataset["train"][random.randint(0, len(dataset["train"]))]["text"]
    # Cut off after a maximum of 20 words
    sentence = " ".join(sentence.split(" ")[:20])
    # If it contains non-ascii characters, skip it
    if not all(ord(c) < 128 for c in sentence) or len(sentence) < 20:
        continue
    sentence_list.append(sentence)


# Encode the sentences with GPT2 tokenizer
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
encoded_sentences = tokenizer(sentence_list, padding=False, truncation=True, return_tensors="pt")

# Run the model
output = model(encoded_sentences["input_ids"])

sentence_len = output.shape[1]
# For each token in the sentence, print the sentence up to that point, and the predicted next token
for i in range(sentence_len):
    partial_tokens = encoded_sentences["input_ids"][0][:i]
    partial_sentence = tokenizer.decode(partial_tokens)
    predicted_token = tokenizer.decode(output.argmax(dim=-1)[0][i])
    print(partial_sentence, " -- ", predicted_token)


# Now test auto-regression
# Start with just a BOS token
n_tokens = 40
temperature = 0.8
n_samples = 5
top_k = 200
for sample_ndx in range(n_samples):
    text = ""
    bos_id = 50256
    for i in range(n_tokens):
        tokens = tokenizer(text, return_tensors="pt")["input_ids"]
        # Add the BOS token
        tokens = torch.cat([torch.tensor([[bos_id]]), tokens], dim=-1).int()
        output = model(tokens)

        # Get the logits for the last token
        logits = output[0][-1]
        # Apply top-k filtering by setting all logits below the top-k to -inf
        logits[logits.argsort()[:-top_k]] = -float("inf")
        # Apply temperature
        logits = logits / temperature
        # Sample from the distribution
        new_token = torch.multinomial(logits.softmax(dim=-1), num_samples=1)

        new_text = tokenizer.decode(new_token)
        text += new_text
    
    print(repr(text))



