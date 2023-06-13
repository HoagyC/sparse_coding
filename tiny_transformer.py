import random

from baukit import Trace
from datasets import load_dataset
import torch
from transformers import GPT2Tokenizer

from nanoGPT_model import GPT, cfg

loaded_model = torch.load(open("models/17kckpt.pt", "rb"), map_location="cpu")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = GPT(cfg)
model_dict = loaded_model["model"]
# need to cut the "_orig_mod." from the keys
model_dict = {k.replace("_orig_mod.", ""): v for k, v in model_dict.items()}
model.load_state_dict(model_dict)

inp = tokenizer("Hello, my dog is cute", return_tensors="pt")
with Trace(model, "transformer.h.5.attn.c_attn") as ret:
    _ = model(inp["input_ids"])
    representation = ret.output
representation[0].shape
