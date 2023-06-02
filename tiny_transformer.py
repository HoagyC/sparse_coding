import random

from baukit import Trace
from datasets import load_dataset
import torch
from transformers import GPT2Tokenizer

from nanoGPT_model import GPT, cfg

loaded_model = torch.load(open("models/17kckpt.pt", "rb"), map_location="cpu")

n_sentences = 10
sentence_list = []
dataset = load_dataset("NeelNanda/pile-10k")
while len(sentence_list) < n_sentences:
    sentence = dataset["train"][random.randint(0, len(dataset["train"]))]["text"]
    # Cut off after a maximum of 20 words
    sentence = " ".join(sentence.split(" ")[:20])
    # If it contains non-ascii characters, skip it
    if not all(ord(c) < 128 for c in sentence) or len(sentence) < 20:
        continue
    sentence_list.append(sentence)


# Encode the sentences with GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
encoded_sentences = tokenizer(sentence_list, padding=False, truncation=True, return_tensors="pt")

model = GPT(cfg)
model_dict = loaded_model["model"]
# need to cut the "_orig_mod." from the keys
model_dict = {k.replace("_orig_mod.", ""): v for k, v in model_dict.items()}
model.load_state_dict(model_dict)

inp = tokenizer("Hello, my dog is cute", return_tensors="pt")
with Trace(model, "transformer.h.5.attn.c_attn") as ret:
    _ = model(inp["input_ids"])
    representation = ret.output
representation[0].shape