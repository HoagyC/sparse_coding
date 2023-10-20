import csv
import os
import pickle
import sys

import tqdm
from transformers import AutoTokenizer

# dataset: https://archive.ics.uci.edu/dataset/591/gender+by+name

max_tok_len = 1
min_tok_len = 1
name_fmt = " {name}"

if __name__ == "__main__":
    model = "EleutherAI/pythia-70m-deduped"
    
    if len(sys.argv) > 1:
        min_tok_len = int(sys.argv[1])
    
    if len(sys.argv) > 2:
        max_tok_len = int(sys.argv[2])

    if len(sys.argv) > 3:
        model = sys.argv[3]
    
    tokenizer = AutoTokenizer.from_pretrained(model)

    entries = []

    with open("name_gender_dataset.csv", "r", newline="") as f:
        reader = csv.reader(f)
        reader.__next__()
        for entry in tqdm.tqdm(reader):
            name = entry[0]

            t = tokenizer(name_fmt.format(name=name))

            # filter names that are more than one token
            if min_tok_len <= len(t["input_ids"]) <= max_tok_len:
                entries.append(entry)

    print(f"Found {len(entries)} entries")

    with open("gender_dataset.pkl", "wb") as f:
        pickle.dump((max_tok_len, entries), f)
