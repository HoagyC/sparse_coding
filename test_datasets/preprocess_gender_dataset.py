import sys
import os
from transformers import AutoTokenizer
import csv
import pickle

import tqdm

# dataset: https://archive.ics.uci.edu/dataset/591/gender+by+name

if __name__ == "__main__":
    model = "EleutherAI/pythia-70m-deduped"
    if len(sys.argv) > 1:
        model = sys.argv[1]
    
    tokenizer = AutoTokenizer.from_pretrained(model)

    entries = []

    with open("name_gender_dataset.csv", "r", newline="") as f:
        reader = csv.reader(f)
        reader.__next__()
        for entry in tqdm.tqdm(reader):
            name = entry[0]

            t = tokenizer(name)
            
            # filter names that are more than one token
            if len(t["input_ids"]) == 1:
                entries.append(entry)

    print(f"Found {len(entries)} entries")

    with open("gender_dataset.pkl", "wb") as f:
        pickle.dump(entries, f)