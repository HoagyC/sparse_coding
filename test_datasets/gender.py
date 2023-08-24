import pickle
import torch
import random

from transformers import AutoTokenizer

COUNT_CUTOFF = 100000

# max_name_toks = 3

#prompt = "{name}'s gender is {gender}"
prompt = "My name is{name} and my pronouns are"
prompt_len = 8
prompt_skip = 3
#prompt_len = 4

codes_map = {"M": 0, "F": 1}
answer_map = {"M": " he", "F": " she"}

def generate_gender_dataset(
    tokenizer_name,
    pad_token_id=0,
    count_cutoff=COUNT_CUTOFF,
    sample_n=None,
    randomise=True,
):
    if sample_n is not None:
        targets = {"M": sample_n, "F": sample_n}
        counts = {"M": 0, "F": 0}

    with open("gender_dataset.pkl", "rb") as f:
        max_name_toks, entries = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token_id = pad_token_id

    entries = [entry for entry in entries if int(entry[2]) > count_cutoff]

    if randomise:
        random.shuffle(entries)

    print(f"Found {len(entries)} entries")

    tokens = {codes_map[g]: tokenizer(a)["input_ids"][0] for g, a in answer_map.items()}

    prompts = []
    classes = []
    sequence_lengths = []

    for entry in entries:
        name, code = entry[0], entry[1]
        
        if sample_n is not None:
            if counts[code] >= targets[code]:
                continue
            counts[code] += 1

        t = tokenizer(
            prompt.format(name=" "+name),
            padding="do_not_pad",
        )

        seq = t["input_ids"]
        seq_len = len(seq)

        sequence_lengths.append(seq_len)
        prompts.append(seq + [pad_token_id]*(prompt_len+max_name_toks-seq_len))
        classes.append(codes_map[code])

    if sample_n is not None:
        assert all([c == sample_n for c in counts.values()]), "Not enough samples for each class"

    prompts = torch.tensor(prompts)
    classes = torch.tensor(classes)
    sequence_lengths = torch.tensor(sequence_lengths)

    return prompts, classes, tokens, sequence_lengths, prompt_skip