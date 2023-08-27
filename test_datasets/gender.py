import pickle
import random

import torch
from transformers import AutoTokenizer

COUNT_CUTOFF = 100000

# max_name_toks = 3

# prompt = "{name}'s gender is {gender}"
prompt = "{name} uses the pronouns"
# prompt_len = 4

codes_map = {"M": 0, "F": 1}
answer_map = {"M": " he", "F": " she"}


def generate_gender_dataset(
    tokenizer_name,
    n_male,
    n_female,
    pad_token_id=0,
):
    with open("gender_dataset.pkl", "rb") as f:
        max_name_toks, entries = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="right")
    tokenizer.pad_token_id = pad_token_id

    prompt_len = len(tokenizer(prompt.format(name=""))["input_ids"])
    print("Prompt length:", prompt_len, "name length:", max_name_toks)

    entries = [entry for entry in entries if int(entry[2]) > COUNT_CUTOFF]
    print(len(entries), "entries satisfy criteria")
    random.shuffle(entries)

    tokens = {codes_map[g]: tokenizer(a)["input_ids"][0] for g, a in answer_map.items()}

    prompts = []
    classes = []
    sequence_lengths = []

    count_male, count_female = 0, 0
    for entry in entries:
        name, code = entry[0], entry[1]

        if code == "M":
            if count_male >= n_male:
                continue
            count_male += 1
        elif code == "F":
            if count_female >= n_female:
                continue
            count_female += 1

        t = tokenizer(
            prompt.format(name=" " + name),
            padding="max_length",
            max_length=prompt_len + max_name_toks,
        )

        try:
            seq_len = t["input_ids"].index(pad_token_id)
        except ValueError:
            seq_len = len(t["input_ids"])

        sequence_lengths.append(seq_len)
        prompts.append(t["input_ids"])
        classes.append(codes_map[code])

    assert count_male == n_male
    assert count_female == n_female

    prompts = torch.tensor(prompts)
    classes = torch.tensor(classes)
    sequence_lengths = torch.tensor(sequence_lengths)

    return prompts, classes, tokens, sequence_lengths
