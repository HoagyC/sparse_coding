import pickle
import random

import torch
from transformers import AutoTokenizer

COUNT_CUTOFF = 100000

def generate_gender_dataset(
    tokenizer_name,
    pad_token_id=0,
    count_cutoff=COUNT_CUTOFF,
    sample_n=10,
    prompts_per_name=1,
    n_few_shot=3,
    randomise=True,
):
    prompt = " My name is{name} and I am a{answer}."
    prompt_q = " My name is{name} and I am a"
    prompt_len = 10
    prompt_q_len = 8

    codes_map = {"M": 0, "F": 1}
    answer_map = {"M": " male", "F": " female"}

    skip_tokens = 2 * n_few_shot * prompt_len + 3

    random.seed(42)
    #if sample_n is not None:
    #    targets = {"M": sample_n, "F": sample_n}
    #    counts = {"M": 0, "F": 0}

    with open("gender_dataset.pkl", "rb") as f:
        name_toks, entries = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token_id = pad_token_id

    entries = [entry for entry in entries if int(entry[2]) > count_cutoff]

    if randomise:
        random.shuffle(entries)

    # split into male and female names
    names_classes = {"M": [], "F": []}

    for (name, gender, _, _) in entries:
        if len(names_classes[gender]) >= sample_n:
            continue
        
        names_classes[gender].append(name)
    
    assert len(names_classes["M"]) == sample_n, len(names_classes["M"])
    assert len(names_classes["F"]) == sample_n, len(names_classes["F"])
    
    names_classes_list = [(k, name) for k, names in names_classes.items() for name in names]

    prompts = []
    classes = []
    sequence_lengths = []

    for gender, name in names_classes_list:
        for _ in range(prompts_per_name):
            names_few_shot = []
            for k, names in names_classes.items():
                names_ = [name_ for name_ in names if name_ != name]
                names_few_shot += [(k, name_) for name_ in random.sample(names_, n_few_shot)]
            
            random.shuffle(names_few_shot)

            strprompt = [prompt.format(name=" "+name_,answer=answer_map[k]) for k, name_ in names_few_shot]
            strprompt = "".join(strprompt) + prompt_q.format(name=" "+name)

            prompt_tokens = tokenizer(strprompt)["input_ids"]
            prompts.append(prompt_tokens)
            sequence_lengths.append(len(prompt_tokens))
            classes.append(codes_map[gender])

    completion_tokens = {codes_map[g]: tokenizer(a)["input_ids"][0] for g, a in answer_map.items()}

    prompts = torch.tensor(prompts)
    classes = torch.tensor(classes)
    sequence_lengths = torch.tensor(sequence_lengths)

    return prompts, classes, completion_tokens, sequence_lengths, skip_tokens


def generate_pronoun_dataset(
    tokenizer_name,
    pad_token_id=0,
    count_cutoff=COUNT_CUTOFF,
    sample_n=10,
    prompts_per_name=1,
    n_few_shot=3,
    randomise=True,
):
    objects = ["cat", "car", "dog", "book", "pen", "mouse", "chair", "table", "phone", "computer"]

    prompt = "{name} went to the store, and{answer} bought a {object}."
    prompt_q = "{name} went to the store, and"
    prompt_len = 11
    prompt_q_len = 7

    codes_map = {"M": 0, "F": 1}
    answer_map = {"M": " he", "F": " she"}

    skip_tokens = 2 * n_few_shot * prompt_len

    random.seed(42)
    #if sample_n is not None:
    #    targets = {"M": sample_n, "F": sample_n}
    #    counts = {"M": 0, "F": 0}

    with open("gender_dataset.pkl", "rb") as f:
        name_toks, entries = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token_id = pad_token_id

    entries = [entry for entry in entries if int(entry[2]) > count_cutoff]

    if randomise:
        random.shuffle(entries)

    # split into male and female names
    names_classes = {"M": [], "F": []}

    for (name, gender, _, _) in entries:
        if len(names_classes[gender]) >= sample_n:
            continue
        
        names_classes[gender].append(name)
    
    assert len(names_classes["M"]) == sample_n, len(names_classes["M"])
    assert len(names_classes["F"]) == sample_n, len(names_classes["F"])
    
    names_classes_list = [(k, name) for k, names in names_classes.items() for name in names]

    prompts = []
    classes = []
    sequence_lengths = []

    for gender, name in names_classes_list:
        for _ in range(prompts_per_name):
            names_few_shot = []
            for k, names in names_classes.items():
                names_ = [name_ for name_ in names if name_ != name]
                names_few_shot += [(k, name_) for name_ in random.sample(names_, n_few_shot)]
            
            random.shuffle(names_few_shot)

            prompt_objects = random.sample(objects, n_few_shot)

            strprompt = [
                prompt.format(name=" "+name_,answer=answer_map[k], object=object_)
                for (k, name_), object_ in zip(names_few_shot, objects)
            ]
            strprompt = "".join(strprompt) + prompt_q.format(name=" "+name)

            #print(strprompt)

            prompt_tokens = tokenizer(strprompt)["input_ids"]
            prompts.append(prompt_tokens)
            sequence_lengths.append(len(prompt_tokens))
            classes.append(codes_map[gender])

    completion_tokens = {codes_map[g]: tokenizer(a)["input_ids"][0] for g, a in answer_map.items()}

    prompts = torch.tensor(prompts)
    classes = torch.tensor(classes)
    sequence_lengths = torch.tensor(sequence_lengths)

    return prompts, classes, completion_tokens, sequence_lengths, skip_tokens