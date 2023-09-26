# code & dataset from: https://github.com/redwoodresearch/Easy-Transformer/blob/main/easy_transformer/ioi_dataset.py

import io
from logging import warning
from typing import Union, List
from site import PREFIXES
import warnings
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
import random
import re
import matplotlib.pyplot as plt
import random as rd
import copy

NAMES = [
    "Michael",
    "Christopher",
    "Jessica",
    "Matthew",
    "Ashley",
    "Jennifer",
    "Joshua",
    "Amanda",
    "Daniel",
    "David",
    "James",
    "Robert",
    "John",
    "Joseph",
    "Andrew",
    "Ryan",
    "Brandon",
    "Jason",
    "Justin",
    "Sarah",
    "William",
    "Jonathan",
    "Stephanie",
    "Brian",
    "Nicole",
    "Nicholas",
    "Anthony",
    "Heather",
    "Eric",
    "Elizabeth",
    "Adam",
    "Megan",
    "Melissa",
    "Kevin",
    "Steven",
    "Thomas",
    "Timothy",
    "Christina",
    "Kyle",
    "Rachel",
    "Laura",
    "Lauren",
    "Amber",
    "Brittany",
    "Danielle",
    "Richard",
    "Kimberly",
    "Jeffrey",
    "Amy",
    "Crystal",
    "Michelle",
    "Tiffany",
    "Jeremy",
    "Benjamin",
    "Mark",
    "Emily",
    "Aaron",
    "Charles",
    "Rebecca",
    "Jacob",
    "Stephen",
    "Patrick",
    "Sean",
    "Erin",
    "Jamie",
    "Kelly",
    "Samantha",
    "Nathan",
    "Sara",
    "Dustin",
    "Paul",
    "Angela",
    "Tyler",
    "Scott",
    "Katherine",
    "Andrea",
    "Gregory",
    "Erica",
    "Mary",
    "Travis",
    "Lisa",
    "Kenneth",
    "Bryan",
    "Lindsey",
    "Kristen",
    "Jose",
    "Alexander",
    "Jesse",
    "Katie",
    "Lindsay",
    "Shannon",
    "Vanessa",
    "Courtney",
    "Christine",
    "Alicia",
    "Cody",
    "Allison",
    "Bradley",
    "Samuel",
]

ABC_TEMPLATES = [
    "Then, [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
    "Afterwards [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
    "When [A], [B] and [C] arrived at the [PLACE], [B] and [C] gave a [OBJECT] to [A]",
    "Friends [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
]

BAC_TEMPLATES = [
    template.replace("[B]", "[A]", 1).replace("[A]", "[B]", 1)
    for template in ABC_TEMPLATES
]

BABA_TEMPLATES = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]

BABA_LONG_TEMPLATES = [
    "Then in the morning, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After taking a long break [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When soon afterwards [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When soon afterwards [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While spending time together [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While spending time together [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch in the afternoon, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, while spending time together [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then in the morning afterwards, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The local big [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends separated at birth [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]

BABA_LATE_IOS = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument and after that [B] said to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
]

BABA_EARLY_IOS = [
    "Then [B] and [A] went to the [PLACE], and [B] gave a [OBJECT] to [A]",
    "Then [B] and [A] had a lot of fun at the [PLACE], and [B] gave a [OBJECT] to [A]",
    "Then [B] and [A] were working at the [PLACE], and [B] decided to give a [OBJECT] to [A]",
    "Then [B] and [A] were thinking about going to the [PLACE], and [B] wanted to give a [OBJECT] to [A]",
    "Then [B] and [A] had a long argument, and after that [B] said to [A]",
    "After the lunch [B] and [A] went to the [PLACE], and [B] gave a [OBJECT] to [A]",
    "Afterwards [B] and [A] went to the [PLACE], and [B] gave a [OBJECT] to [A]",
    "Then [B] and [A] had a long argument, and afterwards [B] said to [A]",
]

TEMPLATES_VARIED_MIDDLE = [
    "",
]

# no end of texts, GPT-2 small wasn't trained this way (ask Arthur)
# warnings.warn("Adding end of text prefixes!")
# for TEMPLATES in [BABA_TEMPLATES, BABA_EARLY_IOS, BABA_LATE_IOS]:
#     for i in range(len(TEMPLATES)):
#         TEMPLATES[i] = "<|endoftext|>" + TEMPLATES[i]

ABBA_TEMPLATES = BABA_TEMPLATES[:]
ABBA_LATE_IOS = BABA_LATE_IOS[:]
ABBA_EARLY_IOS = BABA_EARLY_IOS[:]

for TEMPLATES in [ABBA_TEMPLATES, ABBA_LATE_IOS, ABBA_EARLY_IOS]:
    for i in range(len(TEMPLATES)):
        first_clause = True
        for j in range(1, len(TEMPLATES[i]) - 1):
            if TEMPLATES[i][j - 1 : j + 2] == "[B]" and first_clause:
                TEMPLATES[i] = TEMPLATES[i][:j] + "A" + TEMPLATES[i][j + 1 :]
            elif TEMPLATES[i][j - 1 : j + 2] == "[A]" and first_clause:
                first_clause = False
                TEMPLATES[i] = TEMPLATES[i][:j] + "B" + TEMPLATES[i][j + 1 :]

VERBS = [" tried", " said", " decided", " wanted", " gave"]
PLACES = [
    "store",
    "garden",
    "restaurant",
    "school",
    "hospital",
    "office",
    "house",
    "station",
]
OBJECTS = [
    "ring",
    "kiss",
    "bone",
    "basketball",
    "computer",
    "necklace",
    "drink",
    "snack",
]

ANIMALS = [
    "dog",
    "cat",
    "snake",
    "elephant",
    "beetle",
    "hippo",
    "giraffe",
    "tiger",
    "husky",
    "lion",
    "panther",
    "whale",
    "dolphin",
    "beaver",
    "rabbit",
    "fox",
    "lamb",
    "ferret",
]

def multiple_replace(dict, text):
    # from: https://stackoverflow.com/questions/15175142/how-can-i-do-multiple-substitutions-using-regex
    # Create a regular expression from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start() : mo.end()]], text)


def iter_sample_fast(iterable, samplesize):
    results = []
    # Fill in the first samplesize elements:
    try:
        for _ in range(samplesize):
            results.append(next(iterable))
    except StopIteration:
        raise ValueError("Sample larger than population.")
    random.shuffle(results)  # Randomize their positions

    return results

NOUNS_DICT = NOUNS_DICT = {"[PLACE]": PLACES, "[OBJECT]": OBJECTS}


def gen_prompt_counterfact(
    tokenizer,
    templates, names, nouns_dict, N
):
    nb_gen = 0
    ioi_prompts = []
    ioi_prompts_counterfact = []
    for nb_gen in range(N):
        temp = rd.choice(templates)
        temp_id = templates.index(temp)
        name_1 = ""
        name_2 = ""
        name_3 = ""
        while len(set([name_1, name_2, name_3])) < 3:
            name_1 = rd.choice(names)
            name_2 = rd.choice(names)
            name_3 = rd.choice(names)

            for name in [name_1, name_2, name_3]:
                if len(tokenizer(" " + name)["input_ids"]) != 1:
                    name_1 = ""
                    name_2 = ""
                    name_3 = ""
                    break

        assert all([len(tokenizer(" " + name)["input_ids"]) == 1 for name in [name_1, name_2, name_3]])

        nouns = {}
        ioi_prompt = {}
        ioi_prompt_counterfact = {}
        for k in nouns_dict:
            nouns[k] = rd.choice(nouns_dict[k])
            ioi_prompt[k] = nouns[k]
            ioi_prompt_counterfact[k] = nouns[k]
        prompt = temp
        for k in nouns_dict:
            prompt = prompt.replace(k, nouns[k])

        prompt1 = prompt.replace("[A]", name_1)
        prompt1 = prompt1.replace("[B]", name_2)
        ioi_prompt["text"] = prompt1
        ioi_prompt["IO"] = name_1
        ioi_prompt["S"] = name_2
        ioi_prompt["TEMPLATE_IDX"] = temp_id
        ioi_prompts.append(ioi_prompt)

        prompt2 = prompt.replace("[A]", name_3)
        prompt2 = prompt2.replace("[B]", name_2)
        ioi_prompt_counterfact["text"] = prompt2
        ioi_prompt_counterfact["IO"] = name_3
        ioi_prompt_counterfact["S"] = name_2
        ioi_prompt_counterfact["TEMPLATE_IDX"] = temp_id
        ioi_prompts_counterfact.append(ioi_prompt_counterfact)
    
    return ioi_prompts, ioi_prompts_counterfact

def gen_ioi_dataset(
    tokenizer,
    n_prompts,
):
    assertion = False
    while not assertion:
        prompts, prompts_cf = gen_prompt_counterfact(
            tokenizer,
            ABBA_TEMPLATES + BABA_TEMPLATES,
            NAMES,
            NOUNS_DICT,
            n_prompts,
        )

        prompts = [prompt["text"] for prompt in prompts]
        prompts_cf = [prompt["text"] for prompt in prompts_cf]

        # ignore final token (indirect object)
        prompts = tokenizer(prompts)["input_ids"]
        prompts_cf = tokenizer(prompts_cf)["input_ids"]

        assertion = all([len(prompt) == len(prompt_cf) for prompt, prompt_cf in zip(prompts, prompts_cf)])

    # calc seq lengths & pad
    seq_lengths = torch.tensor([len(prompt)-1 for prompt in prompts])
    max_seq_length = torch.max(seq_lengths)

    prompts = torch.stack(
        [torch.tensor(prompt[:-1] + [0]*(max_seq_length - len(prompt[:-1]))) for prompt in prompts]
    )

    prompts_cf = torch.stack(
        [torch.tensor(prompt[:-1] + [0]*(max_seq_length - len(prompt[:-1]))) for prompt in prompts_cf]
    )

    return prompts, prompts_cf, seq_lengths