import numpy as np
import torch

abb_a_prompt = "Then, {name_a} and {name_b} were working at the {location}. {name_b} decided to give a {object} to {name_a}"
aba_b_prompt = "Then, {name_a} and {name_b} were working at the {location}. {name_a} decided to give a {object} to {name_b}"

names = [
    "Jennifer",
    "Richard",
    "Catherine",
    "Laura",
    "Jake",
    "Parker",
    "Jeff",
    "Charlotte",
    "Michael",
    "Max",
]
locations = ["plateau", "cafe", "home", "bridge", "station"]
objects = ["feather", "towel", "fins", "ring", "tape", "shorts"]


def generate_ioi_dataset(
    tokenizer,
    n_abb_a,
    n_abb_b,
):
    # validate dataset lengths
    error = False

    for name in names:
        if len(tokenizer(" " + name)["input_ids"]) != 1:
            print(f"Name {name} is not a single token")
            error = True

    for location in locations:
        if len(tokenizer(" " + location)["input_ids"]) != 1:
            print(f"Location {location} is not a single token")
            error = True

    for object in objects:
        if len(tokenizer(" " + object)["input_ids"]) != 1:
            print(f"Object {object} is not a single token")
            error = True

    assert not error, "Dataset is not valid"

    clean = []
    corrupted = []

    for i in range(n_abb_a):
        (name_a, name_b), location, object = (
            np.random.choice(names, size=2, replace=False),
            np.random.choice(locations),
            np.random.choice(objects),
        )
        clean.append(abb_a_prompt.format(name_a=name_a, name_b=name_b, location=location, object=object))
        corrupted.append(aba_b_prompt.format(name_a=name_a, name_b=name_b, location=location, object=object))

    for i in range(n_abb_b):
        (name_a, name_b), location, object = (
            np.random.choice(names, size=2, replace=False),
            np.random.choice(locations),
            np.random.choice(objects),
        )
        clean.append(aba_b_prompt.format(name_a=name_a, name_b=name_b, location=location, object=object))
        corrupted.append(abb_a_prompt.format(name_a=name_a, name_b=name_b, location=location, object=object))

    # print(clean)
    # print(corrupted)

    clean = torch.tensor(tokenizer(clean)["input_ids"])
    corrupted = torch.tensor(tokenizer(corrupted)["input_ids"])

    return clean, corrupted
