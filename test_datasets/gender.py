import pickle
import torch

COUNT_CUTOFF = 15000

#prompt = "{name}'s gender is {gender}"
prompt = "{name} uses the pronouns "

codes_map = {"M": 0, "F": 1}
answer_map = {"M": " he", "F": " she"}

def gen_gender_dataset(
    tokenizer,
    n_male, n_female,
):
    with open("gender_dataset.pkl", "rb") as f:
        entries = pickle.load(f)
    
    entries = [entry for entry in entries if int(entry[2]) > COUNT_CUTOFF]

    tokens = {codes_map[g]: tokenizer(a) for g, a in answer_map.items()}

    prompts = []
    classes = []

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
        
        t = tokenizer(prompt.format(name=name))
        
        prompts.append(t["input_ids"])
        classes.append(codes_map[code])
    
    assert count_male == n_male
    assert count_female == n_female

    prompts = torch.tensor(prompts)
    classes = torch.tensor(classes)

    return prompts, classes, tokens