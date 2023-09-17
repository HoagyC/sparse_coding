import pickle
from transformers import AutoTokenizer
import torch

prompt = "I am a{occupation} and my pronouns are"
prompt_len = 7
prompt_skip = 3

codes_map = {"M": 0, "F": 1}
answer_map = {"M": " he", "F": " she"}

def generate_winobias_dataset(
    tokenizer_name="EleutherAI/pythia-70m-deduped",
    pad_token_id=0,
    n_prompts=None,
):
    with open("stereotyped_prompts.pkl", "rb") as f:
        prompts = pickle.load(f)
    
    if n_prompts is not None:
        prompts = prompts[:n_prompts]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    tokens = {codes_map[g]: tokenizer(a)["input_ids"][0] for g, a in answer_map.items()}

    prompt_tokens = []
    classes = []
    sequence_lengths = []

    for entry in prompts:
        occupation = entry["occupation"]
        gender = entry["gender"]
    
        t = tokenizer(
            prompt.format(occupation=occupation),
            padding="do_not_pad",
        )

        sequence_lengths.append(len(t["input_ids"]))
        prompt_tokens.append(t["input_ids"])
        classes.append(codes_map[gender])

    max_seq_len = max(sequence_lengths)

    prompt_tokens = [
        seq + [pad_token_id]*(max_seq_len-seq_len)
        for seq, seq_len in zip(prompt_tokens, sequence_lengths)
    ]

    prompt_tokens = torch.tensor(prompt_tokens)
    classes = torch.tensor(classes)
    sequence_lengths = torch.tensor(sequence_lengths)

    return prompt_tokens, classes, tokens, sequence_lengths