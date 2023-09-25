import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod
from torchtyping import TensorType
from typing import List, Tuple, Union, Optional
from transformer_lens import HookedTransformer
import standard_metrics
import test_datasets.ioi_counterfact
from autoencoders.pca import BatchedPCA
from autoencoders.learned_dict import LearnedDict, Identity
from utils import dotdict
import tqdm

class AttnDecomp(ABC):
    # a class to specify a decomposition of concatenated attn_out

    @abstractmethod
    def decompose(self,
        attn_out: TensorType["batch", "d_model"],
    ) -> TensorType["batch", "n_decomp", "d_decomp"]:
        pass
    
    @abstractmethod
    def recompose(self,
        original: TensorType["batch", "d_model"],
        decomposed: TensorType["batch", "n_decomp", "d_decomp"],
    ) -> TensorType["batch", "d_model"]:
        pass

    @property
    @abstractmethod
    def n_decomp(self) -> int:
        pass

    @property
    @abstractmethod
    def d_decomp(self) -> int:
        pass

class DictAttnDecomp(AttnDecomp):
    def __init__(self, dict: LearnedDict):
        self.dict = dict
    
    @property
    def n_decomp(self) -> int:
        return self.dict.n_dict_components()

    @property
    def d_decomp(self) -> int:
        return 1
    
    def decompose(self,
        attn_out: TensorType["batch", "d_model"],
    ) -> TensorType["batch", "n_decomp", "d_decomp"]:
        B, _ = attn_out.shape
        return self.dict.encode(self.dict.center(attn_out)).reshape(B, self.n_decomp, self.d_decomp)
    
    def recompose(self,
        original: TensorType["batch", "d_model"],
        decomposed: TensorType["batch", "n_decomp", "d_decomp"],
    ) -> TensorType["batch", "d_model"]:
        B, D, _ = decomposed.shape
        resid = original - self.dict.predict(original)
        return resid + self.dict.uncenter(self.dict.decode(decomposed.reshape(B, D)))

class HeadAttnDecomp(AttnDecomp):
    def __init__(self, n_heads: int, d_head: int):
        self.n_heads = n_heads
        self.d_head = d_head
    
    def decompose(self,
        attn_out: TensorType["batch", "d_model"],
    ) -> TensorType["batch", "n_decomp", "d_decomp"]:
        B, D = attn_out.shape
        return attn_out.reshape(B, self.n_decomp, self.d_decomp)
    
    @property
    def n_decomp(self) -> int:
        return self.n_heads
    
    @property
    def d_decomp(self) -> int:
        return self.d_head

    def recompose(self,
        _: TensorType["batch", "d_model"],
        decomposed: TensorType["batch", "n_decomp", "d_decomp"],
    ) -> TensorType["batch", "d_model"]:
        B, N, D = decomposed.shape
        return decomposed.reshape(B, N * D)

def run_with_decomp(
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    prompts: TensorType["batch", "prompt_len"],
    act_cf: TensorType["batch", "d_model"],
    decomposer: AttnDecomp,
    to_swap: List[int],
    location: Tuple[int, str],
    seq_lengths: TensorType["batch"],
    all_positions: bool = False,
) -> Tuple[TensorType["batch", "prompt_len", "d_vocab"], TensorType["batch", "d_model"]]:
    tensor_name = standard_metrics.get_model_tensor_name(location)

    B, D = act_cf.shape

    corrupted_codes = decomposer.decompose(act_cf)

    activation = None

    def intervention(tensor, hook=None):
        nonlocal activation
        if location[1] == "attn_concat":
            B, L, N_head, D_head = tensor.shape
            tensor = tensor.reshape(B, L, N_head * D_head)
        B, L, D = tensor.shape
        if all_positions:
            select_mask = torch.zeros(B, L, dtype=torch.bool, device="cuda:0")
            for i in range(B):
                select_mask[i, :seq_lengths[i]] = True
            act_clean = tensor[select_mask].reshape(-1, D)
        else:
            act_clean = tensor[list(range(B)), seq_lengths-1].clone()
        clean_codes = decomposer.decompose(act_clean)
        codes = corrupted_codes.clone()
        codes[:, to_swap] = clean_codes[:, to_swap]
        recomposed = decomposer.recompose(act_cf, codes)
        activation = recomposed.clone()
        if all_positions:
            select_mask = torch.zeros(B, L, dtype=torch.bool, device="cuda:0")
            for i in range(B):
                select_mask[i, :seq_lengths[i]] = True
            tensor[select_mask] = recomposed
        else:
            tensor[list(range(B)), seq_lengths-1] = recomposed
        if location[1] == "attn_concat":
            tensor = tensor.reshape(B, L, N_head, D_head)
        return tensor
    
    with torch.no_grad():
        logits = model.run_with_hooks(
            prompts,
            fwd_hooks=[(
                tensor_name,
                intervention,
            )]
        )

    return logits, activation

def mean_kl_divergence(
    p_logits: TensorType["batch", "d"],
    q_logits: TensorType["batch", "d"],
) -> float:
    return F.kl_div(
        F.log_softmax(p_logits, dim=-1),
        F.log_softmax(q_logits, dim=-1),
        log_target=True,
        reduction="batchmean",
    ).item()

def feat_ident_run(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: TensorType["batch", "prompt_len"],
    prompts_cf: TensorType["batch", "prompt_len"],
    seq_lengths: TensorType["batch"],
    decomposer: AttnDecomp,
    thresholds: List[float],
    location: Tuple[int, str],
    start_feats: Optional[List[int]] = None,
    all_positions: bool = False,
) -> List[Tuple[List[int], float, float]]:
    B, L = prompts.shape
    B_all = B
    if all_positions:
        B_all = torch.sum(seq_lengths).item()
    tensor_name = standard_metrics.get_model_tensor_name(location)

    if start_feats is None:
        start_feats = list(range(decomposer.n_decomp))
    
    feats_clean = start_feats[:]
    feats_cf = [i for i in range(decomposer.n_decomp) if i not in feats_clean]

    clean_logits, act_cache_clean = model.run_with_cache(
        prompts,
        return_type="logits",
        names_filter=lambda name: name == tensor_name,
    )

    if all_positions:
        select_mask = torch.zeros(B, L, dtype=torch.bool, device="cuda:0")
        for i in range(B):
            select_mask[i, :seq_lengths[i]] = True
        act_clean = act_cache_clean[tensor_name][select_mask].reshape(B_all, -1).clone()
    else:
        act_clean = act_cache_clean[tensor_name][list(range(B)), seq_lengths-1].reshape(B, -1).clone()

    _, act_cache_cf = model.run_with_cache(
        prompts_cf,
        return_type="logits",
        names_filter=lambda name: name == tensor_name,
    )

    if all_positions:
        select_mask = torch.zeros(B, L, dtype=torch.bool, device="cuda:0")
        for i in range(B):
            select_mask[i, :seq_lengths[i]] = True
        act_cf = act_cache_cf[tensor_name][select_mask].reshape(B_all, -1).clone()
    else:
        act_cf = act_cache_cf[tensor_name][list(range(B)), seq_lengths-1].reshape(B, -1).clone()

    prev_div = None

    def eval_div(to_swap):
        logits, activation = run_with_decomp(
            model,
            tokenizer,
            prompts,
            act_cf,
            decomposer,
            to_swap,
            location,
            seq_lengths,
            all_positions,
        )
        div = mean_kl_divergence(clean_logits[list(range(B)), seq_lengths-1], logits[list(range(B)), seq_lengths-1])
        dist = torch.norm(act_clean - activation, dim=-1).mean().item()
        return div, dist

    scores = []

    zero_div, zero_dist = eval_div(list(range(decomposer.n_decomp)))
    scores.append((list(range(decomposer.n_decomp)), zero_div, zero_dist))

    all_div, all_dist = eval_div([])
    scores.append(([], all_div, all_dist))

    start_div, start_dist = eval_div(feats_clean)
    scores.append((feats_cf, start_div, start_dist))

    prev_div = start_div

    for threshold in sorted(thresholds):
        if len(feats_clean) == 0:
            break
        for i in tqdm.tqdm(feats_clean[:]):
            to_swap = [j for j in feats_clean if j != i]
            div, dist = eval_div(to_swap)
            if div - prev_div < threshold:
                feats_clean.remove(i)
                feats_cf.append(i)
                prev_div = div
        print(f"Threshold {threshold} changed {len(feats_clean[:])}")
        scores.append((feats_clean[:], prev_div, dist))

    return scores

def feat_ident_run_layer(cfg):
    model = HookedTransformer.from_pretrained(cfg.model_name).to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    prompts, prompts_cf, seq_lengths = test_datasets.ioi_counterfact.gen_ioi_dataset(tokenizer, cfg.n_prompts)
    prompts = prompts.to("cuda:0")
    prompts_cf = prompts_cf.to("cuda:0")
    seq_lengths = seq_lengths.to("cuda:0")

    print(prompts.shape, prompts_cf.shape, seq_lengths.shape)
    print(seq_lengths)

    activation_dataset = torch.load(cfg.activation_dataset)

    pca = BatchedPCA(activation_dataset.shape[1], "cuda:0")

    for i in range(0, activation_dataset.shape[0], cfg.batch_size):
        j = min(i + cfg.batch_size, activation_dataset.shape[0])
        pca.train_batch(activation_dataset[i:j].to("cuda:0"))
    
    pca_pve = pca.to_pve_rotation_dict()
    pca_pve.to_device("cuda:0")
    pca_rot = pca.to_rotation_dict()
    pca_rot.to_device("cuda:0")

    del activation_dataset

    learned_dicts = torch.load(cfg.learned_dicts)

    l1_vals = [1e-3, 3e-4, 1e-4]

    selected_dicts = []

    for l1_val in l1_vals:
        closest_dist = float("inf")
        selected_dict = None
        for dict, hparams in learned_dicts:
            dist = abs(hparams["l1_alpha"] - l1_val)
            if dist < closest_dist:
                closest_dist = dist
                selected_dict = (dict, cfg.dict_name.format(l1_alpha=l1_val))
                dict.to_device("cuda:0")
            
        selected_dicts.append(selected_dict)

    del learned_dicts

    thresholds = np.logspace(cfg.log_threshold_min, cfg.log_threshold_max, cfg.n_thresholds)

    decomps = [
        ("pca_pve", DictAttnDecomp(pca_pve)),
        ("pca_rot", DictAttnDecomp(pca_rot)),
        ("neuron basis", DictAttnDecomp(Identity(pca_rot.n_dict_components())))
        ("head", HeadAttnDecomp(cfg.n_head, cfg.d_head)),
    ]
    decomps += [(name, DictAttnDecomp(dict)) for dict, name in selected_dicts]

    results = []

    for name, decomp in decomps:
        print(f"Running {name}")
        results.append((name, feat_ident_run(
            model,
            tokenizer,
            prompts,
            prompts_cf,
            seq_lengths,
            decomp,
            thresholds,
            cfg.location,
            all_positions=False,
        )))
    
    torch.save(results, cfg.output_file)

if __name__ == "__main__":
    cfg = dotdict({
        "model_name": "gpt2",
        "activation_dataset": "activation_data/layer_9/0.pt",
        "learned_dicts": "gpt2_small_l9_resid/learned_dicts_epoch_0.pt",
        "dict_name": "dict_{l1_alpha:.2e}",
        "location": (9, "attn_concat"),
        "n_prompts": 50,
        "batch_size": 100,
        "log_threshold_min": -4,
        "log_threshold_max": -1,
        "n_thresholds": 10,
        "n_head": 12,
        "d_head": 64,
        "output_file": "feat_ident_results.pt",
    })

    feat_ident_run_layer(cfg)