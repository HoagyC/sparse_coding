from run import run_mmcs_with_larger, AutoEncoder, plot_hist

import torch

from utils import dotdict, get_activation_size

import os

def main(filepath: str, cfg: dotdict, outputs_folder: str = "outputs"):
    with open(filepath, "rb") as f:
        model_state_dicts = torch.load(f)
    
    activation_dim  = get_activation_size(cfg.model_name, cfg.layer_loc)
    print("loaded model state dicts")

    l1_range = [cfg.l1_exp_base**exp for exp in range(cfg.l1_exp_low, cfg.l1_exp_high)]
    dict_ratios = [cfg.dict_ratio_exp_base**exp for exp in range(cfg.dict_ratio_exp_low, cfg.dict_ratio_exp_high)]
    dict_sizes = [int(activation_dim * ratio) for ratio in dict_ratios]

    auto_encoders = [[AutoEncoder(activation_dim, n_feats, l1_coef=l1_ndx).to(cfg.device) for n_feats in dict_sizes] for l1_ndx in l1_range]
    
    for l1_ndx, l1_coef in enumerate(l1_range):
        for dict_ndx, dict_size in enumerate(dict_sizes):
            auto_encoders[l1_ndx][dict_ndx].load_state_dict(model_state_dicts[f"l1={l1_coef:.2E}_ds={dict_size}"])
    
    learned_dicts = [[auto_e.decoder.weight.detach().cpu().data.t() for auto_e in l1] for l1 in auto_encoders]
    mmcs_with_larger, feats_above_threshold, mcs = run_mmcs_with_larger(learned_dicts, threshold=cfg.threshold, device=cfg.device)

    print("calculated mmcs")

    os.makedirs(outputs_folder, exist_ok=True)

    plot_hist(mcs, l1_range, dict_sizes, show=False, save_folder=outputs_folder, title=f"Max Cosine Similarities", save_name="histogram_max_cosine_sim.png")

if __name__ == "__main__":
    cfg = dotdict({
        "l1_exp_low": -18,
        "l1_exp_high": -9,
        "l1_exp_base": 10 ** (1 / 4),
        "dict_ratio_exp_low": 1,
        "dict_ratio_exp_high": 4,
        "dict_ratio_exp_base": 2,
        "device": "cpu",
        "threshold": 0.9,
        "model_name": "EleutherAI/pythia-70m-deduped",
        "layer_loc": "residual",
    })

    filepath = "autoencoders.pth"

    main(filepath, cfg)