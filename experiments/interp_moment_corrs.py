import os
import sys
from typing import Dict, List

sys.path.append(os.getcwd())

import torch

from interpret import read_transform_scores
from standard_metrics import calc_moments_streaming

OUTPUTS_DIR = "/mnt/ssd-cluster/bigrun0308"
INTERP_DIR = "/mnt/ssd-cluster/auto_interp_results"

def main(device: str) -> None:
    layers = list(range(6))
    ratio_strs = ["0.5", "1.0", "2.0", "4.0"] #"8.0", "16.0"
    
    # init list for each moment
    n_active_levels = []
    mean_levels = []
    var_levels = []
    skew_levels = []
    kurtosis_levels = []
    l4_norm_levels = []
    all_transform_scores = []
    
    n_active_correlations = []
    mean_correlations = []
    variance_correlations = []
    skew_correlations = []
    kurtosis_correlations = []
    l4_norm_correlations = []
    
    for layer in layers:
        chunk = torch.load(os.path.join("/mnt/ssd-cluster/single_chunks", f"l{layer}_residual", "0.pt"))
        chunk = chunk.to(torch.float32).to(device)
        for ratio_str in ratio_strs:
            run_folder = f"tied_residual_l{layer}_r{ratio_str[0]}"
            dict_loc = os.path.join(OUTPUTS_DIR, run_folder, "_9", "learned_dicts.pt")
            dicts = torch.load(dict_loc, map_location=device)
            chosen_dict, dict_params = [d for d in dicts if abs(d[1]["l1_alpha"] - 0.00086) < 1e-5][0]
            chosen_dict.to_device(device)
            
            results_folder_name = f"l{layer}_residual/tied_r{ratio_str}_l1a0.00086"
            results_loc = os.path.join(INTERP_DIR, results_folder_name)
            transform_ndxs, transform_scores = read_transform_scores(results_loc, score_mode="random")

            times_active, means, vars, skews, kurtoses, l4_norm = calc_moments_streaming(chosen_dict, chunk, batch_size=1000)
            all_transform_scores.extend(transform_scores)
            
            mean_levels.extend(means[transform_ndxs].tolist())
            var_levels.extend(vars[transform_ndxs].tolist())
            skew_levels.extend(skews[transform_ndxs].tolist())
            kurtosis_levels.extend(kurtoses[transform_ndxs].tolist())
            n_active_levels.extend(times_active[transform_ndxs].tolist())
            l4_norm_levels.extend(l4_norm[transform_ndxs].tolist())
            
            mini_corr_n_active = torch.corrcoef(torch.stack([torch.tensor(times_active[transform_ndxs]).cpu(), torch.tensor(transform_scores)]))[0, 1]
            mini_corr_means = torch.corrcoef(torch.stack([torch.tensor(means[transform_ndxs]).cpu(), torch.tensor(transform_scores)]))[0, 1]
            mini_corr_vars = torch.corrcoef(torch.stack([torch.tensor(vars[transform_ndxs]).cpu(), torch.tensor(transform_scores)]))[0, 1]
            mini_corr_skews = torch.corrcoef(torch.stack([torch.tensor(skews[transform_ndxs]).cpu(), torch.tensor(transform_scores)]))[0, 1]
            mini_corr_kurtoses = torch.corrcoef(torch.stack([torch.tensor(kurtoses[transform_ndxs]).cpu(), torch.tensor(transform_scores)]))[0, 1]
            mini_corr_l4_norm = torch.corrcoef(torch.stack([torch.tensor(l4_norm[transform_ndxs]).cpu(), torch.tensor(transform_scores)]))[0, 1]
            
            n_active_correlations.append(mini_corr_n_active)
            mean_correlations.append(mini_corr_means)
            variance_correlations.append(mini_corr_vars)
            skew_correlations.append(mini_corr_skews)
            kurtosis_correlations.append(mini_corr_kurtoses)
            l4_norm_correlations.append(mini_corr_l4_norm)
            
            print(f"Layer {layer}, ratio {ratio_str} correlations:")
            print(f"Mean: {mini_corr_means}")
            print(f"Var: {mini_corr_vars}")
            print(f"Skew: {mini_corr_skews}")
            print(f"Kurtosis: {mini_corr_kurtoses}")
            
            print(f"Layer {layer}, ratio {ratio_str} done")
            
    n_active_levels_t = torch.tensor(n_active_levels)
    mean_levels_t = torch.tensor(mean_levels)
    var_levels_t = torch.tensor(var_levels)
    skew_levels_t = torch.tensor(skew_levels)
    kurtosis_levels_t = torch.tensor(kurtosis_levels)
    l4_norm_levels_t = torch.tensor(l4_norm_levels)
    
    # calculate correlations
    mean_corr = torch.corrcoef(torch.stack([mean_levels_t, torch.tensor(all_transform_scores)]))[0, 1]
    var_corr = torch.corrcoef(torch.stack([var_levels_t, torch.tensor(all_transform_scores)]))[0, 1]
    skew_corr = torch.corrcoef(torch.stack([skew_levels_t, torch.tensor(all_transform_scores)]))[0, 1]
    kurt_corr = torch.corrcoef(torch.stack([kurtosis_levels_t, torch.tensor(all_transform_scores)]))[0, 1]
    n_active_corr = torch.corrcoef(torch.stack([n_active_levels_t, torch.tensor(all_transform_scores)]))[0, 1]
    l4_norm_corr = torch.corrcoef(torch.stack([l4_norm_levels_t, torch.tensor(all_transform_scores)]))[0, 1]
    
    log_skews = torch.log(skew_levels_t + skew_levels_t.min())
    log_kurtoses = torch.log(kurtosis_levels_t + kurtosis_levels_t.min())
    
    log_skew_corr = torch.corrcoef(torch.stack([log_skews, torch.tensor(all_transform_scores)]))[0, 1]
    log_kurtosis_corr = torch.corrcoef(torch.stack([log_kurtoses, torch.tensor(all_transform_scores)]))[0, 1]
    log_l4_norm_corr = torch.corrcoef(torch.stack([torch.log(l4_norm_levels_t), torch.tensor(all_transform_scores)]))[0, 1]
    
    print(f"Mean correlation: {mean_corr}")
    print(f"Variance correlation: {var_corr}")
    print(f"Skew correlation: {skew_corr}")
    print(f"Kurtosis correlation: {kurt_corr}")
    print(f"Number of times active correlation: {n_active_corr}")
    print(f"L4 norm correlation: {l4_norm_corr}")
    
    print(f"Log skew correlation: {log_skew_corr}")
    print(f"Log kurtosis correlation: {log_kurtosis_corr}")
    print(f"Log l4 norm correlation: {log_l4_norm_corr}")
    
    print(f"Average n_active correation: {sum(n_active_correlations) / len(n_active_correlations)}")
    print(f"Average mean correation: {sum(mean_correlations) / len(mean_correlations)}")
    print(f"Average variance correation: {sum(variance_correlations) / len(variance_correlations)}")
    print(f"Average skew correation: {sum(skew_correlations) / len(skew_correlations)}")
    print(f"Average kurtosis correation: {sum(kurtosis_correlations) / len(kurtosis_correlations)}")
    print(f"Average l4 norm correation: {sum(l4_norm_correlations) / len(l4_norm_correlations)}")
    

if __name__ == "__main__":
    device = "cuda:6" if torch.cuda.is_available() else "cpu"
    main(device)