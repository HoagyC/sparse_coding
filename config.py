import argparse
from dataclasses import dataclass
from typing import Any, Optional

import torch

@dataclass
class BaseArgs:
    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        for key, value in vars(self).items():
            parser.add_argument(f"--{key}", type=type(value), default=None)
        return parser.parse_args()
        
    def __post_init__(self) -> None:
        # parse command line arguments and update the class
        command_line_args = self.parse_args()
        extra_args = set(vars(command_line_args)) - set(vars(self))
        if extra_args: 
            raise ValueError(f"Unknown arguments: {extra_args}")
        self.update(command_line_args)
    
    def update(self, args: Any) -> None:
        for key, value in vars(args).items():
            if value is not None:
                print(f"From command line, setting {key} to {value}")
                setattr(self, key, value)

@dataclass
class TrainArgs(BaseArgs):
    layer: int = 2
    layer_loc: str = "residual"
    model_name: str = "pythia-70m-deduped"
    dataset_name: str = "openwebtext"
    dataset_folder: str = ""
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    tied_ae: bool = False
    seed: int = 0
    learned_dict_ratio: float = 1.0
    output_folder: str = "outputs"
    dtype: torch.dtype = torch.float32
    epochs: int = 1
    center_dataset: bool = False
    n_chunks: int = 30
    chunk_size_gb: float = 2.0
    batch_size: int = 256
    use_wandb: bool = True
    wandb_images: bool = False
    lr: float = 1e-3
    l1_alpha: float = 1e-3
    save_every: int = 5
    n_epochs: int = 1
    
@dataclass
class EnsembleArgs(TrainArgs):
    activation_width: int = 512
    use_synthetic_dataset: bool = False
    bias_decay: float = 0.0

@dataclass
class SyntheticEnsembleArgs(EnsembleArgs):
    noise_magnitude_scale: float = 0.0
    feature_prob_decay: float = 0.99
    feature_num_nonzero: int = 10
    gen_batch_size: int = 4096
    dataset_folder: str = "activation_data"
    n_ground_truth_components: int = 512  
    correlated_components: bool = False
    
    
@dataclass
class ErasureArgs(BaseArgs):
    model_name: str = "EleutherAI/pythia-70m-deduped"
    device: str = "cuda:4"
    layer: Optional[int] = None
    count_cutoff: int = 10000
    output_folder: str = "output_erasure_pca"
    activation_filename: str = "activation_data_erasure.pt"
    dict_filename: str = "/mnt/ssd-cluster/bigrun0308/tied_residual_l{layer}_r4/_9/learned_dicts.pt"
    
@dataclass
class ToyArgs(BaseArgs):
    layer: int = 2
    layer_loc: str = "residual"
    model_name: str = "pythia-70m-deduped"
    dataset_name: str = "openwebtext"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    tied_ae: bool = False
    seed: int = 0
    learned_dict_ratio: float = 1.0
    output_folder: str = "outputs"
    dtype: torch.dtype = torch.float32
    activation_dim: int = 256
    feature_prob_decay: float = 0.99
    feature_num_nonzero: int = 5
    correlated_components: bool = False
    n_ground_truth_components: int = 512
    noise_std: float = 0.1
    l1_exp_low: int = -12
    l1_exp_high: int = -11
    l1_exp_base: float = 10 ** (1/4)
    dict_ratio_exp_low: int = 1
    dict_ratio_exp_high: int = 7
    dict_ratio_exp_base: float = 2
    batch_size: int = 4096
    lr: float = 1e-3
    epochs: int = 1
    noise_level: float = 0.0
    n_components_dictionary: int = 512
    l1_alpha: float = 1e-3

@dataclass
class InterpArgs(BaseArgs):
    layer: int = 2
    model_name: str = "EleutherAI/pythia-70m-deduped"
    layer_loc: str = "residual"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_feats_explain: int = 10
    load_interpret_autoencoder: str = ""
    tied_ae: bool = False
    interp_name: str = ""
    sort_mode: str = "max"
    use_decoder: bool = True
    df_n_feats: int = 200
    top_k: int = 50
    save_loc: str = ""
    
    
@dataclass
class InterpGraphArgs(BaseArgs):
    layer: int = 1 
    model_name: str = "EleutherAI/pythia-70m/deduped"
    layer_loc: str = "mlp"
    score_mode: str = "all"
    run_all: bool = False
    
class InvestigateArgs(BaseArgs):
    threshold: float = 0.9
    layer: int = 2
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
