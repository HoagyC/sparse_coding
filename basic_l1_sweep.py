from dataclasses import dataclass
import os
import tqdm

import torch
import torchopt
import numpy as np


from autoencoders.sae_ensemble import FunctionalTiedSAE
from autoencoders.ensemble import FunctionalEnsemble
from big_sweep import ensemble_train_loop, unstacked_to_learned_dicts
from config import TrainArgs, EnsembleArgs

class ProgressBar:
    def __init__(self, total, chunk_idx, n_chunks, epoch_idx, n_repetitions):
        """
        Initialize the progress bar.

        Parameters:
        - total (int): Total progress count.
        - chunk_idx (int): Current chunk index.
        - n_chunks (int): Total number of chunks.
        - epoch_idx (int): Current epoch index.
        - n_repetitions (int): Total number of repetitions (epochs).
        """
        if n_repetitions > 1:
            desc = "Epoch {}/{} - Chunk {}/{}".format(epoch_idx+1, n_repetitions, chunk_idx+1, n_chunks)
        else:
            desc = "Chunk {}/{}".format(chunk_idx+1, n_chunks)

        self.bar = tqdm.tqdm(total=total, desc=desc)
        self._value = 0
    
    @property
    def value(self):
        """Get the current progress value."""
        return self._value
    
    @value.setter
    def value(self, v):
        """Set the progress value and update the progress bar."""
        self.bar.update(v - self._value)
        self._value = v

def basic_l1_sweep(
    dataset_dir, output_dir,
    ratio, l1_values=np.logspace(-4, -2, 16), batch_size=256,
    device="cuda", adam_kwargs={"lr": 1e-3},
    n_repetitions=1,
    save_after_every=False, 
):
    # get dataset size
    
    # check that dataset_dir/0.pt exists

    assert os.path.exists(os.path.join(dataset_dir, '0.pt')), "Dataset not found at {}".format(dataset_dir)

    dataset = torch.load(os.path.join(dataset_dir, '0.pt'))
    activation_dim = dataset.shape[1]
    latent_dim = int(activation_dim * ratio)
    del dataset

    # create models

    print(f"Initializing {len(l1_values)} models with latent dimension {latent_dim}...")

    models = [FunctionalTiedSAE.init(activation_dim, latent_dim, l1, device=device) for l1 in l1_values]
    ensemble = FunctionalEnsemble(
        models, FunctionalTiedSAE,
        torchopt.adam, adam_kwargs,
        device=device
    )
    args = {
        "batch_size": batch_size,
        "device": device,
        "dict_size": latent_dim,
    }

    print("Training...")

    n_chunks = len(os.listdir(dataset_dir))

    os.makedirs(output_dir, exist_ok=True)

    for epoch_idx in range(n_repetitions):
        chunk_order = np.random.permutation(n_chunks)

        for chunk_idx, chunk in enumerate(chunk_order):
            assert os.path.exists(os.path.join(dataset_dir, '{}.pt'.format(chunk))), "Chunk not found at {}".format(os.path.join(dataset_dir, '{}.pt'.format(chunk)))
            dataset = torch.load(os.path.join(dataset_dir, '{}.pt'.format(chunk))).to(dtype=torch.float32)
            dataset.pin_memory()

            sampler = torch.utils.data.BatchSampler(
                torch.utils.data.RandomSampler(range(dataset.shape[0])),
                batch_size=batch_size,
                drop_last=False,
            )

            bar = ProgressBar(len(sampler), chunk_idx, n_chunks, epoch_idx, n_repetitions)

            cfg = TrainArgs()
            cfg.use_wandb = False

            ensemble_train_loop(ensemble, cfg, args, "ensemble", sampler, dataset, bar)

            if save_after_every:
                learned_dicts = unstacked_to_learned_dicts(ensemble, args, ["dict_size"], ["l1_alpha"])
            
                torch.save(learned_dicts, os.path.join(output_dir, f"learned_dicts_epoch_{epoch_idx}_chunk_{chunk_idx}.pt"))
        
        if not save_after_every:
            learned_dicts = unstacked_to_learned_dicts(ensemble, args, ["dict_size"], ["l1_alpha"])
        
            torch.save(learned_dicts, os.path.join(output_dir, f"learned_dicts_epoch_{epoch_idx}.pt"))


@dataclass
class SweepArgs(EnsembleArgs):
    dataset_dir: str = "data"
    output_dir: str = "sweep_outputs"
    l1_value_min: float = -4
    l1_value_max: float = -2
    l1_value_n: int = 16
    ratio: float = 1.0
    n_repetitions: int = 1
    save_after_every: bool = False
    adam_lr: float = 1e-3
    
    

if __name__ == "__main__":
    args = SweepArgs()

    #l1_values = list(np.logspace(args.l1_value_min, args.l1_value_max, args.l1_value_n))

    l1_values = [0, 1e-3, 3e-4, 1e-4]

    basic_l1_sweep(
        args.dataset_dir, args.output_dir,
        args.ratio, l1_values, args.batch_size,
        args.device, {"lr": args.adam_lr},
        args.n_repetitions,
        args.save_after_every
    )
