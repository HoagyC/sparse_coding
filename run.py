import argparse
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import datetime
import importlib
import itertools
import math
import multiprocessing as mp
import os
import pickle
from typing import Union, Tuple, List, Any, Optional, TypeVar

from baukit import Trace
from datasets import Dataset, DatasetDict, load_dataset
from einops import rearrange
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtyping import TensorType
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase, GPT2Tokenizer

from utils import dotdict
from nanoGPT_model import GPT

n_ground_truth_components, activation_dim, dataset_size = None, None, None
T = TypeVar("T", bound=Union[Dataset, DatasetDict])


# Nora's Code from https://github.com/AlignmentResearch/tuned-lens/blob/main/tuned_lens/data.py
def chunk_and_tokenize(
    data: T,
    tokenizer: PreTrainedTokenizerBase,
    *,
    format: str = "torch",
    num_proc: int = min(mp.cpu_count() // 2, 8),
    text_key: str = "text",
    max_length: int = 2048,
    return_final_batch: bool = False,
    load_from_cache_file: bool = True,
) -> tuple[T, float]:
    """Perform GPT-style chunking and tokenization on a dataset.

    The resulting dataset will consist entirely of chunks exactly `max_length` tokens
    long. Long sequences will be split into multiple chunks, and short sequences will
    be merged with their neighbors, using `eos_token` as a separator. The fist token
    will also always be an `eos_token`.

    Args:
        data: The dataset to chunk and tokenize.
        tokenizer: The tokenizer to use.
        format: The format to return the dataset in, passed to `Dataset.with_format`.
        num_proc: The number of processes to use for tokenization.
        text_key: The key in the dataset to use as the text to tokenize.
        max_length: The maximum length of a batch of input ids.
        return_final_batch: Whether to return the final batch, which may be smaller
            than the others.
        load_from_cache_file: Whether to load from the cache file.

    Returns:
        * The chunked and tokenized dataset.
        * The ratio of nats to bits per byte see https://arxiv.org/pdf/2101.00027.pdf,
            section 3.1.
    """

    def _tokenize_fn(x: dict[str, list]):
        chunk_size = min(tokenizer.model_max_length, max_length) # tokenizer max length is 1024 for gpt2
        sep = tokenizer.eos_token or "<|endoftext|>"
        joined_text = sep.join([""] + x[text_key])
        output = tokenizer(
            # Concatenate all the samples together, separated by the EOS token.
            joined_text,  # start with an eos token
            max_length=chunk_size,
            return_attention_mask=False,
            return_overflowing_tokens=True,
            truncation=True,
        )

        if overflow := output.pop("overflowing_tokens", None):
            # Slow Tokenizers return unnested lists of ints
            assert isinstance(output["input_ids"][0], int)

            # Chunk the overflow into batches of size `chunk_size`
            chunks = [output["input_ids"]] + [
                overflow[i * chunk_size : (i + 1) * chunk_size]
                for i in range(math.ceil(len(overflow) / chunk_size))
            ]
            output = {"input_ids": chunks}

        total_tokens = sum(len(ids) for ids in output["input_ids"])
        total_bytes = len(joined_text.encode("utf-8"))

        if not return_final_batch:
            # We know that the last sample will almost always be less than the max
            # number of tokens, and we don't want to pad, so we just drop it.
            output = {k: v[:-1] for k, v in output.items()}

        output_batch_size = len(output["input_ids"])

        if output_batch_size == 0:
            raise ValueError(
                "Not enough data to create a single batch complete batch."
                " Either allow the final batch to be returned,"
                " or supply more data."
            )

        # We need to output this in order to compute the number of bits per byte
        div, rem = divmod(total_tokens, output_batch_size)
        output["length"] = [div] * output_batch_size
        output["length"][-1] += rem

        div, rem = divmod(total_bytes, output_batch_size)
        output["bytes"] = [div] * output_batch_size
        output["bytes"][-1] += rem

        return output

    data = data.map(
        _tokenize_fn,
        # Batching is important for ensuring that we don't waste tokens
        # since we always throw away the last element of the batch we
        # want to keep the batch size as large as possible
        batched=True,
        batch_size=2048,
        num_proc=num_proc,
        remove_columns=get_columns_all_equal(data),
        load_from_cache_file=load_from_cache_file,
    )
    total_bytes: float = sum(data["bytes"])
    total_tokens: float = sum(data["length"])
    return data.with_format(format, columns=["input_ids"]), (
        total_tokens / total_bytes
    ) / math.log(2)


def get_columns_all_equal(dataset: Union[Dataset, DatasetDict]) -> list[str]:
    """Get a single list of columns in a `Dataset` or `DatasetDict`.

    We assert the columms are the same across splits if it's a `DatasetDict`.

    Args:
        dataset: The dataset to get the columns from.

    Returns:
        A list of columns.
    """
    if isinstance(dataset, DatasetDict):
        cols_by_split = dataset.column_names.values()
        columns = next(iter(cols_by_split))
        if not all(cols == columns for cols in cols_by_split):
            raise ValueError("All splits must have the same columns")

        return columns

    return dataset.column_names
# End Nora's Code from https://github.com/AlignmentResearch/tuned-lens/blob/main/tuned_lens/data.py

@dataclass
class RandomDatasetGenerator(Generator):
    activation_dim: int
    n_ground_truth_components: int
    batch_size: int
    feature_num_nonzero: int
    feature_prob_decay: float
    correlated: bool
    device: Union[torch.device, str]
    
    frac_nonzero: float = field(init=False)
    decay: TensorType['n_ground_truth_components'] = field(init=False)
    feats: TensorType['n_ground_truth_components', 'activation_dim'] = field(init=False)
    corr_matrix: Optional[TensorType['n_ground_truth_components', 
                                     'n_ground_truth_components']] = field(init=False) 
    component_probs: Optional[TensorType['n_ground_truth_components']] = field(init=False)

    def __post_init__(self):
        self.frac_nonzero = self.feature_num_nonzero / self.n_ground_truth_components

        # Define the probabilities of each component being included in the data
        self.decay = torch.tensor(
            [self.feature_prob_decay ** i for i in range(self.n_ground_truth_components)]
        ).to(self.device) #FIXME: 1 / i

        if self.correlated:
            self.corr_matrix = generate_corr_matrix(
                self.n_ground_truth_components, device=self.device
            )
        else:
            self.component_probs = (
                self.decay * self.frac_nonzero
            )  # Only if non-correlated
        self.feats = generate_rand_feats(
            self.activation_dim,
            self.n_ground_truth_components,
            device=self.device,
        )

    def send(self, ignored_arg: Any) -> TensorType['dataset_size', 'activation_dim']:

        if self.correlated:
            _, _, data = generate_correlated_dataset(
                self.n_ground_truth_components,
                self.batch_size,
                self.corr_matrix,
                self.feats,
                self.frac_nonzero,
                self.decay,
                self.device,
            )
        else:
            _, _, data = generate_rand_dataset(
                self.n_ground_truth_components,
                self.batch_size,
                self.component_probs,
                self.feats,
                self.device,
            )
        return data.to(torch.float16)

    def throw(self, type: Any = None, value: Any = None, traceback: Any = None) -> None:
        raise StopIteration

def generate_rand_dataset(
    n_ground_truth_components: int, # 
    dataset_size: int,
    feature_probs: TensorType['n_ground_truth_components'],
    feats: TensorType['n_ground_truth_components', 'activation_dim'],
    device: Union[torch.device, str],
) -> Tuple[
      TensorType['n_ground_truth_components', 'activation_dim'], 
      TensorType['dataset_size','n_ground_truth_components'], 
      TensorType['dataset_size', 'activation_dim']
    ]:

    dataset_thresh = torch.rand(dataset_size, n_ground_truth_components, device=device)
    dataset_values = torch.rand(dataset_size, n_ground_truth_components, device=device)

    data_zero = torch.zeros_like(dataset_thresh, device=device)


    dataset_codes = torch.where(
        dataset_thresh <= feature_probs,
        dataset_values,
        data_zero,
    ) # dim: dataset_size x n_ground_truth_components

    # Multiply by a 2D random matrix of feature strengths
    feature_strengths = torch.rand((dataset_size, n_ground_truth_components), device=device)
    dataset = (dataset_codes * feature_strengths) @ feats

    # dataset = dataset_codes @ feats

    return feats, dataset_codes, dataset


def generate_correlated_dataset(
    n_ground_truth_components: int,
    dataset_size: int,
    corr_matrix: TensorType['n_ground_truth_components', 'n_ground_truth_components'],
    feats: TensorType['n_ground_truth_components', 'activation_dim'],
    frac_nonzero: float,
    decay: TensorType['n_ground_truth_components'],
    device: Union[torch.device, str],
) -> Tuple[
      TensorType['n_ground_truth_components', 'activation_dim'], 
      TensorType['dataset_size','n_ground_truth_components'], 
      TensorType['dataset_size', 'activation_dim']
    ]:

    # Get a correlated gaussian sample
    mvn = torch.distributions.MultivariateNormal(
        loc=torch.zeros(n_ground_truth_components, device=device), covariance_matrix=corr_matrix
    )
    corr_thresh = mvn.sample()

    # Take the CDF of that sample.
    normal = torch.distributions.Normal(
        torch.tensor([0.0], device=device), torch.tensor([1.0], device=device)
    )
    cdf = normal.cdf(corr_thresh.squeeze())

    # Decay it
    component_probs = cdf * decay

    # Scale it to get the right % of nonzeros
    mean_prob = torch.mean(component_probs)
    scaler = frac_nonzero / mean_prob
    component_probs *= scaler
    # So np.isclose(np.mean(component_probs), frac_nonzero) will be True

    # Generate sparse correlated codes
    dataset_thresh = torch.rand(dataset_size, n_ground_truth_components, device=device)
    dataset_values = torch.rand(dataset_size, n_ground_truth_components, device=device)

    data_zero = torch.zeros_like(corr_thresh, device=device)
    dataset_codes = torch.where(
        dataset_thresh <= component_probs,
        dataset_values,
        data_zero,
    )
    # Ensure there are no datapoints w/ 0 features
    zero_sample_index = (dataset_codes.count_nonzero(dim=1) == 0).nonzero()[:,0]
    random_index = torch.randint(low=0, high=n_ground_truth_components, size=(zero_sample_index.shape[0],)).to(dataset_codes.device)
    dataset_codes[zero_sample_index, random_index] = 1.0

    # Multiply by a 2D random matrix of feature strengths
    feature_strengths = torch.rand((dataset_size, n_ground_truth_components), device=device)
    dataset = (dataset_codes * feature_strengths) @ feats

    return feats, dataset_codes, dataset


def generate_rand_feats(
    feat_dim: int,
    num_feats: int,
    device: Union[torch.device, str],
) -> TensorType['n_ground_truth_components', 'activation_dim']:
    data_path = os.path.join(os.getcwd(), "data")
    data_filename = os.path.join(data_path, f"feats_{feat_dim}_{num_feats}.npy")

    feats = np.random.multivariate_normal(
        np.zeros(feat_dim), np.eye(feat_dim), size=num_feats
    )
    feats = feats.T / np.linalg.norm(feats, axis=1)
    feats = feats.T

    feats_tensor = torch.from_numpy(feats).to(device).float()
    return feats_tensor


def generate_corr_matrix(
    num_feats: int, device: Union[torch.device, str]
) -> TensorType['n_ground_truth_components', 'n_ground_truth_components']:
    corr_mat_path = os.path.join(os.getcwd(), "data")
    corr_mat_filename = os.path.join(corr_mat_path, f"corr_mat_{num_feats}.npy")

    # Create a correlation matrix
    corr_matrix = np.random.rand(num_feats, num_feats)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    min_eig = np.min(np.real(np.linalg.eigvals(corr_matrix)))
    if min_eig < 0:
        corr_matrix -= (
            1.001 * min_eig * np.eye(corr_matrix.shape[0], corr_matrix.shape[1])
        )

    corr_matrix_tensor = torch.from_numpy(corr_matrix).to(device).float()

    return corr_matrix_tensor


# AutoEncoder Definition
class AutoEncoder(nn.Module):
    def __init__(self, activation_size, n_dict_components):
        super(AutoEncoder, self).__init__()
        # create decoder using float16 to save memory
        self.decoder = nn.Linear(n_dict_components, activation_size, bias=False)
        # Initialize the decoder weights orthogonally
        nn.init.orthogonal_(self.decoder.weight)
        self.decoder = self.decoder.to(torch.float16)

        self.encoder = nn.Sequential(
            nn.Linear(activation_size, n_dict_components).to(torch.float16),
            nn.ReLU()
        )
        
    def forward(self, x):
        c = self.encoder(x)

        # Apply unit norm constraint to the decoder weights
        self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)
    
        x_hat = self.decoder(c)
        return x_hat, c
    
    @property
    def device(self):
        return next(self.parameters()).device

def cosine_sim(
    vecs1: Union[torch.Tensor, torch.nn.parameter.Parameter, np.ndarray],
    vecs2: Union[torch.Tensor, torch.nn.parameter.Parameter, np.ndarray],
) -> np.ndarray:
    vecs = [vecs1, vecs2]
    for i in range(len(vecs)):
        if isinstance(vecs[i], torch.Tensor) or isinstance(vecs[i], torch.nn.parameter.Parameter):
            vecs[i] = vecs[i].detach().cpu().numpy()
    vecs1, vecs2 = vecs
    normalize = lambda v: (v.T / np.linalg.norm(v, axis=1)).T
    vecs1_norm = normalize(vecs1)
    vecs2_norm = normalize(vecs2)

    return vecs1_norm @ vecs2_norm.T


def mean_max_cosine_similarity(ground_truth_features, learned_dictionary, debug=False):
    # Calculate cosine similarity between all pairs of ground truth and learned features
    cos_sim = cosine_sim(ground_truth_features, learned_dictionary)
    # Find the maximum cosine similarity for each ground truth feature, then average
    mmcs = cos_sim.max(axis=1).mean()
    return mmcs


def get_n_dead_neurons(auto_encoder, data_generator, n_batches=10, device="cuda"):
    """
    :param result_dict: dictionary containing the results of a single run
    :return: number of dead neurons

    Estimates the number of dead neurons in the network by running a few batches of data through the network and
    calculating the mean activation of each neuron. If the mean activation is 0 for a neuron, it is considered dead.
    """
    outputs = []
    
    for batch_ndx, batch in enumerate(data_generator):
        input = batch.to(device).to(torch.float16)
        with torch.no_grad():
            x_hat, c = auto_encoder=(input)
        outputs.append(c)
        if batch_ndx >= n_batches:
            break
    outputs = torch.cat(outputs) # (n_batches * batch_size, n_dict_components)
    mean_activations = outputs.mean(dim=0) # (n_dict_components), c is after the ReLU, no need to take abs
    n_dead_neurons = (mean_activations == 0).sum().item()
    return n_dead_neurons

def analyse_result(result):
    get_n_dead_neurons(result)

def run_single_go(cfg: dotdict, data_generator: Optional[RandomDatasetGenerator]):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if not data_generator:
        data_generator = RandomDatasetGenerator(
            activation_dim=cfg.activation_dim,
            n_ground_truth_components=cfg.n_ground_truth_components,
            batch_size=cfg.batch_size,
            feature_num_nonzero=cfg.feature_num_nonzero,
            feature_prob_decay=cfg.feature_prob_decay,
            correlated=cfg.correlated_components,
            device=device,
        )

    auto_encoder = AutoEncoder(cfg.activation_dim, cfg.n_components_dictionary).to(device)

    ground_truth_features = data_generator.feats
    # Train the model
    optimizer = optim.Adam(auto_encoder.parameters(), lr=cfg.learning_rate)
    
    # Hold a running average of the reconstruction loss
    running_recon_loss = 0.0
    time_horizon = 10
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0

        batch = next(data_generator)
        batch = batch + cfg.noise_level * torch.randn_like(batch)

        optimizer.zero_grad()
        # Forward pass
        x_hat, c = auto_encoder(batch)
        # Compute the reconstruction loss and L1 regularization
        l_reconstruction = torch.nn.MSELoss()(batch, x_hat)
        l_l1 = cfg.l1_alpha * torch.norm(c,1, dim=1).mean() / c.size(1)
        # Compute the total loss
        loss = l_reconstruction + l_l1

        # Backward pass
        loss.backward()
        optimizer.step()

        # Add the loss for this batch to the total loss for this epoch
        epoch_loss += loss.item()
        running_recon_loss *= (time_horizon - 1) / time_horizon
        running_recon_loss += loss.item() / time_horizon

        if (epoch + 1) % 1000 == 0:
            # Calculate MMCS
            learned_dictionary = auto_encoder.decoder.weight.data.t()
            mmcs = mean_max_cosine_similarity(ground_truth_features.to(auto_encoder.device), learned_dictionary)
            print(f"Mean Max Cosine Similarity: {mmcs:.3f}")
            
            if(True):
                print(f"Epoch {epoch+1}/{cfg.epochs}: Reconstruction = {l_reconstruction:.6f} | l1: {l_l1:.6f}")
            
    learned_dictionary = auto_encoder.decoder.weight.data.t()
    mmcs = mean_max_cosine_similarity(ground_truth_features.to(auto_encoder.device), learned_dictionary)
    n_dead_neurons = get_n_dead_neurons(auto_encoder, data_generator)
    return mmcs, auto_encoder, n_dead_neurons, running_recon_loss


def plot_mat(mat, l1_alphas, learned_dict_ratios, show=True, save_folder=None, save_name=None, title=None):
    """
    :param mmsc_mat: matrix values
    :param l1_alphas: list of l1_alphas
    :param learned_dict_ratios: list of learned_dict_ratios
    :param show_plots: whether to show the plot
    :param save_path: path to save the plot
    :param title: title of the plot
    :return: None
    """
    assert mat.shape == (len(l1_alphas), len(learned_dict_ratios))
    mat = mat.T
    plt.imshow(mat, interpolation="nearest")
    # turn to str with 2 decimal places
    x_labels = [f"{l1_alpha:.2f}" for l1_alpha in l1_alphas]
    plt.xticks(range(len(x_labels)), x_labels)
    plt.xlabel("l1_alpha")
    y_labels = [str(learned_dict_ratio) for learned_dict_ratio in learned_dict_ratios]
    plt.yticks(range(len(y_labels)), y_labels)
    plt.ylabel("learned_dict_ratio")
    plt.colorbar()
    plt.set_cmap('viridis')
    # turn x labels 90 degrees
    plt.xticks(rotation=90)
    if title:
        plt.title(title)

    if show:
        plt.show()
    
    if save_folder:
        plt.savefig(os.path.join(save_folder, save_name))
        plt.close()
        
def compare_mmsc_with_larger_dicts(dict: np.array, larger_dicts: List[np.array]) -> float:
    """
    :param dict: The dict to compare to others. Shape (activation_dim, n_dict_elements)
    :param larger_dicts: A list of dicts to compare to. Shape (activation_dim, n_dict_elements(variable)]) * n_larger_dicts
    :return The mean max cosine similarity of the dict to the larger dicts

    Takes a dict, and for each element finds the most similar element in each of the larger dicts, takes the average
    Repeats this for all elements in the dict
    """
    n_larger_dicts = len(larger_dicts)
    n_elements = dict.shape[0]
    max_cosine_similarities = np.zeros((n_elements, n_larger_dicts))
    for elem_ndx in range(n_elements):
        element =  np.expand_dims(dict[elem_ndx], 0)
        for dict_ndx, larger_dict in enumerate(larger_dicts):
            cosine_sims = cosine_sim(element, larger_dict).squeeze()
            max_cosine_similarity = max(cosine_sims)
            max_cosine_similarities[elem_ndx, dict_ndx] = max_cosine_similarity
    mean_max_cosine_similarity = max_cosine_similarities.mean()
    return mean_max_cosine_similarity

def recalculate_results(auto_encoder, data_generator):
    """Take a fully trained auto_encoder and a data_generator and return the results of the auto_encoder on the data_generator"""
    time_horizon = 10
    recon_loss = 0
    for epoch in range(time_horizon):
        # Get a batch of data
        batch = data_generator.get_batch()
        batch = torch.from_numpy(batch).to(auto_encoder.device)

        # Forward pass
        x_hat, feat_levels = auto_encoder(batch)

        # Compute the reconstruction loss
        l_reconstruction = torch.norm(x_hat - batch, 2, dim=1).sum() / batch.size(1)

        # Add the loss for this batch to the total loss for this epoch
        recon_loss += l_reconstruction.item() / time_horizon

    ground_truth_features = data_generator.feats
    learned_dictionary = auto_encoder.decoder.weight.data.t()
    mmcs = mean_max_cosine_similarity(ground_truth_features.to(auto_encoder.device), learned_dictionary)   
    n_dead_neurons = get_n_dead_neurons(auto_encoder, data_generator)
    return mmcs, learned_dictionary, n_dead_neurons, recon_loss

def run_toy_model(cfg):
    # Using a single data generator for all runs so that can compare learned dicts
    data_generator = RandomDatasetGenerator(
        activation_dim=cfg.activation_dim,
        n_ground_truth_components=cfg.n_ground_truth_components,
        batch_size=cfg.batch_size,
        feature_num_nonzero=cfg.feature_num_nonzero,
        feature_prob_decay=cfg.feature_prob_decay,
        correlated=cfg.correlated_components,
        device=cfg.device,
    )

    l1_range = [10 ** (exp/4) for exp in range(cfg.l1_exp_low, cfg.l1_exp_high)] # replicate is (-8,9)
    learned_dict_ratios = [2 ** exp for exp in range(cfg.dict_ratio_exp_low, cfg.dict_ratio_exp_high)] # replicate is (-2,6)
    print("Range of l1 values being used: ", l1_range)
    print("Range of dict_sizes compared to ground truth being used:",  learned_dict_ratios)
    mmsc_matrix = np.zeros((len(l1_range), len(learned_dict_ratios)))
    dead_neurons_matrix = np.zeros((len(l1_range), len(learned_dict_ratios)))
    recon_loss_matrix = np.zeros((len(l1_range), len(learned_dict_ratios)))


    # 2D array of learned dictionaries, indexed by l1_alpha and learned_dict_ratio, start with Nones
    auto_encoders = [[None for _ in range(len(learned_dict_ratios))] for _ in range(len(l1_range))]
    learned_dicts = [[None for _ in range(len(learned_dict_ratios))] for _ in range(len(l1_range))]

    for l1_alpha, learned_dict_ratio in tqdm(list(itertools.product(l1_range, learned_dict_ratios))):
        cfg.l1_alpha = l1_alpha
        cfg.learned_dict_ratio = learned_dict_ratio
        cfg.n_components_dictionary = int(cfg.n_ground_truth_components * cfg.learned_dict_ratio)
        mmsc, auto_encoder, n_dead_neurons, reconstruction_loss = run_single_go(cfg, data_generator)
        print(f"l1_alpha: {l1_alpha} | learned_dict_ratio: {learned_dict_ratio} | mmsc: {mmsc:.3f} | n_dead_neurons: {n_dead_neurons} | reconstruction_loss: {reconstruction_loss:.3f}")

        mmsc_matrix[l1_range.index(l1_alpha), learned_dict_ratios.index(learned_dict_ratio)] = mmsc
        dead_neurons_matrix[l1_range.index(l1_alpha), learned_dict_ratios.index(learned_dict_ratio)] = n_dead_neurons
        recon_loss_matrix[l1_range.index(l1_alpha), learned_dict_ratios.index(learned_dict_ratio)] = reconstruction_loss
        auto_encoders[l1_range.index(l1_alpha)][learned_dict_ratios.index(learned_dict_ratio)] = auto_encoder.cpu()
        learned_dicts[l1_range.index(l1_alpha)][learned_dict_ratios.index(learned_dict_ratio)] = auto_encoder.decoder.weight.detach().cpu().data.t()
    
    outputs_folder = f"outputs_{cfg.model_name}"
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    outputs_folder = os.path.join(outputs_folder, current_time)
    os.makedirs(outputs_folder, exist_ok=True)

    # Save the matrices and the data generator
    plot_mat(mmsc_matrix, l1_range, learned_dict_ratios, show=False, save_folder=outputs_folder, title="Mean Max Cosine Similarity w/ True", save_name="mmsc_matrix.png")
    # clamp dead_neurons to 0-100 for better visualisation
    dead_neurons_matrix = np.clip(dead_neurons_matrix, 0, 100)
    plot_mat(dead_neurons_matrix, l1_range, learned_dict_ratios, show=False, save_folder=outputs_folder, title="Dead Neurons", save_name="dead_neurons_matrix.png")
    plot_mat(recon_loss_matrix, l1_range, learned_dict_ratios, show=False, save_folder=outputs_folder, title="Reconstruction Loss", save_name="recon_loss_matrix.png")
    with open(os.path.join(outputs_folder, "auto_encoders.pkl"), "wb") as f:
        pickle.dump(auto_encoders, f)
    with open(os.path.join(outputs_folder, "config.pkl"), "wb") as f:
        pickle.dump(cfg, f)
    with open(os.path.join(outputs_folder, "data_generator.pkl"), "wb") as f:
        pickle.dump(data_generator, f)
    with open(os.path.join(outputs_folder, "mmsc_matrix.pkl"), "wb") as f:
        pickle.dump(mmsc_matrix, f)
    with open(os.path.join(outputs_folder, "dead_neurons.pkl"), "wb") as f:
        pickle.dump(dead_neurons_matrix, f)
    with open(os.path.join(outputs_folder, "recon_loss.pkl"), "wb") as f:
        pickle.dump(recon_loss_matrix, f)

    # Compare each learned dictionary to the larger ones
    av_mmsc_with_larger_dicts = np.zeros((len(l1_range), len(learned_dict_ratios)))
    percentage_above_threshold_mmsc_with_larger_dicts = np.zeros((len(l1_range), len(learned_dict_ratios)))
    for l1_ndx, dict_size_ndx in tqdm(list(itertools.product(range(len(l1_range)), range(len(learned_dict_ratios))))):
        if dict_size_ndx == len(learned_dict_ratios) - 1:
            continue
        smaller_dict = learned_dicts[l1_ndx][dict_size_ndx]
        larger_dict = learned_dicts[l1_ndx][dict_size_ndx+1]
        smaller_dict_features, shared_vector_size = smaller_dict.shape
        # larger_dict_features, shared_vector_size = larger_dict.shape
        max_cosine_similarities = np.zeros((smaller_dict_features))
        larger_dict = larger_dict.to(cfg.device)
        for i, vector in enumerate(smaller_dict):
            max_cosine_similarities[i] = torch.nn.functional.cosine_similarity(vector.to(cfg.device), larger_dict, dim=1).max()
        av_mmsc_with_larger_dicts[l1_ndx, dict_size_ndx] = max_cosine_similarities.mean().item()
        threshold = 0.9
        percentage_above_threshold_mmsc_with_larger_dicts[l1_ndx, dict_size_ndx] = (max_cosine_similarities > threshold).sum().item()

    with open(os.path.join(outputs_folder, "larger_dict_compare.pkl"), "wb") as f:
        pickle.dump(av_mmsc_with_larger_dicts, f)
    with open(os.path.join(outputs_folder, "larger_dict_threshold.pkl"), "wb") as f:
        pickle.dump(percentage_above_threshold_mmsc_with_larger_dicts, f)
    
    plot_mat(av_mmsc_with_larger_dicts, l1_range, learned_dict_ratios, show=False, save_folder=outputs_folder, title="Average MMSC with larger dicts", save_name="av_mmsc_with_larger_dicts.png")
    plot_mat(percentage_above_threshold_mmsc_with_larger_dicts, l1_range, learned_dict_ratios, show=False, save_folder=outputs_folder, title=f"MMSC with larger dicts above {threshold}", save_name="percentage_above_threshold_mmsc_with_larger_dicts.png")



def run_single_go_with_real_data(cfg, dataset_folder: str):
    auto_encoder = AutoEncoder(cfg.activation_dim, cfg.n_components_dictionary).to(cfg.device)
    optimizer = optim.Adam(auto_encoder.parameters(), lr=cfg.learning_rate, eps=1e-4)
    running_recon_loss = 0.0
    time_horizon = 1000
    # torch.autograd.set_detect_anomaly(True)
    n_chunks_in_folder = len(os.listdir(dataset_folder))
    
    for epoch in range(cfg.epochs):
        chunk_order = np.random.permutation(n_chunks_in_folder)
        for chunk_ndx, chunk_id in enumerate(chunk_order):
            chunk_loc = os.path.join(dataset_folder, f"{chunk_id}.pkl")
            dataset = DataLoader(pickle.load(open(chunk_loc, "rb")), batch_size=cfg.batch_size, shuffle=True)
            for batch_idx, batch in enumerate(dataset):
                batch = batch[0].to(cfg.device)
                optimizer.zero_grad()
                # Run through auto_encoder

                x_hat, dict_levels = auto_encoder(batch)
                l_reconstruction = torch.nn.MSELoss()(batch, x_hat)
                l_l1 = cfg.l1_alpha * torch.norm(dict_levels, 1, dim=1).mean() / dict_levels.size(1)
                loss = l_reconstruction + l_l1
                loss.backward()
                optimizer.step()
                
                if running_recon_loss == 0.0:
                    running_recon_loss = loss.item()
                else:
                    running_recon_loss *= (time_horizon - 1) / time_horizon
                    running_recon_loss += loss.item() / time_horizon
<<<<<<< HEAD
                if (batch_idx + 1) % 10 == 0:
                    print(f"Batch: {batch_idx+1}/{len(dataset)} | Chunk: {chunk_ndx+1}/{n_chunks_in_folder} | Epoch: {epoch+1}/{cfg.epochs} | Reconstruction loss: {running_recon_loss:.6f} | l1: {l_l1:.6f}")
=======
                if (batch_idx + 1) % 100 == 0:
                    print(f"L1 Coef: {cfg.l1_alpha:.3f} | Dict ratio: {cfg.n_components_dictionary / cfg.activation_dim} | " + \
                            f"Batch: {batch_idx+1}/{len(dataset)} | Chunk: {chunk_ndx+1}/{n_chunks_in_folder} | " + \
                            f"Epoch: {epoch+1}/{cfg.epochs} | Reconstruction loss: {running_recon_loss:.6f} | l1: {l_l1:.6f}")
>>>>>>> e01c27c (Improve alpha format.)
            
    n_dead_neurons = get_n_dead_neurons(auto_encoder, dataset)
    return auto_encoder, n_dead_neurons, running_recon_loss

def make_activation_dataset(cfg, sentence_dataset: DataLoader, model: HookedTransformer, tensor_name: str, dataset_folder: str, baukit: bool = False):
    print(f"Running model and saving activations to {dataset_folder}")
    with torch.no_grad():
        chunk_size =  2 * (2 ** 30) # 2GB
        activation_size = cfg.mlp_width * 2 * cfg.model_batch_size * cfg.max_length # 3072 mlp activations, 2 bytes per half, 1024 context window
        max_chunks = chunk_size // activation_size
        dataset = []
        n_saved_chunks = 0
        for batch_idx, batch in enumerate(sentence_dataset):
            batch = batch["input_ids"].to(cfg.device)
            if baukit:
                # Don't have nanoGPT models integrated with transformer_lens so using baukit for activations
                with Trace(model, tensor_name) as ret:
                    _ = model(batch)
                    mlp_activation_data = ret.output
                    mlp_activation_data = rearrange(mlp_activation_data, 'b s n -> (b s) n').to(torch.float16)
                    # breakpoint()
            else:       
                _, cache = model.run_with_cache(batch)
                mlp_activation_data = cache[tensor_name].to(cfg.device).to(torch.float16) # NOTE: could do all layers at once, but currently just doing 1 layer
                mlp_activation_data = rearrange(mlp_activation_data, 'b s n -> (b s) n')
            print(batch_idx, batch_idx * activation_size)
            dataset.append(mlp_activation_data)
            if len(dataset) >= max_chunks:
                # Need to save, restart the list
                dataset = torch.cat(dataset, dim=0).to("cpu")
                dataset = torch.utils.data.TensorDataset(dataset)
                with open(dataset_folder + "/" + str(n_saved_chunks) + ".pkl", "wb") as f:
                    pickle.dump(dataset, f)
                n_saved_chunks += 1
                print(f"Saved chunk {n_saved_chunks} of activations")
                dataset = []
                if n_saved_chunks == cfg.n_chunks:
                    break

def run_real_data_model(cfg):
    # cfg.model_name = "EleutherAI/pythia-70m-deduped"
    if cfg.model_name in ["gpt2", "EleutherAI/pythia-70m-deduped"]:
        model = HookedTransformer.from_pretrained(cfg.model_name, device=cfg.device)
        use_baukit = False
    elif cfg.model_name == "nanoGPT":
        model_dict = torch.load(open(cfg.model_path, "rb"), map_location="cpu")["model"]
        model_dict = {k.replace("_orig_mod.", ""): v for k, v in model_dict.items()}
        cfg_loc = cfg.model_path[:-3] + "cfg" # cfg loc is same as model_loc but with .pt replaced with cfg.py
        cfg_loc = cfg_loc.replace("/", ".")
        model_cfg = importlib.import_module(cfg_loc).model_cfg
        model = GPT(model_cfg).to(cfg.device)
        model.load_state_dict(model_dict)
        use_baukit = True
    else:
        raise ValueError("Model name not recognised")

    sentence_dataset = load_dataset(cfg.dataset_name, split="train")
    if hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer
    else:
        print("Using default tokenizer from gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    sentence_dataset, bits_per_byte = chunk_and_tokenize(sentence_dataset, tokenizer, max_length=cfg.max_length)
    sentence_dataset = DataLoader(sentence_dataset, batch_size=cfg.model_batch_size, shuffle=True)

    # Check if we have already run this model and got the activations
    dataset_name = cfg.dataset_name.split("/")[-1] + "-" + cfg.model_name + "-" + str(cfg.layer)
    dataset_folder = os.path.join(cfg.datasets_folder, dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)
    if len(os.listdir(dataset_folder)) == 0:
        if cfg.model_name in ["gpt2", "EleutherAI/pythia-70m-deduped"]":
            tensor_name = f"blocks.{cfg.layer}.mlp.hook_post"
            cfg.mlp_width = 3072
        elif cfg.model_name == "nanoGPT":
            tensor_name = f"transformer.h.{cfg.layer}.mlp.c_fc"
            cfg.mlp_width = 128
        else:
            raise NotImplementedError(f"Model {cfg.model_name} not supported")
        make_activation_dataset(cfg, sentence_dataset, model, tensor_name, dataset_folder, use_baukit)
    else:
        print(f"Activations in {dataset_folder} already exist, loading them")
        # get mlp_width from first file
        with open(os.path.join(dataset_folder, "0.pkl"), "rb") as f:
            dataset = pickle.load(f)
        cfg.mlp_width = dataset.tensors[0][0].shape[-1]
        del dataset

    l1_range = [1.0]
    dict_sizes = [cfg.mlp_width * (2 ** i) for i in range(1, 7)]
    print("Range of l1 values being used: ", l1_range)
    print("Range of dict_sizes being used:",  dict_sizes)
    dead_neurons_matrix = np.zeros((len(l1_range), len(dict_sizes)))
    recon_loss_matrix = np.zeros((len(l1_range), len(dict_sizes)))

    # 2D array of learned dictionaries, indexed by l1_alpha and learned_dict_ratio, start with Nones
    auto_encoders = [[None for _ in range(len(dict_sizes))] for _ in range(len(l1_range))]
    learned_dicts = [[None for _ in range(len(dict_sizes))] for _ in range(len(l1_range))]

    for l1_ndx, dict_size_ndx in tqdm(list(itertools.product(range(len(l1_range)), range(len(dict_sizes))))):
        l1_loss = l1_range[l1_ndx]
        dict_size = dict_sizes[dict_size_ndx]

        cfg.l1_alpha = l1_loss
        cfg.n_components_dictionary = dict_size
        auto_encoder, n_dead_neurons, reconstruction_loss = run_single_go_with_real_data(cfg, dataset_folder)
        print(f"l1: {l1_loss} | dict_size: {dict_size} | n_dead_neurons: {n_dead_neurons} | reconstruction_loss: {reconstruction_loss:.3f}")

        dead_neurons_matrix[l1_ndx, dict_size_ndx] = n_dead_neurons
        recon_loss_matrix[l1_ndx, dict_size_ndx] = reconstruction_loss
        auto_encoders[l1_ndx][dict_size_ndx] = auto_encoder.cpu()
        learned_dicts[l1_ndx][dict_size_ndx] = auto_encoder.decoder.weight.detach().cpu().data.t()
    
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    outputs_folder = os.path.join(cfg.outputs_folder, current_time)
    os.makedirs(outputs_folder, exist_ok=True)

    # clamp dead_neurons to 0-100 for better visualisation
    dead_neurons_matrix = np.clip(dead_neurons_matrix, 0, 100)
    plot_mat(dead_neurons_matrix, l1_range, dict_sizes, show=False, save_folder=outputs_folder, title="Dead Neurons", save_name="dead_neurons_matrix.png")
    plot_mat(recon_loss_matrix, l1_range, dict_sizes, show=False, save_folder=outputs_folder, title="Reconstruction Loss", save_name="recon_loss_matrix.png")
    with open(os.path.join(outputs_folder, "auto_encoders.pkl"), "wb") as f:
        pickle.dump(auto_encoders, f)
    with open(os.path.join(outputs_folder, "config.pkl"), "wb") as f:
        pickle.dump(cfg, f)
    with open(os.path.join(outputs_folder, "dead_neurons.pkl"), "wb") as f:
        pickle.dump(dead_neurons_matrix, f)
    with open(os.path.join(outputs_folder, "recon_loss.pkl"), "wb") as f:
        pickle.dump(recon_loss_matrix, f)

    # Compare each learned dictionary to the larger ones
    av_mmsc_with_larger_dicts = np.zeros((len(l1_range), len(dict_sizes)))
    percentage_above_threshold_mmsc_with_larger_dicts = np.zeros((len(l1_range), len(dict_sizes)))
    for l1_ndx, dict_size_ndx in tqdm(list(itertools.product(range(len(l1_range)), range(len(dict_sizes))))):
        l1_loss = l1_range[l1_ndx]
        dict_size = dict_sizes[dict_size_ndx]
        if dict_size_ndx == len(dict_sizes) - 1:
            continue
        smaller_dict = learned_dicts[l1_ndx][dict_size_ndx]
        larger_dict = learned_dicts[l1_ndx][dict_size_ndx+1]
        smaller_dict_features, shared_vector_size = smaller_dict.shape
        # larger_dict_features, shared_vector_size = larger_dict.shape
        max_cosine_similarities = np.zeros((smaller_dict_features))
        larger_dict = larger_dict.to(cfg.device)
        for i, vector in enumerate(smaller_dict):
            max_cosine_similarities[i] = torch.nn.functional.cosine_similarity(vector.to(cfg.device), larger_dict, dim=1).max()
        av_mmsc_with_larger_dicts[l1_ndx, dict_size_ndx] = max_cosine_similarities.mean().item()
        threshold = 0.9
        percentage_above_threshold_mmsc_with_larger_dicts[l1_ndx, dict_size_ndx] = (max_cosine_similarities > threshold).sum().item()

    with open(os.path.join(outputs_folder, "larger_dict_compare.pkl"), "wb") as f:
        pickle.dump(av_mmsc_with_larger_dicts, f)
    with open(os.path.join(outputs_folder, "larger_dict_threshold.pkl"), "wb") as f:
        pickle.dump(percentage_above_threshold_mmsc_with_larger_dicts, f)
    
    plot_mat(av_mmsc_with_larger_dicts, l1_range, dict_sizes, show=False, save_folder=outputs_folder, title="Average MMSC with larger dicts", save_name="av_mmsc_with_larger_dicts.png")
    plot_mat(percentage_above_threshold_mmsc_with_larger_dicts, l1_range, dict_sizes, show=False, save_folder=outputs_folder, title=f"MMSC with larger dicts above {threshold}", save_name="percentage_above_threshold_mmsc_with_larger_dicts.png")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation_dim", type=int, default=256)
    parser.add_argument("--n_ground_truth_components", type=int, default=512)
    parser.add_argument("--learned_dict_ratio", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=256) # when tokenizing, truncate to this length, basically the context size
    
    parser.add_argument("--model_batch_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--l1_alpha", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--noise_level", type=float, default=0.0)

    parser.add_argument("--feature_prob_decay", type=float, default=0.99)
    parser.add_argument("--feature_num_nonzero", type=int, default=5)
    parser.add_argument("--correlated_components", type=bool, default=True)

    parser.add_argument("--l1_exp_low", type=int, default=-8)
    parser.add_argument("--l1_exp_high", type=int, default=9) # not inclusive
    parser.add_argument("--dict_ratio_exp_low", type=int, default=-3)
    parser.add_argument("--dict_ratio_exp_high", type=int, default=6) # not inclusive

    parser.add_argument("--run_toy", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default="nanoGPT")
    parser.add_argument("--model_path", type=str, default="models/32d70k.pt")
    parser.add_argument("--dataset_name", type=str, default="NeelNanda/pile-10k")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer", type=int, default=2) # layer to extract mlp-post-non-lin features from, only if using real model

    parser.add_argument("--outputs_folder", type=str, default="outputs")
    parser.add_argument("--datasets_folder", type=str, default="datasets")
    parser.add_argument("--n_chunks", type=int, default=400)

    args = parser.parse_args()
    cfg = dotdict(vars(args)) # convert to dotdict via dict
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.run_toy:
        run_toy_model(cfg)
    else:
        run_real_data_model(cfg)

if __name__ == "__main__":
    main()