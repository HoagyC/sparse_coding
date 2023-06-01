from collections.abc import Generator
import copy
from dataclasses import dataclass, field
import datetime
from functools import partial
import itertools
import multiprocessing as mp
import os
import pickle
from typing import Union, Tuple, List, Dict, Any, Iterator, Callable, Optional

from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, multivariate_normal  # type: ignore

import torch
import torch.nn as nn
import torch.optim as optim
from torchtyping import TensorType
from tqdm import tqdm

from utils import dotdict

n_ground_truth_components, activation_dim, dataset_size = None, None, None

@dataclass
class RandomDatasetGenerator(Generator):
    activation_dim: int
    n_ground_truth_components: int
    batch_size: int
    feature_num_nonzero: int
    feature_prob_decay: float
    runs_share_feats: bool
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
                self.n_ground_truth_components, self.runs_share_feats, device=self.device
            )
        else:
            self.component_probs = (
                self.decay * self.frac_nonzero
            )  # Only if non-correlated
        self.feats = generate_rand_feats(
            self.activation_dim,
            self.n_ground_truth_components,
            runs_share_feats=self.runs_share_feats,
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
        return data

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
    runs_share_feats: bool,
    device: Union[torch.device, str],
) -> TensorType['n_ground_truth_components', 'activation_dim']:
    data_path = os.path.join(os.getcwd(), "data")
    data_filename = os.path.join(data_path, f"feats_{feat_dim}_{num_feats}.npy")

    feats = np.random.multivariate_normal(
        np.zeros(feat_dim), np.eye(feat_dim), size=num_feats
    )
    feats = feats.T / np.linalg.norm(feats, axis=1)
    feats = feats.T
    if runs_share_feats:
        if os.path.exists(data_filename):
            feats = np.load(data_filename)
        else:
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            np.save(data_filename, feats)

    feats_tensor = torch.from_numpy(feats).to(device).float()
    return feats_tensor


def generate_corr_matrix(
    num_feats: int, runs_share_feats: bool, device: Union[torch.device, str]
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

    if runs_share_feats:
        if os.path.exists(corr_mat_filename):
            corr_matrix = np.load(corr_mat_filename)
        else:
            if not os.path.exists(corr_mat_path):
                os.makedirs(corr_mat_path)
            np.save(corr_mat_filename, corr_matrix)
    corr_matrix_tensor = torch.from_numpy(corr_matrix).to(device).float()

    return corr_matrix_tensor


# AutoEncoder Definition
class AutoEncoder(nn.Module):
    def __init__(self, activation_size, n_dict_components):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(activation_size, n_dict_components),
            nn.ReLU()
        )
        self.decoder = nn.Linear(n_dict_components, activation_size, bias=False)

        # Initialize the decoder weights orthogonally
        nn.init.orthogonal_(self.decoder.weight)
        
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
    vecs1: Union[torch.Tensor, torch.nn.parameter.Parameter],
    vecs2: Union[torch.Tensor, torch.nn.parameter.Parameter],
) -> np.ndarray:
    vecs = [vecs1, vecs2]
    for i in range(len(vecs)):
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


def get_n_dead_neurons(auto_encoder, data_generator, n_batches=10):
    """
    :param result_dict: dictionary containing the results of a single run
    :return: number of dead neurons

    Estimates the number of dead neurons in the network by running a few batches of data through the network and
    calculating the mean activation of each neuron. If the mean activation is 0 for a neuron, it is considered dead.
    """
    outputs = []
    for i in range(n_batches):
        batch = next(data_generator)
        x_hat, c = auto_encoder(batch)
        outputs.append(c)
    outputs = torch.cat(outputs)
    mean_activations = outputs.mean(dim=0)  
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
            correlated=True,
            runs_share_feats=False,
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

        # for batch_index in range(dataset_size // batch_size):
            # Generate a batch of samples
            # batch = final_dataset[batch_index*batch_size:(batch_index+1)*batch_size].to(device)
            # batch = create_dataset(ground_truth_features, probabilities, batch_size).float().to(device)
        batch = next(data_generator)
        batch = batch + cfg.noise_level * torch.randn_like(batch)


        optimizer.zero_grad()

        # Forward pass
        x_hat, c = auto_encoder(batch)
        
        # Compute the reconstruction loss and L1 regularization
        l_reconstruction = torch.nn.MSELoss()(batch, x_hat)
        l_l1 = cfg.l1_alpha * torch.norm(c,1, dim=1).mean() / c.size(1)
        # l_l1 = l1_alpha * torch.norm(c,1, dim=1).sum() / c.size(1)

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

            # Compute the average loss for this epoch
            # epoch_loss /= (dataset_size // batch_size)
            # debug_sparsity_of_c(auto_encoder, ground_truth_features, probabilities, batch_size)
            
            if(True):
                print(f"Epoch {epoch+1}/{cfg.epochs}: Reconstruction = {l_reconstruction:.6f} | l1: {l_l1:.6f}")
            
    # debug_sparsity_of_c(auto_encoder, ground_truth_features, probabilities, batch_size)

    learned_dictionary = auto_encoder.decoder.weight.data.t()
    mmcs = mean_max_cosine_similarity(ground_truth_features.to(auto_encoder.device), learned_dictionary)
    n_dead_neurons = get_n_dead_neurons(auto_encoder, data_generator)
    return mmcs, learned_dictionary, n_dead_neurons, running_recon_loss


def plot_mmsc_mat(mmsc_mat, l1_alphas, learned_dict_ratios, show_plots=True, save_path=None, title=None):
    """
    :param mmsc_mat: matrix of MMCS values
    :param l1_alphas: list of l1_alphas
    :param learned_dict_ratios: list of learned_dict_ratios
    :param show_plots: whether to show the plot
    :param save_path: path to save the plot
    :param title: title of the plot
    :return: None
    """
    mmsc_mat = mmsc_mat.T
    assert mmsc_mat.shape == (len(l1_alphas), len(learned_dict_ratios))
    plt.imshow(mmsc_mat, interpolation="nearest")
    x_labels = [str(l1_alpha) for l1_alpha in l1_alphas]
    plt.xticks(range(len(x_labels)), x_labels)
    plt.xlabel("l1_alpha")
    y_labels = [str(learned_dict_ratio) for learned_dict_ratio in learned_dict_ratios]
    plt.yticks(range(len(y_labels)), y_labels)
    plt.ylabel("learned_dict_ratio")
    plt.colorbar()
    plt.set_cmap('viridis')
    if show_plots:
        plt.show()
    
    if save_path:
        plt.savefig(os.path.join(save_path, title))
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
    max_cosine_similarities = np.zeros(n_elements, n_larger_dicts)
    for elem_ndx in range(n_elements):
        element = dict[elem_ndx]
        for dict_ndx, larger_dict in enumerate(larger_dicts):
            cosine_sims = cosine_sim(element.unsqueeze(0), larger_dict)
            max_cosine_similarity = max(cosine_sims)
            max_cosine_similarities[elem_ndx, dict_ndx] = max_cosine_similarity
    mean_max_cosine_similarity = max_cosine_similarities.mean()
    return mean_max_cosine_similarity

def main():
    cfg = dotdict()
    cfg.activation_dim = 256
    cfg.n_ground_truth_components = cfg.activation_dim * 2
    cfg.learned_dict_ratio = 1.0
    cfg.n_components_dictionary = int(cfg.n_ground_truth_components * cfg.learned_dict_ratio)

    cfg.batch_size = 256
    cfg.noise_std = 0.1
    cfg.l1_alpha = 0.1
    cfg.learning_rate=0.001
    cfg.epochs = 8000
    cfg.noise_level = 0.0

    cfg.feature_prob_decay = 0.99
    cfg.feature_num_nonzero = 5
    cfg.correlated_components = False

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    l1_range = [10 ** (exp/4) for exp in range(-8, 9)] # replicate is (-8,9)
    learned_dict_ratios = [2 ** exp for exp in range(-2, 6)] # replicate is (-2,6)
    print("Range of l1 values being used: ", l1_range)
    print("Range of dict_sizes compared to ground truth being used:",  learned_dict_ratios)
    mmsc_matrix = np.zeros((len(l1_range), len(learned_dict_ratios)))
    dead_neurons_matrix = np.zeros((len(l1_range), len(learned_dict_ratios)))
    recon_loss_matrix = np.zeros((len(l1_range), len(learned_dict_ratios)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Using a single data generator for all runs so that can compare learned dicts
    data_generator = RandomDatasetGenerator(
        activation_dim=cfg.activation_dim,
        n_ground_truth_components=cfg.n_ground_truth_components,
        batch_size=cfg.batch_size,
        feature_num_nonzero=cfg.feature_num_nonzero,
        feature_prob_decay=cfg.feature_prob_decay,
        correlated=True,
        runs_share_feats=False,
        device=device,
    )

    # 2D array of learned dictionaries, indexed by l1_alpha and learned_dict_ratio, start with Nones
    learned_dicts = [[None for _ in range(len(learned_dict_ratios))] for _ in range(len(l1_range))]


    for l1_alpha, learned_dict_ratio in tqdm(list(itertools.product(l1_range, learned_dict_ratios))):
        cfg.l1_alpha = l1_alpha
        cfg.learned_dict_ratio = learned_dict_ratio
        cfg.n_components_dictionary = int(cfg.n_ground_truth_components * cfg.learned_dict_ratio)
        mmsc, learned_dict, n_dead_neurons, reconstruction_loss = run_single_go(cfg, data_generator)
        print(f"l1_alpha: {l1_alpha} | learned_dict_ratio: {learned_dict_ratio} | mmsc: {mmsc:.3f} | n_dead_neurons: {n_dead_neurons} | reconstruction_loss: {reconstruction_loss:.3f}")

        mmsc_matrix[l1_range.index(l1_alpha), learned_dict_ratios.index(learned_dict_ratio)] = mmsc
        dead_neurons_matrix[l1_range.index(l1_alpha), learned_dict_ratios.index(learned_dict_ratio)] = n_dead_neurons
        recon_loss_matrix[l1_range.index(l1_alpha), learned_dict_ratios.index(learned_dict_ratio)] = reconstruction_loss
        learned_dicts[l1_range.index(l1_alpha)][learned_dict_ratios.index(learned_dict_ratio)] = learned_dict.cpu().numpy()
    
    outputs_folder = "outputs"
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    outputs_folder = os.path.join(outputs_folder, current_time)
    os.makedirs(outputs_folder, exist_ok=True)

    # Save the matrices and the data generator
    plot_mmsc_mat(mmsc_matrix, l1_range, learned_dict_ratios, show=False, save_folder=outputs_folder, title="mmsc_matrix.png")
    plot_mmsc_mat(dead_neurons_matrix, l1_range, learned_dict_ratios, show=False, save_folder=outputs_folder, title="dead_neurons_matrix.png")
    plot_mmsc_mat(recon_loss_matrix, l1_range, learned_dict_ratios, show=False, save_folder=outputs_folder, title="recon_loss_matrix.png")
    with open(os.path.join(outputs_folder, "learned_dicts.pkl"), "wb") as f:
        pickle.dump(learned_dicts, f)
    with open(os.path.join(outputs_folder, "data_generator.pkl"), "wb") as f:
        pickle.dump(data_generator, f)

    # Compare each learned dictionary to the larger ones
    av_mmsc_with_larger_dicts = np.zeros((len(l1_range), len(learned_dict_ratios)))
    for l1_alpha, learned_dict_ratio in tqdm(list(itertools.product(l1_range, learned_dict_ratios))):
        learned_dict = learned_dicts[l1_range.index(l1_alpha)][learned_dict_ratios.index(learned_dict_ratio)]
        larger_dicts = [learned_dicts[l1_range.index(l1_alpha)][learned_dict_ratios.index(learned_dict_ratio)] for learned_dict_ratio in learned_dict_ratios if learned_dict_ratio > learned_dict_ratio]
        mean_max_cosine_similarity = compare_mmsc_with_larger_dicts(learned_dict, larger_dicts)
        av_mmsc_with_larger_dicts[l1_range.index(l1_alpha), learned_dict_ratios.index(learned_dict_ratio)] = mean_max_cosine_similarity
    
    plot_mmsc_mat(av_mmsc_with_larger_dicts, l1_range, learned_dict_ratios, show=False, save_folder=outputs_folder, title="av_mmsc_with_larger_dicts")

if __name__ == "__main__":
    main()