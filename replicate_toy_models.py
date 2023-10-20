"""
This file is a frozen verison of toy_models.py. No changes should be made to this file except bug-fixes and explanations.
Running this file should replicate the toy models section of the post:
https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition
"""

import argparse
import itertools
import os
import pickle
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchtyping import TensorType
from tqdm import tqdm

from config import ToyArgs

n_ground_truth_components, activation_dim, dataset_size = None, None, None # type: Tuple[None, None, None]


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
    decay: TensorType["n_ground_truth_components"] = field(init=False)
    feats: TensorType["n_ground_truth_components", "activation_dim"] = field(init=False)
    corr_matrix: Optional[TensorType["n_ground_truth_components", "n_ground_truth_components"]] = field(init=False)
    component_probs: Optional[TensorType["n_ground_truth_components"]] = field(init=False)

    def __post_init__(self):
        self.frac_nonzero = self.feature_num_nonzero / self.n_ground_truth_components

        # Define the probabilities of each component being included in the data
        self.decay = torch.tensor([self.feature_prob_decay**i for i in range(self.n_ground_truth_components)]).to(
            self.device
        )  # FIXME: 1 / i

        if self.correlated:
            self.corr_matrix = generate_corr_matrix(self.n_ground_truth_components, device=self.device)
        else:
            self.component_probs = self.decay * self.frac_nonzero  # Only if non-correlated
        self.feats = generate_rand_feats(
            self.activation_dim,
            self.n_ground_truth_components,
            device=self.device,
        )

    def send(self, ignored_arg: Any) -> TensorType["dataset_size", "activation_dim"]:
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
    n_ground_truth_components: int,  #
    dataset_size: int,
    feature_probs: TensorType["n_ground_truth_components"],
    feats: TensorType["n_ground_truth_components", "activation_dim"],
    device: Union[torch.device, str],
) -> Tuple[
    TensorType["n_ground_truth_components", "activation_dim"],
    TensorType["dataset_size", "n_ground_truth_components"],
    TensorType["dataset_size", "activation_dim"],
]:
    # generate random feature strengths
    feature_strengths = torch.rand((dataset_size, n_ground_truth_components), device=device)
    # only some features are activated, chosen at random
    dataset_thresh = torch.rand(dataset_size, n_ground_truth_components, device=device)
    data_zero = torch.zeros_like(dataset_thresh, device=device)

    dataset_codes = torch.where(
        dataset_thresh <= feature_probs,
        feature_strengths,
        data_zero,
    )  # dim: dataset_size x n_ground_truth_components

    dataset = dataset_codes @ feats

    return feats, dataset_codes, dataset


def generate_correlated_dataset(
    n_ground_truth_components: int,
    dataset_size: int,
    corr_matrix: TensorType["n_ground_truth_components", "n_ground_truth_components"],
    feats: TensorType["n_ground_truth_components", "activation_dim"],
    frac_nonzero: float,
    decay: TensorType["n_ground_truth_components"],
    device: Union[torch.device, str],
) -> Tuple[
    TensorType["n_ground_truth_components", "activation_dim"],
    TensorType["dataset_size", "n_ground_truth_components"],
    TensorType["dataset_size", "activation_dim"],
]:
    # Get a correlated gaussian sample
    mvn = torch.distributions.MultivariateNormal(
        loc=torch.zeros(n_ground_truth_components, device=device),
        covariance_matrix=corr_matrix,
    )
    corr_thresh = mvn.sample()

    # Take the CDF of that sample.
    normal = torch.distributions.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
    cdf = normal.cdf(corr_thresh.squeeze())

    # Decay it
    component_probs = cdf * decay

    # Scale it to get the right % of nonzeros
    mean_prob = torch.mean(component_probs)
    scaler = frac_nonzero / mean_prob
    component_probs *= scaler
    # So np.isclose(np.mean(component_probs), frac_nonzero) will be True

    # generate random feature strengths
    feature_strengths = torch.rand((dataset_size, n_ground_truth_components), device=device)
    data_zero = torch.zeros_like(corr_thresh, device=device)
    # only some features are activated, chosen at random
    dataset_thresh = torch.rand(dataset_size, n_ground_truth_components, device=device)
    dataset_codes = torch.where(
        dataset_thresh <= component_probs,
        feature_strengths,
        data_zero,
    )
    # Ensure there are no datapoints w/ 0 features
    zero_sample_index = (dataset_codes.count_nonzero(dim=1) == 0).nonzero()[:, 0]
    random_index = torch.randint(low=0, high=n_ground_truth_components, size=(zero_sample_index.shape[0],)).to(
        dataset_codes.device
    )
    dataset_codes[zero_sample_index, random_index] = 1.0

    # Multiply by a 2D random matrix of feature strengths
    dataset = dataset_codes @ feats

    return feats, dataset_codes, dataset


def generate_rand_feats(
    feat_dim: int,
    num_feats: int,
    device: Union[torch.device, str],
) -> TensorType["n_ground_truth_components", "activation_dim"]:
    data_path = os.path.join(os.getcwd(), "data")
    data_filename = os.path.join(data_path, f"feats_{feat_dim}_{num_feats}.npy")

    feats = np.random.multivariate_normal(np.zeros(feat_dim), np.eye(feat_dim), size=num_feats)
    feats = feats.T / np.linalg.norm(feats, axis=1)
    feats = feats.T

    feats_tensor = torch.from_numpy(feats).to(device).float()
    return feats_tensor


def generate_corr_matrix(
    num_feats: int, device: Union[torch.device, str]
) -> TensorType["n_ground_truth_components", "n_ground_truth_components"]:
    corr_mat_path = os.path.join(os.getcwd(), "data")
    corr_mat_filename = os.path.join(corr_mat_path, f"corr_mat_{num_feats}.npy")

    # Create a correlation matrix
    corr_matrix = np.random.rand(num_feats, num_feats)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    min_eig = np.min(np.real(np.linalg.eigvals(corr_matrix)))
    if min_eig < 0:
        corr_matrix -= 1.001 * min_eig * np.eye(corr_matrix.shape[0], corr_matrix.shape[1])

    corr_matrix_tensor = torch.from_numpy(corr_matrix).to(device).float()

    return corr_matrix_tensor


# AutoEncoder Definition
class AutoEncoder(nn.Module):
    def __init__(self, activation_size, n_dict_components):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(activation_size, n_dict_components), nn.ReLU())
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
    vecs1: Union[torch.Tensor, torch.nn.parameter.Parameter, npt.NDArray],
    vecs2: Union[torch.Tensor, torch.nn.parameter.Parameter, npt.NDArray],
) -> np.ndarray:
    vecs = [vecs1, vecs2]
    for i in range(len(vecs)):
        if not isinstance(vecs[i], np.ndarray):
            vecs[i] = vecs[i].detach().cpu().numpy()  # type: ignore
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
        x_hat, c = auto_encoder(batch)  # x_hat: (batch_size, activation_dim), c: (batch_size, n_dict_components)
        outputs.append(c)
    outputs = torch.cat(outputs)  # (n_batches * batch_size, n_dict_components)
    mean_activations = outputs.mean(dim=0)  # (n_dict_components), c is after the ReLU, no need to take abs
    n_dead_neurons = (mean_activations == 0).sum().item()
    return n_dead_neurons


def analyse_result(result):
    get_n_dead_neurons(result)


def run_single_go(cfg: ToyArgs, data_generator: Optional[RandomDatasetGenerator]):
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
    optimizer = optim.Adam(auto_encoder.parameters(), lr=cfg.lr)

    # Hold a running average of the reconstruction loss
    running_recon_loss = 0.0
    time_horizon = 1000
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
        l_l1 = cfg.l1_alpha * torch.norm(c, 1, dim=1).mean() / c.size(1)
        # l_l1 = l1_alpha * torch.norm(c,1, dim=1).sum() / c.size(1)

        # Compute the total loss
        loss = l_reconstruction + l_l1

        # Backward pass
        loss.backward()

        optimizer.step()

        # Add the loss for this batch to the total loss for this epoch
        epoch_loss += loss.item()
        running_recon_loss *= (time_horizon - 1) / time_horizon
        running_recon_loss += l_reconstruction.item() / time_horizon

        if (epoch + 1) % 1000 == 0:
            # Calculate MMCS
            learned_dictionary = auto_encoder.decoder.weight.data.t()
            mmcs = mean_max_cosine_similarity(ground_truth_features.to(auto_encoder.device), learned_dictionary)
            print(f"Mean Max Cosine Similarity: {mmcs:.3f}")

            # Compute the average loss for this epoch
            # epoch_loss /= (dataset_size // batch_size)
            # debug_sparsity_of_c(auto_encoder, ground_truth_features, probabilities, batch_size)

            if True:
                print(f"Epoch {epoch+1}/{cfg.epochs}: Reconstruction = {l_reconstruction:.6f} | l1: {l_l1:.6f}")

    # debug_sparsity_of_c(auto_encoder, ground_truth_features, probabilities, batch_size)

    learned_dictionary = auto_encoder.decoder.weight.data.t()
    mmcs = mean_max_cosine_similarity(ground_truth_features.to(auto_encoder.device), learned_dictionary)
    n_dead_neurons = get_n_dead_neurons(auto_encoder, data_generator)
    return mmcs, auto_encoder, n_dead_neurons, running_recon_loss


def plot_mat(
    mat,
    l1_alphas,
    learned_dict_ratios,
    show=True,
    save_folder=None,
    save_name=None,
    title=None,
):
    """
    :param mmcs_mat: matrix values
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
    plt.set_cmap("viridis")
    # turn x labels 90 degrees
    plt.xticks(rotation=90)
    if title:
        plt.title(title)

    if show:
        plt.show()

    if save_folder:
        plt.savefig(os.path.join(save_folder, save_name))
        plt.close()


def compare_mmcs_with_larger_dicts(dict: npt.NDArray, larger_dicts: List[npt.NDArray]) -> float:
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
        element = np.expand_dims(dict[elem_ndx], 0)
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
        c, x_hat = auto_encoder(batch)

        # Compute the reconstruction loss
        l_reconstruction = torch.norm(x_hat - batch, 2, dim=1).sum() / batch.size(1)

        # Add the loss for this batch to the total loss for this epoch
        recon_loss += l_reconstruction.item() / time_horizon

    ground_truth_features = data_generator.feats
    learned_dictionary = auto_encoder.decoder.weight.data.t()
    mmcs = mean_max_cosine_similarity(ground_truth_features.to(auto_encoder.device), learned_dictionary)
    n_dead_neurons = get_n_dead_neurons(auto_encoder, data_generator)
    return mmcs, learned_dictionary, n_dead_neurons, recon_loss


def main():
    cfg = ToyArgs()
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    l1_range = [10 ** (exp / 4) for exp in range(cfg.l1_exp_low, cfg.l1_exp_high)]  # replicate is (-8,9)
    learned_dict_ratios = [2**exp for exp in range(cfg.dict_ratio_exp_low, cfg.dict_ratio_exp_high)]  # replicate is (-2,6)
    print("Range of l1 values being used: ", l1_range)
    print("Range of dict_sizes compared to ground truth being used:", learned_dict_ratios)
    mmcs_matrix = np.zeros((len(l1_range), len(learned_dict_ratios)))
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
        correlated=cfg.correlated_components,
        device=device,
    )

    # 2D array of learned dictionaries, indexed by l1_alpha and learned_dict_ratio, start with Nones
    auto_encoders = [[None for _ in range(len(learned_dict_ratios))] for _ in range(len(l1_range))]
    learned_dicts = [[None for _ in range(len(learned_dict_ratios))] for _ in range(len(l1_range))]

    for l1_alpha, learned_dict_ratio in tqdm(list(itertools.product(l1_range, learned_dict_ratios))):
        cfg.l1_alpha = l1_alpha
        cfg.learned_dict_ratio = learned_dict_ratio
        cfg.n_components_dictionary = int(cfg.n_ground_truth_components * cfg.learned_dict_ratio)
        mmcs, auto_encoder, n_dead_neurons, reconstruction_loss = run_single_go(cfg, data_generator)
        print(
            f"l1_alpha: {l1_alpha} | learned_dict_ratio: {learned_dict_ratio} | mmcs: {mmcs:.3f} | n_dead_neurons: {n_dead_neurons} | reconstruction_loss: {reconstruction_loss:.3f}"
        )

        mmcs_matrix[l1_range.index(l1_alpha), learned_dict_ratios.index(learned_dict_ratio)] = mmcs
        dead_neurons_matrix[l1_range.index(l1_alpha), learned_dict_ratios.index(learned_dict_ratio)] = n_dead_neurons
        recon_loss_matrix[l1_range.index(l1_alpha), learned_dict_ratios.index(learned_dict_ratio)] = reconstruction_loss
        auto_encoders[l1_range.index(l1_alpha)][learned_dict_ratios.index(learned_dict_ratio)] = auto_encoder.cpu()
        learned_dicts[l1_range.index(l1_alpha)][learned_dict_ratios.index(learned_dict_ratio)] = (
            auto_encoder.decoder.weight.detach().cpu().data.t()
        )

    outputs_folder = "outputs"
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    outputs_folder = os.path.join(outputs_folder, current_time)
    os.makedirs(outputs_folder, exist_ok=True)

    # Save the matrices and the data generator
    plot_mat(
        mmcs_matrix,
        l1_range,
        learned_dict_ratios,
        show=False,
        save_folder=outputs_folder,
        title="Mean Max Cosine Similarity w/ True",
        save_name="mmcs_matrix.png",
    )
    # clamp dead_neurons to 0-100 for better visualisation
    dead_neurons_matrix = np.clip(dead_neurons_matrix, 0, 100)
    plot_mat(
        dead_neurons_matrix,
        l1_range,
        learned_dict_ratios,
        show=False,
        save_folder=outputs_folder,
        title="Dead Neurons",
        save_name="dead_neurons_matrix.png",
    )
    plot_mat(
        recon_loss_matrix,
        l1_range,
        learned_dict_ratios,
        show=False,
        save_folder=outputs_folder,
        title="Reconstruction Loss",
        save_name="recon_loss_matrix.png",
    )
    with open(os.path.join(outputs_folder, "auto_encoders.pkl"), "wb") as f:
        pickle.dump(auto_encoders, f)
    with open(os.path.join(outputs_folder, "data_generator.pkl"), "wb") as f:
        pickle.dump(data_generator, f)
    with open(os.path.join(outputs_folder, "config.pkl"), "wb") as f:
        pickle.dump(cfg, f)

    # Compare each learned dictionary to the larger ones
    av_mmcs_with_larger_dicts = np.zeros((len(l1_range), len(learned_dict_ratios)))
    for l1_ndx, dict_size_ndx in tqdm(list(itertools.product(range(len(l1_range)), range(len(learned_dict_ratios))))):
        if dict_size_ndx == len(learned_dict_ratios) - 1:
            continue
        smaller_dict = learned_dicts[l1_ndx][dict_size_ndx]
        larger_dict = learned_dicts[l1_ndx][dict_size_ndx + 1]
        smaller_dict_features, shared_vector_size = smaller_dict.shape

        max_cosine_similarities = np.zeros((smaller_dict_features))
        for i, vector in enumerate(smaller_dict):
            max_cosine_similarities[i] = torch.nn.functional.cosine_similarity(
                vector.to(cfg.device), larger_dict.to(cfg.device), dim=1
            ).max()
        av_mmcs_with_larger_dicts[l1_ndx, dict_size_ndx] = max_cosine_similarities.mean().item()

    plot_mat(
        av_mmcs_with_larger_dicts,
        l1_range,
        learned_dict_ratios,
        show=False,
        save_folder=outputs_folder,
        title="Average mmcs with larger dicts",
        save_name="av_mmcs_with_larger_dicts.png",
    )


if __name__ == "__main__":
    main()
