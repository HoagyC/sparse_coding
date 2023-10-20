import os
from dataclasses import dataclass, field
from typing import Any, Generator, Optional, Tuple, Union

import numpy as np
import torch
from torchtyping import TensorType

n_ground_truth_components_, activation_dim_, dataset_size_ = (
    None,
    None,
    None,
) # type: Tuple[None, None, None]


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
        self.t_type = torch.float32

    def send(self, ignored_arg: Any) -> TensorType["dataset_size_", "activation_dim_"]:
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
        return data.to(self.t_type)

    def throw(self, type: Any = None, value: Any = None, traceback: Any = None) -> None:
        raise StopIteration


@dataclass
class SparseMixDataset(Generator):
    activation_dim: int
    n_sparse_components: int
    batch_size: int
    feature_num_nonzero: int
    feature_prob_decay: float
    noise_magnitude_scale: float
    device: Union[torch.device, str]

    sparse_component_dict: Optional[TensorType["n_sparse_components", "activation_dim"]] = None
    sparse_component_covariance: Optional[TensorType["n_sparse_components", "n_sparse_components"]] = None
    noise_covariance: Optional[TensorType["activation_dim", "activation_dim"]] = None
    t_type: Optional[torch.dtype] = None

    sparse_component_probs: Optional[TensorType["n_sparse_components"]] = field(init=False)

    def __post_init__(self):
        self.frac_nonzero = self.feature_num_nonzero / self.n_sparse_components
        if self.sparse_component_dict is None:
            print("generating features...")
            self.sparse_component_dict = generate_rand_feats(
                self.activation_dim,
                self.n_sparse_components,
                device=self.device,
            )

        if self.sparse_component_covariance is None:
            print("generating covariances...")
            self.sparse_component_covariance = generate_corr_matrix(self.n_sparse_components, device=self.device)

        if self.noise_covariance is None:
            self.noise_covariance = torch.eye(self.activation_dim, device=self.device)

        self.frac_nonzero = self.feature_num_nonzero / self.n_sparse_components
        self.sparse_component_probs = torch.tensor(
            [self.feature_prob_decay**i for i in range(self.n_sparse_components)],
            dtype=torch.float32,
        )
        self.sparse_component_probs = self.sparse_component_probs.to(self.device)

        if self.t_type is None:
            self.t_type = torch.float32

    def send(self, batch_size: Optional[int]) -> TensorType["dataset_size_", "activation_dim_"]:
        _, _, sparse_data = generate_correlated_dataset(
            self.n_sparse_components,
            self.batch_size if batch_size is None else batch_size,
            self.sparse_component_covariance,
            self.sparse_component_dict,
            self.frac_nonzero,
            self.sparse_component_probs,
            self.device,
        )
        noise_data = generate_noise_dataset(
            self.batch_size if batch_size is None else batch_size,
            self.noise_covariance,
            self.noise_magnitude_scale,
            self.device,
        )

        data = sparse_data + noise_data

        return data.to(self.t_type)

    def throw(self, type: Any = None, value: Any = None, traceback: Any = None) -> None:
        raise StopIteration


def generate_noise_dataset(
    dataset_size: int,
    noise_covariance: TensorType["activation_dim_", "activation_dim_"],
    noise_magnitude_scale: float,
    device: Union[torch.device, str],
) -> TensorType["dataset_size_", "activation_dim_"]:
    noise = torch.distributions.MultivariateNormal(
        loc=torch.zeros(noise_covariance.shape[0], device=device),
        covariance_matrix=noise_covariance,
    ).sample(torch.Size([dataset_size]))
    noise *= noise_magnitude_scale

    return noise


def generate_rand_dataset(
    n_ground_truth_components: int,  #
    dataset_size: int,
    feature_probs: TensorType["n_ground_truth_components_"],
    feats: TensorType["n_ground_truth_components_", "activation_dim_"],
    device: Union[torch.device, str],
) -> Tuple[
    TensorType["n_ground_truth_components_", "activation_dim_"],
    TensorType["dataset_size_", "n_ground_truth_components_"],
    TensorType["dataset_size_", "activation_dim_"],
]:
    dataset_thresh = torch.rand(dataset_size, n_ground_truth_components, device=device)
    dataset_values = torch.rand(dataset_size, n_ground_truth_components, device=device)

    data_zero = torch.zeros_like(dataset_thresh, device=device)

    dataset_codes = torch.where(
        dataset_thresh <= feature_probs,
        dataset_values,
        data_zero,
    )  # dim: dataset_size x n_ground_truth_components

    # Multiply by a 2D random matrix of feature strengths
    feature_strengths = torch.rand((dataset_size, n_ground_truth_components), device=device)
    dataset = (dataset_codes * feature_strengths) @ feats

    # dataset = dataset_codes @ feats

    return feats, dataset_codes, dataset


def generate_correlated_dataset(
    n_ground_truth_components: int,
    dataset_size: int,
    corr_matrix: TensorType["n_ground_truth_components_", "n_ground_truth_components_"],
    feats: TensorType["n_ground_truth_components_", "activation_dim_"],
    frac_nonzero: float,
    decay: TensorType["n_ground_truth_components_"],
    device: Union[torch.device, str],
) -> Tuple[
    TensorType["n_ground_truth_components_", "activation_dim_"],
    TensorType["dataset_size_", "n_ground_truth_components_"],
    TensorType["dataset_size_", "activation_dim_"],
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
    zero_sample_index = (dataset_codes.count_nonzero(dim=1) == 0).nonzero()[:, 0]
    random_index = torch.randint(low=0, high=n_ground_truth_components, size=(zero_sample_index.shape[0],)).to(
        dataset_codes.device
    )
    dataset_codes[zero_sample_index, random_index] = 1.0

    # Multiply by a 2D random matrix of feature strengths
    feature_strengths = torch.rand((dataset_size, n_ground_truth_components), device=device)
    dataset = (dataset_codes * feature_strengths) @ feats

    return feats, dataset_codes, dataset


def generate_rand_feats(
    feat_dim: int,
    num_feats: int,
    device: Union[torch.device, str],
) -> TensorType["n_ground_truth_components_", "activation_dim_"]:
    data_path = os.path.join(os.getcwd(), "data")
    data_filename = os.path.join(data_path, f"feats_{feat_dim}_{num_feats}.npy")

    feats = np.random.multivariate_normal(np.zeros(feat_dim), np.eye(feat_dim), size=num_feats)
    feats = feats.T / np.linalg.norm(feats, axis=1)
    feats = feats.T

    feats_tensor = torch.from_numpy(feats).to(device).float()
    return feats_tensor


def generate_corr_matrix(
    num_feats: int, device: Union[torch.device, str]
) -> TensorType["n_ground_truth_components_", "n_ground_truth_components_"]:
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
