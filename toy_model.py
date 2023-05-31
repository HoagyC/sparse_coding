from collections.abc import Generator
from dataclasses import dataclass, field
from functools import partial
import os
from typing import Union, Tuple, List, Dict, Any, Iterator, Callable, Optional

import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, multivariate_normal  # type: ignore

import torch
import torch.nn as nn
import torch.optim as optim

from torchtyping import TensorType

n_ground_truth_components, n_features, dataset_size = None, None, None

@dataclass
class RandomDatasetGenerator(Generator):
    n_features: int
    n_ground_truth_components: int
    batch_size: int
    feature_num_nonzero: int
    feature_prob_decay: float
    runs_share_feats: bool
    correlated: bool
    device: Union[torch.device, str]
    
    frac_nonzero: float = field(init=False)
    decay: TensorType['n_ground_truth_components'] = field(init=False)
    feats: TensorType['n_ground_truth_components', 'n_features'] = field(init=False)
    corr_matrix: Optional[TensorType['n_ground_truth_components', 
                                     'n_ground_truth_components']] = field(init=False) 
    component_probs: Optional[TensorType['n_ground_truth_components']] = field(init=False)

    def __post_init__(self):
        self.frac_nonzero = self.feature_num_nonzero / self.n_ground_truth_components
        print("Fraction of nonzero elements in the codes is ", self.frac_nonzero)

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
            print(
                "Mean component probabilities are ",
                torch.mean(self.component_probs).item(),
            )
        self.feats = generate_rand_feats(
            self.n_features,
            self.n_ground_truth_components,
            runs_share_feats=self.runs_share_feats,
            device=self.device,
        )

    def send(self, ignored_arg: Any) -> TensorType['dataset_size', 'n_features']:

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
    num_feats: int,
    dataset_size: int,
    feature_probs: TensorType['n_ground_truth_components'],
    feats: TensorType['n_ground_truth_components', 'n_features'],
    device: Union[torch.device, str],
) -> Tuple[
      TensorType['n_ground_truth_components', 'n_features'], 
      TensorType['dataset_size','n_ground_truth_components'], 
      TensorType['dataset_size', 'n_features']
    ]:

    dataset_thresh = torch.rand(dataset_size, num_feats, device=device)
    dataset_values = torch.rand(dataset_size, num_feats, device=device)

    data_zero = torch.zeros_like(dataset_thresh, device=device)

    dataset_codes = torch.where(
        dataset_thresh <= feature_probs,
        dataset_values,
        data_zero,
    )

    dataset = dataset_codes @ feats

    return feats, dataset_codes, dataset


def generate_correlated_dataset(
    num_feats: int,
    dataset_size: int,
    corr_matrix: TensorType['n_ground_truth_components', 'n_ground_truth_components'],
    feats: TensorType['n_ground_truth_components', 'n_features'],
    frac_nonzero: float,
    decay: TensorType['n_ground_truth_components'],
    device: Union[torch.device, str],
) -> Tuple[
      TensorType['n_ground_truth_components', 'n_features'], 
      TensorType['dataset_size','n_ground_truth_components'], 
      TensorType['dataset_size', 'n_features']
    ]:

    # Get a correlated gaussian sample
    mvn = torch.distributions.MultivariateNormal(
        loc=torch.zeros(num_feats, device=device), covariance_matrix=corr_matrix
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
    dataset_thresh = torch.rand(dataset_size, num_feats, device=device)
    dataset_values = torch.rand(dataset_size, num_feats, device=device)

    data_zero = torch.zeros_like(corr_thresh, device=device)
    dataset_codes = torch.where(
        dataset_thresh <= component_probs,
        dataset_values,
        data_zero,
    )
    # Ensure there are no datapoints w/ 0 features
    zero_sample_index = (dataset_codes.count_nonzero(dim=1) == 0).nonzero()[:,0]
    random_index = torch.randint(low=0, high=num_feats, size=(zero_sample_index.shape[0],)).to(dataset_codes.device)
    dataset_codes[zero_sample_index, random_index] = 1.0

    dataset = dataset_codes @ feats

    return feats, dataset_codes, dataset


def generate_rand_feats(
    feat_dim: int,
    num_feats: int,
    runs_share_feats: bool,
    device: Union[torch.device, str],
) -> TensorType['n_ground_truth_components', 'n_features']:
    data_path = os.path.join(os.getcwd(), "data")
    data_filename = os.path.join(data_path, f"feats_{feat_dim}_{num_feats}.npy")

    print("feat_dim is ", feat_dim, " and num_feats is ", num_feats)
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
    

def mean_max_cosine_similarity(ground_truth_features, learned_dictionary, debug=False):
    # Calculate cosine similarity between all pairs of ground truth and learned features
    cos_sim = torch.nn.functional.cosine_similarity(ground_truth_features.unsqueeze(1), learned_dictionary.unsqueeze(0),  dim=2)
    
    # Find the maximum cosine similarity for each ground truth feature
    max_cos_sim, _ = torch.max(cos_sim, dim=1)

    # Calculate the mean of the maximum cosine similarities
    mmcs = torch.mean(max_cos_sim)

    if(debug):
        # Sort the tensor values
        sorted_tensor, _ = torch.sort(max_cos_sim.cpu())

        # Create the histogram
        fig = go.Figure(data=[go.Histogram(x=sorted_tensor, nbinsx=10)])

        fig.update_layout(
            title="Histogram of Max Cosine Similarity",
            xaxis=dict(title="Max Cosine Sim", range=[0, 1]),
            yaxis=dict(title="Frequency")
        )
        fig.update_xaxes(range=[0, 1])

        fig.show()
        return mmcs, max_cos_sim

    return mmcs

def main():    
    activation_dim = 256
    n_ground_truth_components = activation_dim * 2
    n_components_dictionary = n_ground_truth_components * 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    noise_std = 0.1
    l1_alpha = 0.1
    learning_rate=0.001
    epochs = 50000

    feature_prob_decay = 1.00
    feature_num_nonzero = 5
    correlated_components = False

    data_generator = RandomDatasetGenerator(
        n_features=activation_dim,
        n_ground_truth_components=n_ground_truth_components,
        batch_size=batch_size,
        feature_num_nonzero=feature_num_nonzero,
        feature_prob_decay=feature_prob_decay,
        correlated=True,
        runs_share_feats=False,
        device=device,
    )

    auto_encoder = AutoEncoder(activation_dim, n_components_dictionary).to(device)

    ground_truth_features = data_generator.feats
    # Train the model
    optimizer = optim.Adam(auto_encoder.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        epoch_loss = 0.0

        # for batch_index in range(dataset_size // batch_size):
            # Generate a batch of samples
            # batch = final_dataset[batch_index*batch_size:(batch_index+1)*batch_size].to(device)
            # batch = create_dataset(ground_truth_features, probabilities, batch_size).float().to(device)
        batch = next(data_generator)
        batch = batch + 0.1 * torch.randn_like(batch)


        optimizer.zero_grad()

        # Forward pass
        x_hat, c = auto_encoder(batch)
        
        # Compute the reconstruction loss and L1 regularization
        l_reconstruction = (batch - x_hat).pow(2).sum(dim=1).sqrt().mean() / x_hat.size(1)
        # l_reconstruction = torch.nn.MSELoss()(batch, x_hat)
        # l_reconstruction = (batch - x_hat).pow(2).mean(dim=1).sqrt().mean() / x_hat.size(1)
        l_l1 = l1_alpha * torch.norm(c,1, dim=1).mean() / c.size(1)
        # l_l1 = l1_alpha * torch.norm(c,1, dim=1).sum() / c.size(1)

        # Compute the total loss
        loss = l_reconstruction + l_l1

        # Backward pass
        loss.backward()

        optimizer.step()

        # Add the loss for this batch to the total loss for this epoch
        epoch_loss += loss.item()

        if epoch % 100 == 0:
            # Calculate MMCS
            learned_dictionary = auto_encoder.decoder.weight.data.t()
            mmcs = mean_max_cosine_similarity(ground_truth_features.to(auto_encoder.device), learned_dictionary)
            print(f"Mean Max Cosine Similarity: {mmcs:.3f}")

            # Compute the average loss for this epoch
            # epoch_loss /= (dataset_size // batch_size)
            # debug_sparsity_of_c(auto_encoder, ground_truth_features, probabilities, batch_size)
            
            if(True):
                print(f"Epoch {epoch+1}/{epochs}: Reconstruction = {l_reconstruction:.6f} | l1: {l_l1:.6f}")
    # debug_sparsity_of_c(auto_encoder, ground_truth_features, probabilities, batch_size)
    mmcs, max_cos = mean_max_cosine_similarity(ground_truth_features.to("cpu"), learned_dictionary.to("cpu"), debug=True)

if __name__ == "__main__":
    main()