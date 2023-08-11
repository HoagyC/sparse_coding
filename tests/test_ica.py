import unittest

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from autoencoders.ica import ICAEncoder

class TestICA(unittest.TestCase):
    def test_sparse_data(self):
        # Create sparse synthetic data
        np.random.seed(0)
        X = torch.tensor(np.random.laplace(0, 1, (1000, 2)))

        # Apply your ICA function
        ica = ICAEncoder(2)
        output = ica.train(X)

        repeat_output = ica.encode(X)

        self.assertTrue(np.allclose(output, repeat_output, atol=1e-5))

        # Check that the ICA components are close to the identity matrix
        self.assertTrue(np.allclose(abs(ica.ica.components_), np.eye(2), atol=1e-1))

    def test_gaussian_data(self):
        # Create Gaussian synthetic data with independent components
        np.random.seed(42)
        X = torch.tensor(np.random.randn(1000, 2))

        # Apply your ICA function multiple times
        ica1 = ICAEncoder(2)
        ica2 = ICAEncoder(2)
        output1 = ica1.train(X)
        output2 = ica2.train(X)

        # Since the data is Gaussian, the ICA solutions are not identifiable, and multiple runs should produce different results
        self.assertFalse(np.allclose(output1, output2, atol=1e-5))

        # Now try for non-Gaussian data
        np.random.seed(42)
        X = torch.tensor(np.random.laplace(0, 1, (1000, 2)))

        # Apply your ICA function multiple times
        ica1 = ICAEncoder(2)
        ica2 = ICAEncoder(2)
        output1 = ica1.train(X)
        output2 = ica2.train(X)

        # Since the data is non-Gaussian, the ICA solutions are identifiable, and multiple runs should produce the same results
        self.assertTrue(np.allclose(output1, output2, atol=1e-5))

if __name__ == '__main__':
    unittest.main()