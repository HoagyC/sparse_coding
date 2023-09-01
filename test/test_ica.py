import os
import sys
import unittest

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

        # Check that the ICA components are close to the identity matrix
        # normalised to unit length
        comps = ica.ica.components_ / np.linalg.norm(ica.ica.components_, axis=1)[:, None]
        # order components by first element
        comps = comps[np.argsort(comps[:, 0])]
        self.assertTrue(np.allclose(abs(comps), np.eye(2), atol=1e-1))

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

        # Now try for non-Gaussian data
        np.random.seed(42)
        X = torch.tensor(np.random.laplace(0, 1, (1000, 4)))

        # Apply your ICA function multiple times
        ica1 = ICAEncoder(4)
        ica2 = ICAEncoder(4)
        output1 = ica1.train(X)
        output2 = ica2.train(X)
        
        # sort components by first element and get the ordering
        ordering1 = np.argsort(abs(ica1.ica.components_[:, 0]))
        ordering2 = np.argsort(abs(ica2.ica.components_[:, 0]))
        # print(abs(ica1.ica.components_[ordering1]))
        # print(abs(ica2.ica.components_[ordering2]))
        # print(output1[:, ordering1])
        # print(output2[:, ordering2])
        # print(np.max(abs(abs(output1[:, ordering1]) - abs(output2[:, ordering2]))))
        
        # Since the data is non-Gaussian, the ICA solutions are identifiable, and multiple runs should produce the same results
        self.assertTrue(np.allclose(abs(ica1.ica.components_[ordering1]), abs(ica2.ica.components_[ordering2]), atol=1e-3))
        self.assertTrue(np.allclose(abs(output1[:, ordering1]), abs(output2[:, ordering2]), atol=3e-3))


if __name__ == "__main__":
    unittest.main()
