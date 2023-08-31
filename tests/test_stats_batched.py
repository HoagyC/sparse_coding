import os
import sys
import unittest

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from standard_metrics import calc_feature_variance, calc_feature_skew, calc_feature_kurtosis, calc_moments_streaming

class TestStatisticalFunctions(unittest.TestCase):
    
    def test_moments(self):
        activations = torch.randn(10000)
        learned_dict = type('obj', (object,), {'n_feats': 1, 'encode': lambda x: x})
        
        exact_mean = torch.mean(activations)
        exact_var = calc_feature_variance(activations)
        exact_skew = calc_feature_skew(activations)
        exact_kurtosis = calc_feature_kurtosis(activations)
        
        _, batch_mean, batch_var, batch_skew, batch_kurtosis, _ = calc_moments_streaming(learned_dict, activations, batch_size=1000)
        
        self.assertAlmostEqual(exact_mean.item(), batch_mean.item(), places=5)
        self.assertAlmostEqual(exact_var.item(), batch_var.item(), places=3)
        self.assertAlmostEqual(exact_skew.item(), batch_skew.item(), places=3)
        self.assertAlmostEqual(exact_kurtosis.item(), batch_kurtosis.item(), places=2)
        
if __name__ == "__main__":
    unittest.main()
