import unittest

# set the path to the root of the project
import sys
import os
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from interpret import make_activation_dataset, make_feature_activation_dataset
from argparser import parse_args
from utils import dotdict

from datasets import load_dataset
from transformer_lens import HookedTransformer

cfg: dotdict = parse_args()

class TestMain(unittest.TestCase):
    def setUp(self):
        self.cfg = parse_args()
        self.layer = 1
        cfg.model_name = "EleutherAI/pythia-70m-deduped"
        cfg.activation_transform = "feature_dict"
        with open("saved_autoencoders/logan_ae.pkl", "rb") as f:
            self.autoencoder = pickle.load(f).to("cuda")
        activation_fn_kwargs = {"autoencoder": self.autoencoder}
        self.transform_folder = os.path.join("auto_interp_results", "test_transform")
        # clear the folder
        if os.path.exists(self.transform_folder):
            for file in os.listdir(self.transform_folder):
                os.remove(os.path.join(self.transform_folder, file))
        else:
            os.makedirs(self.transform_folder)

        self.sentence_dataset = load_dataset("openwebtext", split="train", streaming=True)
        self.model = HookedTransformer.from_pretrained(cfg.model_name)

        self.df = make_feature_activation_dataset(
            model_name=cfg.model_name,
            model=self.model,
            layer=self.layer,
            use_residual=False,
            activation_fn_name="feature_dict",
            activation_fn_kwargs=activation_fn_kwargs,
            activation_dim=2048,
            use_baukit=False,
            device="cuda",
            n_fragments=3,
            random_fragment=False,
        )
        
    
    def test_main(self):
        tensor_name = f"blocks.{self.layer}.mlp.hook_post"
        sentence = next(iter(self.sentence_dataset))["text"]
        tokens = self.model.to_tokens(sentence)[0, 1:65]
        _, cache = self.model.run_with_cache(tokens)
        mlp_activation_data = cache[tensor_name].to("cuda")[0]
        for position in [0, 10, 63]:
            print(f"Testing position {position}")
            x_hat, activations = self.autoencoder(mlp_activation_data[position])
            for feature in range(2048):
                assert abs(self.df[f"feature_{feature}_activation_{position}"][0] - activations[feature]) < 1e-3, f"feature {feature} does not match. Got {self.df[f'feature_{feature}_activation_{position}'][0]} but expected {activations[feature]}"
    
    def tearDown(self) -> None:
        return super().tearDown()


if __name__ == "__main__":
    unittest.main()