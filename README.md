## Sparse Coding

This repo contains code for applying sparse coding to activation vectors in language models. Work done with Logan Riggs and Aidan Ewart, advised by Lee Sharkey.

`run.py` contains a more set of functions for generating datasets using Pile10k and then running sparse autoencoders activations on the data to try and learn the features that the model is using for its computation. It is set up by default to run hyperparameter sweeps using across dictionary size and l1 coefficient.

`python replicate_toy_models.py` runs code which allows for the replication of the first half of the post [Taking features out of superposition with sparse autoencoders](https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition).

The repo also contains utils for running code on vast.ai computers which can speed up these sweeps.

## Automatic Interpretation

`interpret.py` contains tools to interpret learned dictionaries using OpenAI's automatic interpretation protocol. Set `--load_interpret_autoencoder` to the location of the autoencoder you want to test, and `--model_name`, `--layer` and `--use_residual` to specify the activations that should be used. `--activation_tranform` should be set to `feature_dict` for interpreting a learned dictionary but there are many baselines that can also be run, including `pca`, `ica`, `nmf`, `neuron_basis`, and `random`.

If you run `interpret.py read_results --kwargs..` and select the `--model_name`, `--layer` and `--use_residual`, this will produce a series of plots comparing 

## Training a custom small transformer

One part of replicating Conjecture's sparse coding work was to use a very small transformer for some early tests using sparse autoencoders to find features.
There doesn't appear to be an open-source model of this kind, and the original model is proprietary, so below are the instructions I followed to create a similar small transformer.

Make sure you have >200GB disk space.
Tested using a [vast.ai](vast.ai) RTX3090 and pytorch:latest docker image.

```
git clone https://github.com/karpathy/nanoGPT
cd nanoGPT
python -m venv .env
source .env/bin/activate
apt install -y build-essential
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Change config/train_gpt2.py to have:
```
import time
wandb_project = 'sparsecode'
wandb_run_name = 'supertiny-' + str(time.time())
n_layer = 6 # (same as train_shakespeare and Lee's work)
n_embd = 16 # (same as Lee's)
n_head = 8 # (needs to divide n_embd)
dropout = 0.2 # (used in shakespeare_char)
block_size = 256 # (just to make faster?)
batch_size = 64
```

To set up the dataset run:

`python data/openwebtext/prepare.py`

Then if using multiple gpus, run:

`torchrun --standalone --nproc_per_node={N_GPU} train.py config/train_gpt2.py`

else simply run:

`python train.py`
