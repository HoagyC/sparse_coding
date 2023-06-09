
*Work done with Logan Riggs who wrote the original replication notebook. Thanks to Pierre Peigne for the data generating code and Lee Sharkey for answering questions.*

## Sparse Coding

`python replicate_toy_models.py` runs code which allows for the replication of the first half of the post [Taking features out of superposition with sparse autoencoders](https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition).

`run.py` contains a more flexible set of functions for generating datasets using Pile10k and then running sparse coding activations on real models, incluidng gpt-2-small and custom models.

The repo also contains utils for running code on vast.ai computers which can speed up these sweeps.

## Training a custom small transformer

The next part of the sparse coding work uses a very small transformer to do some early tests using sparse autoencoders to find features.
There doesn't appear to be an open-source model of this kind, and the original model is proprietary, so below are the instructions I followed to create a similar small transformer.

Make sure you have >200GB space.
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
