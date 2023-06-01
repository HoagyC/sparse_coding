
## Sparse Coding

`toy_model.py` contains code which allows for the replication of the post [Taking features out of superposition with sparse autoencoders](https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition).

Run `python toy_model.py` to copy the l1 / dict_size sweep in the original post.

## Training a custom small transformer

The next part of the sparse coding work uses a very small transformer to 
There doesn't appear to be an open-source model of this kind, and the original model is proprietary, so below are the instructions I followed to create a similar small transformer.

Make sure you have >200GB space
Using a rented RTX3090

```
git clone https://github.com/karpathy/nanoGPT
cd nanoGPT
python -m venv .env
source .env/bin/activate
apt install -y build-essential
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Change config/train_gpt2.py to have:
* n_layer = 6 (same as train_shakespeare and Lee's work)
* n_embd = 16 (same as Lee's)
* n_head = 8 (needs to divide n_embd)
* dropout = 0.2 (used in shakespeare_char)
* block_size = 256 (just to make faster?)
* batch_size = 64

Run the standard commands in the nanoGPT readme 
but in the run command change nproc_per_node to 1
so:

`python data/openwebtext/prepare.py`
`torchrun --standalone --nproc_per_node=1 train.py config/train_gpt2.py`

