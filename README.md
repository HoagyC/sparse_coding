## Sparse Coding

This repo contains code for applying sparse coding to activation vectors in language models, including the code used for the results in the paper [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/pdf/2309.08600.pdf). Work done with Logan Riggs and Aidan Ewart, advised by Lee Sharkey.

The repo is designed to train multiple sparse autoencoders simultaneously using different L1 values, on either a single GPU or across multiple. `big_sweep_experiments` contains a number of examples of run functions. 

## Automatic Interpretation

`interpret.py` contains tools to interpret learned dictionaries using OpenAI's automatic interpretation protocol. Set `--load_interpret_autoencoder` to the location of the autoencoder you want to test, and `--model_name`, `--layer` and `--layer_loc` to specify the activations that should be used. `--activation_tranform` should be set to `feature_dict` for interpreting a learned dictionary but there are many baselines that can also be run, including `pca`, `ica`, `nmf`, `neuron_basis`, and `random`.

If you run `interpret.py read_results --kwargs..` and select the `--model_name`, `--layer` and `--layer_loc`, this will produce a series of plots comparing 