import torch
from torchtyping import TensorType
from typing import List, Tuple, Union, Optional

from autoencoders.learned_dict import LearnedDict

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image

from transformer_lens import HookedTransformer
import utils

from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn import metrics

matplotlib.use('Agg')

def run_with_model_intervention(cfg, transformer: HookedTransformer, model: LearnedDict, tensor_name, tokens, other_hooks=[], **kwargs):
    def intervention(tensor, hook=None):
        B, L, C = tensor.shape
        reshaped_tensor = tensor.reshape(B * L, C)

        prediction = model.predict(tensor)
        reshaped_prediction = prediction.reshape(B * L, C)

        return reshaped_prediction

    return transformer.run_with_hooks(
        tokens,
        fwd_hooks = other_hooks + [(
            tensor_name,
            intervention
        )],
        **kwargs
    )

def run_ablating_model_directions(cfg, transformer: HookedTransformer, model: LearnedDict, tensor_name, features_to_ablate, tokens, other_hooks=[], **kwargs):
    def intervention(tensor, hook=None):
        B, L, C = tensor.shape
        reshaped_tensor = tensor.reshape(B * L, C)

        feature_activations = model.encode(reshaped_tensor)
        not_ablated = feature_activations.clone()
        not_ablated[:, features_to_ablate] = 0.0
        to_ablate = feature_activations - not_ablated
        ablation = torch.einsum("nd,bn->bd", model.get_learned_dict(), to_ablate)
        reshaped_ablation = ablation.reshape(B, L, C)

        print(torch.norm(tensor - reshaped_ablation), torch.norm(tensor))

        return tensor - reshaped_ablation

    return transformer.run_with_hooks(
        tokens,
        fwd_hooks = other_hooks + [(
            tensor_name,
            intervention
        )],
        **kwargs
    )

def logistic_regression_auroc(cfg, activations: TensorType["batch_size", "activation_size"], labels: TensorType["batch_size"], **kwargs):
    clf = LogisticRegression(**kwargs)

    activations_, labels_ = activations.cpu().numpy(), labels.cpu().numpy()

    clf.fit(activations_, labels_)
    return metrics.roc_auc_score(labels_, clf.predict_proba(activations_)[:, 1])

def ridge_regression_auroc(cfg, activations: TensorType["batch_size", "activation_size"], labels: TensorType["batch_size"], **kwargs):
    clf = RidgeClassifier(**kwargs)

    activations_, labels_ = activations.cpu().numpy(), labels.cpu().numpy()

    clf.fit(activations_, labels_)
    return metrics.roc_auc_score(labels_, clf.predict(activations_))

def measure_concept_erasure(cfg, transformer: HookedTransformer, model: LearnedDict, ablation_tensor, features_to_ablate, read_tensor, tokens, labels: TensorType["batch_size"], **kwargs):
    # tokens: [batch_size, sequence_length]
    B, L = tokens.shape

    activation_cache = None

    def read_tensor_hook(tensor, hook=None):
        nonlocal activation_cache
        activation_cache = tensor.clone()
        return tensor
    
    with torch.no_grad():
        transformer.run_with_hooks(
            tokens,
            fwd_hooks = [(read_tensor, read_tensor_hook)],
            **kwargs
        )
    y_true = activation_cache.reshape(B * L, -1)

    with torch.no_grad():
        run_ablating_model_directions(cfg, transformer, model, ablation_tensor, features_to_ablate, tokens, other_hooks=[(read_tensor, read_tensor_hook)], **kwargs)
    y_ablated = activation_cache.reshape(B * L, -1)

    labels_reshaped = labels.reshape(B * L)

    # should use different classifiers for each task, we want to compare linear separability
    # just using this for sanity checks atm
    model = RidgeClassifier()
    model.fit(y_true.cpu().numpy(), labels_reshaped.cpu().numpy())

    y_true_pred = model.predict(y_true.cpu().numpy())
    y_ablated_pred = model.predict(y_ablated.cpu().numpy())

    auroc_true = metrics.roc_auc_score(labels_reshaped.cpu().numpy(), y_true_pred)
    auroc_ablated = metrics.roc_auc_score(labels_reshaped.cpu().numpy(), y_ablated_pred)

    # not sure what summary to use here?
    return (auroc_ablated - 0.5) / (auroc_true - 0.5), auroc_true, auroc_ablated

def mcs_duplicates(ground: LearnedDict, model: LearnedDict):
    # get max cosine sim between each model atom and all ground atoms
    cosine_sim = torch.einsum("md,gd->mg", model.get_learned_dict(), ground.get_learned_dict())
    max_cosine_sim = cosine_sim.max(dim=-1).values
    return max_cosine_sim

def mean_nonzero_activations(model: LearnedDict, batch: TensorType["batch_size", "activation_size"]):
    c = model.encode(batch)
    return (c > 0.0).float().mean(dim=0)

def fraction_variance_unexplained(model: LearnedDict, batch: TensorType["batch_size", "activation_size"]):
    x_hat = model.predict(batch)
    residuals = (batch - x_hat).pow(2).mean()
    total = (batch - batch.mean(dim=0)).pow(2).mean()
    return residuals / total

def r_squared(model: LearnedDict, batch: TensorType["batch_size", "activation_size"]):
    return 1.0 - fraction_variance_unexplained(model, batch)

def plot_hist(scores: TensorType["n_dict_components"], x_label, y_label, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(scores, **kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return Image.fromarray(data, mode="RGB")

def plot_scatter(scores_x: TensorType["n_dict_components"], scores_y: TensorType["n_dict_components"], x_label, y_label, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(scores_x, scores_y, **kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return Image.fromarray(data, mode="RGB")

def plot_grid(scores: np.ndarray, first_tick_labels, second_tick_labels, first_label, second_label, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(scores, **kwargs)
    ax.set_xticks(np.arange(len(first_tick_labels)))
    ax.set_yticks(np.arange(len(second_tick_labels)))
    ax.set_xticklabels(first_tick_labels)
    ax.set_yticklabels(second_tick_labels)
    ax.set_xlabel(first_label)
    ax.set_ylabel(second_label)
    
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return Image.fromarray(data, mode="RGB")

if __name__ == "__main__":
    from argparser import parse_args

    device = "cuda:0"

    transformer = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device=device)
    dicts = torch.load("output/_7/learned_dicts.pt")

    print(dicts)

    model = dicts[0][0]
    model.to_device(device)

    text = """
Minus nesciunt voluptatem reiciendis facere culpa aut aperiam quae. Quod sit assumenda aperiam eos dignissimos autem. Minus id nobis placeat impedit reiciendis debitis similique nesciunt. Fugit suscipit vel ea. Sit quibusdam a quos neque sunt adipisci officiis qui.

Alias tempore cupiditate aperiam recusandae aperiam cupiditate maxime dolores. Minima autem aut impedit eligendi minus. Provident blanditiis ratione saepe qui. Ut officia qui aut id facere.

Non est cum quae. Eos nobis sit omnis. Recusandae quo molestias rem accusamus. Hic eum sed consectetur veniam aut accusantium. Ullam blanditiis non deleniti quia iste culpa distinctio odit.

Ab sit at amet molestiae culpa nesciunt qui eum. Praesentium eum assumenda et illo quae unde. Ex et id numquam dolore voluptates necessitatibus quo reprehenderit. Est delectus omnis nesciunt et et quaerat. Magni deserunt est dolore.

Quo laborum quis sit ratione dolores dignissimos tempore in. Quisquam occaecati non repellendus vel. Blanditiis ut nisi atque. Illum molestias impedit nesciunt officiis voluptas quidem eius enim.
    """
    tokens = transformer.to_tokens(text)

    directions_to_ablate = list(range(model.get_learned_dict().shape[0]))

    cfg = parse_args()

    cfg.layer = 2
    cfg.use_residual = True
    cfg.model_name = "EleutherAI/pythia-70m-deduped"

    tensor_to_ablate = utils.make_tensor_name(cfg.layer, cfg.use_residual, cfg.model_name)
    tensor_to_read = utils.make_tensor_name(cfg.layer+1, cfg.use_residual, cfg.model_name)

    erasure_score, true, ablated = measure_concept_erasure(
        cfg,
        transformer,
        model,
        tensor_to_ablate,
        directions_to_ablate,
        tensor_to_read,
        tokens,
        torch.randint(0, 2, tokens.shape, device=device)
    )

    activations = None
    def copy_to_activations(tensor, hook=None):
        global activations
        activations = tensor.clone()
        return tensor

    with torch.no_grad():
        transformer.run_with_hooks(
            tokens,
            fwd_hooks = [(tensor_to_read, copy_to_activations)]
        )

    r_sq = fraction_variance_unexplained(model, activations.reshape(-1, activations.shape[-1])).item()

    print(erasure_score, true, ablated, r_sq)