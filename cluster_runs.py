import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as data

import torch.multiprocessing as mp

from autoencoders.ensemble import FunctionalEnsemble

import progressbar
import time

import itertools

def gen_done_text(done, tags):
    # in the format "tag1: done, tag2: done, ..."
    return ", ".join(f"{tag}: {'#' if done else 'X'}" for tag, d in zip(tags, done))

# suboptimal solution (have to wait for all jobs to finish using dataloader to exit)
# works for now
def dispatch_on_chunk(ensembles, args, tags, dataset, logger=None, interval=0):
    dataset.share_memory_()
    for ensemble in ensembles:
        ensemble.to_shared_memory()

    processes = []
    queues = []
    for i in range(len(args)):
        q = mp.Queue()
        finished = mp.Value("i", 0)
        progress = mp.Value("i", 0)
        p = mp.Process(target=train_as_process, args=(i, args[i], (q, finished, progress), logger, interval))
        #q.put((ensembles[i].state_dict(), dataset))
        ensemble_state = ensembles[i].state_dict()
        #del ensemble_state["modeldesc"]
        q.put((ensemble_state, dataset))

        n_batches = dataset.shape[0] // args[i]["batch_size"]

        p.start()
        processes.append(p)
        queues.append((q, finished, progress, n_batches))
    
    done_width = len(gen_done_text([False] * len(ensembles), tags))

    bar = progressbar.ProgressBar(
        widgets=[
            progressbar.Bar(),
            " ", progressbar.AdaptiveETA(),
            " | ", *[
                progressbar.Variable(tags[i], precision=0, width=1, format="{formatted_value}") for i in range(len(tags))
            ]
        ], max_value=sum(n_batches for _, _, _, n_batches in queues))

    while True:
        done = [finished.value for _, finished, _, _ in queues]
        n_batches_done = sum(progress.value for _, _, progress, _ in queues)
        
        done_text = gen_done_text(done, tags)

        bar.update(n_batches_done, **{tags[i]: done[i] for i in range(len(tags))})

        if all(done):
            break
            
        time.sleep(0.1)
    
    outputs = [q[0].get() for q in queues]

    for p in processes:
        p.join()
    
    return outputs

def train_as_process(instance, args, queues, logger, interval):
    torch.set_grad_enabled(False)

    queue, finished, progress = queues

    state_dict, dataset = queue.get()

    ensemble = FunctionalEnsemble.from_state(state_dict)

    batch_size = args["batch_size"]
    device = args["device"]
    
    # can't use DataLoaders because they copy the dataset
    # instead, we use a custom sampler for indexes
    sampler = torch.utils.data.BatchSampler(
        torch.utils.data.RandomSampler(range(dataset.shape[0])),
        batch_size=batch_size,
        drop_last=False
    )

    logger_data = []
    losses = []
    for n_batch, batch_idxs in enumerate(iter(sampler)):
        batch = dataset[batch_idxs].to(device)
        
        loss, aux = ensemble.step_batch(batch)

        losses.append(loss)

        progress.value = n_batch + 1

        if logger is not None and n_batch % interval == 0:
            data = logger(ensemble, n_batch, loss, aux)
            logger_data.append((n_batch, data))
    
    finished.value = 1

    if logger is not None:
        queue.put((losses, logger_data))
    else:
        queue.put(losses)

    return

if __name__ == "__main__":
    from autoencoders.sae_ensemble import FunctionalSAE
    import torchopt

    torch.set_grad_enabled(False)

    mp.set_start_method("spawn")

    l1_alphas = [0.001, 0.01, 0.05, 0.1]
    dict_sizes = [1024, 2048, 4096, 8192]

    activation_size = 1024

    ensembles = []
    devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    for i, dict_size in enumerate(dict_sizes):
        #ensembles.append(DummyEnsemble(devices[i]))
        models = [FunctionalSAE.init(activation_size, dict_size, l1_alpha) for l1_alpha in l1_alphas]
        ensemble = FunctionalEnsemble(
            models, FunctionalSAE.loss,
            torchopt.adam, {
                "lr": 0.01
            },
            device=devices[i])
        ensembles.append(ensemble)
    
    target_size = 1048576

    args = [{"batch_size": target_size // dict_sizes[i], "device": devices[i]} for i in range(len(ensembles))]
    
    dataset = torch.randn(100000, activation_size)

    outputs = dispatch_on_chunk(ensembles, args, dataset)

    print(outputs)