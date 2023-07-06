import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as data

import torch.multiprocessing as mp

from ensemble import VectorizedEnsemble

# suboptimal solution (have to wait for all jobs to finish using dataloader to exit)
# works for now

mp.set_start_method("spawn")

# jobs: iterable -> process
# TODO: queue listener
def dispatch_on_chunk(ensembles, args, dataset):
    dataset.share_memory()
    for ensemble in ensembles:
        ensemble.share_memory()

    processes = []
    queues = []
    for i in range(args):
        q = mp.Queue()
        p = mp.Process(target=train_as_process, args=(i, args[i], q))
        q.put((ensembles[i].state_dict(), dataset))

        p.start()
        processes.append(p)
        queues.append(q)
    
    outputs = [p.join() for p in processes]
    
    return outputs

# TODO: output debug/logging info
def train_as_process(instance, args, queue):
    state_dict, dataset = queue.get()

    ensemble = VectorizedEnsemble.from_state(state_dict)

    batch_size = args["batch_size"]
    device = args["device"]
    
    # can't use DataLoaders because they copy the dataset
    # instead, we use a custom sampler for indexes
    sampler = data.BatchSampler(
        data.RandomSampler(range(dataset.shape[0])),
        batch_size=batch_size,
        drop_last=False
    )

    losses = []
    for batch_idxs in iter(dataloader):
        batch = dataset[batch_idxs].to(device)
        loss = ensemble.step_batch(batch)
        losses.append(loss)
    
    return losses

if __name__ == "__main__":
    from autoencoders.sae_ensemble import SAE
    import torchopt

    l1_alphas = [0.001, 0.01, 0.05, 0.1]
    dict_sizes = [1024, 2048, 4096, 8192]

    activation_size = 1024

    ensembles = []
    for dict_size in dict_sizes:
        models = [SAE(activation_size, dict_size, l1_alpha) for l1_alpha in l1_alphas]
        ensemble = VectorizedEnsemble(models, torchopt.Adam(0.001))
        ensembles.append(ensemble)
    
    devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    args = [{"batch_size": 128, "device": device} for device in devices]
    for i in range(ensembles):
        ensemble.to_device(devices[i])
    
    dataset = torch.randn(100000, activation_size)

    outputs = dispatch_on_chunk(ensembles, args, dataset)
    
    print(outputs)