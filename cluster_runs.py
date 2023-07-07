import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as data

import torch.multiprocessing as mp

from autoencoders.ensemble import FunctionalEnsemble

# suboptimal solution (have to wait for all jobs to finish using dataloader to exit)
# works for now

# jobs: iterable -> process
# TODO: queue listener
def dispatch_on_chunk(ensembles, args, dataset):
    dataset.share_memory_()
    for ensemble in ensembles:
        ensemble.to_shared_memory()

    processes = []
    queues = []
    for i in range(len(args)):
        q = mp.Queue()
        p = mp.Process(target=train_as_process, args=(i, args[i], q))
        #q.put((ensembles[i].state_dict(), dataset))
        ensemble_state = ensembles[i].state_dict()
        #del ensemble_state["modeldesc"]
        q.put((ensemble_state, dataset))

        p.start()
        processes.append(p)
        queues.append(q)
    
    outputs = [p.join() for p in processes]
    
    return outputs

# TODO: output debug/logging info
def train_as_process(instance, args, queue):
    state_dict, dataset = queue.get()

    ensemble = FunctionalEnsemble.from_state(state_dict)

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
    for epoch, batch_idxs in enumerate(iter(sampler)):
        print(f"Instance {instance} epoch {epoch}")
        batch = dataset[batch_idxs].to(device)
        ensemble.step_batch(batch)
        #losses.append(loss.detach().cpu().numpy())
    
    return losses

if __name__ == "__main__":
    from autoencoders.sae_ensemble import FunctionalSAE
    import torchopt

    torch.set_grad_enabled(False)

    if mp.get_start_method(allow_none=True) != "spawn":
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