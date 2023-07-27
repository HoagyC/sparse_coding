import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as data

import torch.multiprocessing as mp

from autoencoders.ensemble import FunctionalEnsemble

import progressbar
import time

import itertools

def job_wrapper(job, ensemble_state_dict, cfg, args, tag, dataset, done_flag, progress_counter):
    ensemble = FunctionalEnsemble.from_state(ensemble_state_dict)

    batch_size = args["batch_size"]
    device = args["device"]
 
    # can't use DataLoaders because they copy the dataset
    # instead, we use a custom sampler for indexes
    sampler = torch.utils.data.BatchSampler(
        torch.utils.data.RandomSampler(range(dataset.shape[0])),
        batch_size=batch_size,
        drop_last=False
    )

    job(ensemble, cfg, args, tag, sampler, dataset, progress_counter)

    done_flag.value = 1

def dispatch_job_on_chunk(ensembles, cfg, dataset, job):
    dataset.pin_memory()
    dataset.share_memory_()
    for ensemble, _, _ in ensembles:
        ensemble.to_shared_memory()
    
    processes = []
    done_flags = []
    progress_counters = []
    n_batches_total = 0
    for ensemble, args, tag in ensembles:
        finished = mp.Value("i", 0)
        progress = mp.Value("i", 0)
        p = mp.Process(target=job_wrapper, args=(job, ensemble.state_dict(), cfg, args, tag, dataset, finished, progress))
        p.start()
        processes.append(p)
        done_flags.append(finished)
        n_batches_total += dataset.shape[0] // args["batch_size"] + 1
        progress_counters.append(progress)

    bar = progressbar.ProgressBar(
        widgets=[
            progressbar.Bar(),
            " ", progressbar.AdaptiveETA(),
            " | ", progressbar.Timer(),
            " | ", *[
                progressbar.Variable(tag, precision=0, width=1, format="{formatted_value}") for _, _, tag in ensembles
            ]
        ], max_value=n_batches_total)

    while True:
        done = [finished.value for finished in done_flags]
        n_batches_done = sum(counter.value for counter in progress_counters)

        bar.update(n_batches_done, **{tag: done.value for done, (_, _, tag) in zip(done_flags, ensembles)})

        if all(done):
            break
            
        time.sleep(0.1)
    
    for p in processes:
        p.join()