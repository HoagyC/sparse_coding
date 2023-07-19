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

def dispatch_job_on_chunk(ensembles, cfg, args, tags, dataset, job):
    dataset.share_memory_()
    for ensemble in ensembles:
        ensemble.to_shared_memory()
    
    processes = []
    done_flags = []
    progress_counters = []
    n_batches_total = 0
    for i in range(len(args)):
        finished = mp.Value("i", 0)
        progress = mp.Value("i", 0)
        p = mp.Process(target=job_wrapper, args=(job, ensembles[i].state_dict(), cfg, args[i], tags[i], dataset, finished, progress))
        p.start()
        processes.append(p)
        done_flags.append(finished)
        n_batches_total += dataset.shape[0] // args[i]["batch_size"] + 1
        progress_counters.append(progress)
    
    done_width = len(gen_done_text([False] * len(ensembles), tags))

    bar = progressbar.ProgressBar(
        widgets=[
            progressbar.Bar(),
            " ", progressbar.AdaptiveETA(),
            " | ", *[
                progressbar.Variable(tags[i], precision=0, width=1, format="{formatted_value}") for i in range(len(tags))
            ]
        ], max_value=n_batches_total)

    while True:
        done = [finished.value for finished in done_flags]
        n_batches_done = sum(counter.value for counter in progress_counters)
        
        done_text = gen_done_text(done, tags)

        bar.update(n_batches_done, **{tags[i]: done[i] for i in range(len(tags))})

        if all(done):
            break
            
        time.sleep(0.1)
    
    for p in processes:
        p.join()