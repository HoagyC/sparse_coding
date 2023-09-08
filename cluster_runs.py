import itertools
import sys
import time

import progressbar
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from autoencoders.ensemble import FunctionalEnsemble


def job_wrapper(job, ensemble_state_dict, cfg, args, tag, dataset, done_flag, progress_counter):
    if not sys.warnoptions:
        import warnings

        warnings.filterwarnings("ignore")

    ensemble = FunctionalEnsemble.from_state(ensemble_state_dict)

    batch_size = args["batch_size"]
    device = args["device"]

    # can't use DataLoaders because they copy the dataset
    # instead, we use a custom sampler for indexes
    sampler = torch.utils.data.BatchSampler(
        torch.utils.data.RandomSampler(range(dataset.shape[0])),
        batch_size=batch_size,
        drop_last=False,
    )

    job(ensemble, cfg, args, tag, sampler, dataset, progress_counter)

    done_flag.value = 1

def job_wrapper_lite(ensemble_state_dict, cfg, args, tag, done_flag, progress_counter, job):
    if not sys.warnoptions:
        import warnings

        warnings.filterwarnings("ignore")

    ensemble = FunctionalEnsemble.from_state(ensemble_state_dict)

    job(ensemble, cfg, args, tag, progress_counter)

    done_flag.value = 1

def dispatch_lite(cfg, ensemble, args, name, job):
    ensemble.to_shared_memory()

    finished = mp.Value("i", 0)
    progress = mp.Value("f", 0)

    p = mp.Process(
        target=job_wrapper_lite,
        args=(ensemble.state_dict(), cfg, args, name, finished, progress, job),
    )

    p.start()

    return p, finished, progress

def statusbar_lite(processes, n_points=1000):
    # initialize progress bar
    bar = progressbar.ProgressBar(
        widgets=[
            progressbar.Bar(),
            " ",
            progressbar.AdaptiveETA(),
            " | ",
            progressbar.Timer(),
            " | ",
            *[progressbar.Variable(tag, precision=0, width=1, format="{formatted_value}") for (_, _, _), _, tag in processes],
        ],
        max_value=n_points,
    )

    return bar

def update_statusbar_lite(bar, processes, n_points=1000):
    sum_progress = sum(progress.value for (_, _, progress), _, _ in processes)
    mean_progress = sum_progress / len(processes)
    progress_count = int(mean_progress * n_points)

    bar.update(progress_count, **{tag: done.value for (_, done, _), _, tag in processes})

def collect_lite(processes):
    all_done = all(done.value == 1 for (_, done, _), _, _ in processes)

    if all_done:
        for (p, _, _), _, _ in processes:
            p.join()

        return True
    else:
        return False

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
        p = mp.Process(
            target=job_wrapper,
            args=(
                job,
                ensemble.state_dict(),
                cfg,
                args,
                tag,
                dataset,
                finished,
                progress,
            ),
        )
        p.start()
        processes.append(p)
        done_flags.append(finished)
        n_batches_total += dataset.shape[0] // args["batch_size"] + 1
        progress_counters.append(progress)

    bar = progressbar.ProgressBar(
        widgets=[
            progressbar.Bar(),
            " ",
            progressbar.AdaptiveETA(),
            " | ",
            progressbar.Timer(),
            " | ",
            *[progressbar.Variable(tag, precision=0, width=1, format="{formatted_value}") for _, _, tag in ensembles],
        ],
        max_value=n_batches_total,
    )

    while True:
        done = [finished.value for finished in done_flags]
        n_batches_done = sum(counter.value for counter in progress_counters)

        bar.update(n_batches_done, **{tag: done.value for done, (_, _, tag) in zip(done_flags, ensembles)})

        if all(done):
            break

        time.sleep(0.1)

    for p in processes:
        p.join()
