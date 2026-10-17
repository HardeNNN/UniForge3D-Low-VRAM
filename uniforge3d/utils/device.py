"""Device-memory cleanup shared by the sequential inference stages."""

import gc

import torch


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
