"""Small torchrun helpers with a single-process fallback."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def initialize(seed: int) -> tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    rank = dist.get_rank() if dist.is_initialized() else 0
    process_seed = seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    return device, rank, world_size


def wrap(model: nn.Module, device: torch.device, world_size: int) -> nn.Module:
    if world_size == 1:
        return model
    kwargs = (
        {"device_ids": [device.index], "output_device": device.index}
        if device.type == "cuda"
        else {}
    )
    return DistributedDataParallel(model, **kwargs)


def finish() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def decay_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    frequency: int,
) -> None:
    # Legacy scripts passed a zero-based epoch while displaying epoch + 1.
    zero_based_epoch = epoch - 1
    if zero_based_epoch and zero_based_epoch % frequency == 0:
        for group in optimizer.param_groups:
            group["lr"] *= 0.5
