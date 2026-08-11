"""Checkpoint I/O compatible with both plain and DDP legacy files."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def load_model(
    model: nn.Module,
    path: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state = {key.removeprefix("module."): value for key, value in state.items()}
    unwrap(model).load_state_dict(state, strict=strict)
    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0


def save_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
