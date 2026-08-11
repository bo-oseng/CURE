"""Text embedding loading, interpolation, and prompt decomposition."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from .constants import EMBEDDER_TYPES, EMBEDDING_DIM, IDENTITY_NAME


def _checkpoint_state(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")
    return {key.removeprefix("module."): value for key, value in checkpoint.items()}


class PromptEncoder(nn.Module):
    """The lightweight text branch of the pretrained OneRestore embedder.

    Restoration never uses the visual branch when prompts are known. Loading
    only these three tensors avoids constructing a ResNet for every training
    process while remaining numerically identical to the legacy text encoder.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        type_names: Sequence[str] = EMBEDDER_TYPES,
        word_dim: int = 300,
        output_dim: int = EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.type_names = tuple(type_names)
        self.type_to_index = {name: index for index, name in enumerate(self.type_names)}
        self.output_dim = output_dim
        self.embedder = nn.Embedding(len(self.type_names), word_dim)
        self.mlp = nn.Sequential(nn.Linear(word_dim, output_dim), nn.ReLU(inplace=True))
        if checkpoint is not None:
            self.load_legacy_checkpoint(checkpoint)

    def load_legacy_checkpoint(self, path: str | Path) -> None:
        state = _checkpoint_state(path)
        keys = ("embedder.weight", "mlp.0.weight", "mlp.0.bias")
        missing = [key for key in keys if key not in state]
        if missing:
            raise KeyError(f"Text tensors missing from {path}: {missing}")
        self.load_state_dict({key: state[key] for key in keys}, strict=True)

    def forward(self, prompts: Sequence[str]) -> torch.Tensor:
        try:
            indices = [self.type_to_index[prompt] for prompt in prompts]
        except KeyError as error:
            valid = ", ".join(self.type_names)
            raise ValueError(f"Unknown prompt {error.args[0]!r}; choose one of: {valid}") from error
        index = torch.tensor(indices, device=self.embedder.weight.device)
        return self.mlp(self.embedder(index))

    def identity(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        reference = self.embedder.weight
        return torch.ones(
            batch_size,
            self.output_dim,
            device=reference.device if device is None else device,
            dtype=reference.dtype if dtype is None else dtype,
        )

    def ratio(self, prompts: Sequence[str], strength: float | torch.Tensor) -> torch.Tensor:
        """Interpolate from identity (0) to full restoration (1)."""

        full = self(prompts)
        identity = self.identity(len(prompts), device=full.device, dtype=full.dtype)
        if isinstance(strength, torch.Tensor):
            strength = strength.to(device=full.device, dtype=full.dtype)
            while strength.ndim < full.ndim:
                strength = strength.unsqueeze(-1)
        return torch.lerp(identity, full, strength)

    def components(
        self,
        prompts: Sequence[str],
        *,
        randomize_order: bool = False,
        rng: random.Random | None = None,
    ) -> tuple[torch.Tensor, list[tuple[str, str]], torch.Tensor]:
        pairs, swapped = decompose_prompts(prompts, randomize_order=randomize_order, rng=rng)
        flat_names = [name for pair in pairs for name in pair if name != IDENTITY_NAME]
        flat_embeddings = iter(self(flat_names))
        identity = self.identity(1).squeeze(0)
        rows = []
        for pair in pairs:
            rows.append(
                torch.stack(
                    [identity if name == IDENTITY_NAME else next(flat_embeddings) for name in pair]
                )
            )
        return torch.stack(rows), pairs, swapped.to(self.embedder.weight.device)


def decompose_prompts(
    prompts: Sequence[str],
    *,
    randomize_order: bool = False,
    rng: random.Random | None = None,
) -> tuple[list[tuple[str, str]], torch.Tensor]:
    """Split each supported prompt into two restoration operations.

    A single degradation is paired with the identity operation. A two-factor
    prompt is randomly presented in either order during CURE training. Triple
    prompts are intentionally rejected because the paper excludes them from the
    fine-tuning set.
    """

    rng = random if rng is None else rng
    pairs: list[tuple[str, str]] = []
    swapped: list[bool] = []
    for prompt in prompts:
        parts = prompt.split("_")
        if len(parts) == 1:
            pair = (parts[0], IDENTITY_NAME)
        elif len(parts) == 2:
            pair = (parts[0], parts[1])
        else:
            raise ValueError(
                f"CURE fine-tuning expects one or two factors, got {prompt!r}. "
                "Triple degradations are evaluation-only."
            )
        do_swap = bool(randomize_order and rng.random() < 0.5)
        pairs.append(pair[::-1] if do_swap else pair)
        swapped.append(do_swap)
    return pairs, torch.tensor(swapped, dtype=torch.bool)


def align_intermediate_targets(
    type1_images: torch.Tensor,
    type2_images: torch.Tensor,
    swapped: torch.Tensor,
) -> torch.Tensor:
    """Align partial GT images with a possibly permuted component order.

    In CCDD, ``type1`` retains the first named degradation. Consequently,
    removing component 1 targets ``type2`` and removing component 2 targets
    ``type1``. When component order is swapped, the target order must swap too.
    """

    targets = torch.stack((type2_images, type1_images), dim=1)
    swapped = swapped.to(device=targets.device)
    if swapped.any():
        targets[swapped] = targets[swapped].flip(1)
    return targets
