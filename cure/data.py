"""Datasets and batch sampling shared by the released training scripts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .constants import BASELINE_TYPES, EMBEDDER_TYPES, RESTORATION_TYPES


def _import_h5py():
    try:
        import h5py
    except (ImportError, ValueError) as error:
        raise RuntimeError(
            "h5py could not be imported. Install the pinned requirements; in particular, "
            "use NumPy < 2 with binary h5py builds made for NumPy 1.x."
        ) from error
    return h5py


class _H5Dataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._handle = None
        h5py = _import_h5py()
        with h5py.File(self.path, "r") as handle:
            self.keys = sorted(handle.keys(), key=lambda value: int(value))
            encoded_names = handle.attrs.get("degradation_names")
        self.degradation_names = (
            tuple(json.loads(encoded_names)) if encoded_names is not None else None
        )

    def __len__(self) -> int:
        return len(self.keys)

    def _file(self):
        if self._handle is None:
            self._handle = _import_h5py().File(self.path, "r")
        return self._handle

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        if self._handle is not None:
            self._handle.close()


class BaselineH5Dataset(_H5Dataset):
    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        if self.degradation_names is None:
            self.degradation_names = tuple(BASELINE_TYPES)

    def __getitem__(self, index: int) -> torch.Tensor:
        array = np.asarray(self._file()[self.keys[index]], dtype=np.float32)
        return torch.from_numpy(array)


class CUREH5Dataset(_H5Dataset):
    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        if self.degradation_names is None:
            self.degradation_names = tuple(RESTORATION_TYPES)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        group = self._file()[self.keys[index]]
        return {
            name: torch.from_numpy(np.asarray(group[name], dtype=np.float32))
            for name in ("zero", "gt", "lr", "type1", "type2", "half")
        }


class EmbeddingImageDataset(Dataset):
    """Class-folder dataset for training the visual/text prompt embedder."""

    def __init__(
        self,
        root: str | Path,
        type_names: Sequence[str] = EMBEDDER_TYPES,
        *,
        train: bool,
    ) -> None:
        self.root = Path(root)
        self.type_names = tuple(type_names)
        transform_list: list[object]
        if train:
            transform_list = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            transform_list = [transforms.Resize((224, 224)), transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)
        self.samples: list[tuple[Path, int]] = []
        for label, type_name in enumerate(self.type_names):
            directory = self.root / type_name
            if not directory.is_dir():
                raise FileNotFoundError(f"Missing embedding class directory: {directory}")
            for path in sorted(directory.iterdir()):
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    self.samples.append((path, label))
        if not self.samples:
            raise RuntimeError(f"No images found below {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


def sample_baseline_batch(
    batch: torch.Tensor,
    degradation_names: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Select one positive degradation and use all other classes as negatives."""

    clean = batch[:, 0]
    degraded = batch[:, 1:]
    batch_size, class_count = degraded.shape[:2]
    if class_count != len(degradation_names):
        raise ValueError(
            f"HDF5 has {class_count} degradation classes, vocabulary has {len(degradation_names)}"
        )
    selected_index = torch.randint(class_count, (batch_size,), device=batch.device)
    row_index = torch.arange(batch_size, device=batch.device)
    selected = degraded[row_index, selected_index]
    mask = torch.ones(batch_size, class_count, dtype=torch.bool, device=batch.device)
    mask[row_index, selected_index] = False
    negatives = degraded[mask].reshape(batch_size, class_count - 1, *degraded.shape[2:])
    prompts = [degradation_names[index] for index in selected_index.cpu().tolist()]
    return clean, selected, negatives, prompts


def sample_cure_batch(
    batch: dict[str, torch.Tensor],
    degradation_names: Sequence[str],
) -> dict[str, torch.Tensor | list[str]]:
    degraded = batch["lr"]
    batch_size, class_count = degraded.shape[:2]
    if class_count != len(degradation_names):
        raise ValueError(
            f"HDF5 has {class_count} degradation classes, vocabulary has {len(degradation_names)}"
        )
    device = degraded.device
    selected_index = torch.randint(class_count, (batch_size,), device=device)
    rows = torch.arange(batch_size, device=device)
    mask = torch.ones(batch_size, class_count, dtype=torch.bool, device=device)
    mask[rows, selected_index] = False
    return {
        "clean": batch["gt"],
        "identity_input": batch["zero"],
        "degraded": degraded[rows, selected_index],
        "negatives": degraded[mask].reshape(batch_size, class_count - 1, *degraded.shape[2:]),
        "type1": batch["type1"][rows, selected_index],
        "type2": batch["type2"][rows, selected_index],
        "half": batch["half"][rows, selected_index],
        "prompts": [degradation_names[index] for index in selected_index.cpu().tolist()],
    }
