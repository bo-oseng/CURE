"""Shared image-classification helpers for evaluating the OneRestore embedder."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .checkpoint import load_model
from .constants import EMBEDDER_TYPES
from .inference_utils import IMAGE_SUFFIXES
from .models.embedder import EmbeddingClassifier


@dataclass(frozen=True)
class ClassificationMetrics:
    image_count: int
    target_correct: int
    loss_sum: float
    target_probability_sum: float
    clear_probability_sum: float
    prediction_counts: tuple[int, ...]

    @property
    def accuracy(self) -> float:
        return self.target_correct / self.image_count

    @property
    def mean_loss(self) -> float:
        return self.loss_sum / self.image_count

    @property
    def mean_target_probability(self) -> float:
        return self.target_probability_sum / self.image_count

    @property
    def mean_clear_probability(self) -> float:
        return self.clear_probability_sum / self.image_count

    def prediction_distribution(self) -> dict[str, float]:
        return {
            name: count / self.image_count
            for name, count in zip(EMBEDDER_TYPES, self.prediction_counts, strict=True)
        }

    def predominant_prediction(self) -> tuple[str, float]:
        index = max(range(len(self.prediction_counts)), key=self.prediction_counts.__getitem__)
        return EMBEDDER_TYPES[index], self.prediction_counts[index] / self.image_count


class ImagePathDataset(Dataset[torch.Tensor]):
    def __init__(self, paths: Sequence[Path]) -> None:
        self.paths = tuple(paths)
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return self.transform(image)


def validate_type_names(type_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(type_names)
    unknown = [name for name in names if name not in EMBEDDER_TYPES]
    if unknown:
        valid = ", ".join(EMBEDDER_TYPES)
        raise ValueError(f"Unknown degradation classes {unknown}; choose from: {valid}")
    if len(set(names)) != len(names):
        raise ValueError("Degradation classes must not contain duplicates")
    if not names:
        raise ValueError("At least one degradation class is required")
    return names


def image_paths(root: str | Path, max_images: int | None = None) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("--max-images must be positive")
        paths = paths[:max_images]
    if not paths:
        raise ValueError(f"No supported images found under {root}")
    return paths


def load_embedder_classifier(
    checkpoint: str | Path,
    glove: str | Path,
    device: torch.device,
) -> EmbeddingClassifier:
    checkpoint = Path(checkpoint)
    glove = Path(glove)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Embedder checkpoint not found: {checkpoint}")
    if not glove.is_file():
        raise FileNotFoundError(f"GloVe file not found: {glove}")

    # The checkpoint replaces the entire ResNet, so ImageNet initialization is
    # intentionally disabled to avoid an unnecessary network download.
    model = EmbeddingClassifier(glove, pretrained_backbone=False).to(device)
    load_model(model, checkpoint)
    return model.eval()


@torch.inference_mode()
def classify_paths(
    model: EmbeddingClassifier,
    paths: Sequence[Path],
    target_name: str,
    device: torch.device,
    *,
    batch_size: int,
    workers: int,
) -> ClassificationMetrics:
    if target_name not in EMBEDDER_TYPES:
        raise ValueError(f"Unknown target degradation: {target_name!r}")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if workers < 0:
        raise ValueError("--workers must be non-negative")

    target_index = EMBEDDER_TYPES.index(target_name)
    clear_index = EMBEDDER_TYPES.index("clear")
    loader = DataLoader(
        ImagePathDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    counts: Counter[int] = Counter()
    image_count = 0
    target_correct = 0
    loss_sum = 0.0
    target_probability_sum = 0.0
    clear_probability_sum = 0.0

    for images in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        batch_count = images.shape[0]
        targets = torch.full((batch_count,), target_index, device=device, dtype=torch.long)
        probabilities = logits.softmax(dim=1)
        predictions = logits.argmax(dim=1)

        image_count += batch_count
        target_correct += (predictions == target_index).sum().item()
        loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
        target_probability_sum += probabilities[:, target_index].sum().item()
        clear_probability_sum += probabilities[:, clear_index].sum().item()
        counts.update(predictions.cpu().tolist())

    return ClassificationMetrics(
        image_count=image_count,
        target_correct=target_correct,
        loss_sum=loss_sum,
        target_probability_sum=target_probability_sum,
        clear_probability_sum=clear_probability_sum,
        prediction_counts=tuple(counts[index] for index in range(len(EMBEDDER_TYPES))),
    )


def metrics_dict(metrics: ClassificationMetrics) -> dict[str, object]:
    predominant_name, predominant_fraction = metrics.predominant_prediction()
    return {
        "images": metrics.image_count,
        "target_correct": metrics.target_correct,
        "target_accuracy": metrics.accuracy,
        "mean_loss": metrics.mean_loss,
        "mean_target_probability": metrics.mean_target_probability,
        "mean_clear_probability": metrics.mean_clear_probability,
        "predominant_prediction": predominant_name,
        "predominant_fraction": predominant_fraction,
        "prediction_counts": dict(zip(EMBEDDER_TYPES, metrics.prediction_counts, strict=True)),
        "prediction_distribution": metrics.prediction_distribution(),
    }
