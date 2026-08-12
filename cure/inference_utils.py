"""Shared runtime helpers for the standalone inference commands."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

from .checkpoint import load_model
from .embeddings import PromptEncoder
from .models import OneRestore


IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "CURE_restorer.tar"
DEFAULT_EMBEDDER_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "OneRestore_embedder.tar"
DEFAULT_TEST_DATA = PROJECT_ROOT / "data" / "half_test" / "main_data"


@dataclass(frozen=True)
class ImageJob:
    source: Path
    destination: Path


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=(
            "Restorer checkpoint (default: checkpoints/CURE_restorer.tar). "
            "Use checkpoints/OneRestore_restorer.tar for the baseline."
        ),
    )
    parser.add_argument(
        "--embedder-checkpoint",
        default=DEFAULT_EMBEDDER_CHECKPOINT,
        help="OneRestore embedder checkpoint used by the text prompt encoder",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device, for example cpu, cuda, or cuda:1",
    )


def resolve_input(input_path: str | Path | None, prompt: str) -> Path:
    """Use the prompt-specific half-test directory when input is omitted."""

    if input_path is not None:
        return Path(input_path)
    default = DEFAULT_TEST_DATA / prompt
    if not default.is_dir():
        raise FileNotFoundError(
            f"Default input directory not found for prompt {prompt!r}: {default}. "
            "Pass --input explicitly."
        )
    return default


def load_runtime(
    checkpoint: str | Path,
    embedder_checkpoint: str | Path,
    device_name: str,
) -> tuple[OneRestore, PromptEncoder, torch.device]:
    checkpoint = Path(checkpoint)
    embedder_checkpoint = Path(embedder_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Restorer checkpoint not found: {checkpoint}")
    if not embedder_checkpoint.is_file():
        raise FileNotFoundError(f"Embedder checkpoint not found: {embedder_checkpoint}")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device_name!r} was requested, but CUDA is unavailable")

    device = torch.device(device_name)
    restorer = OneRestore().to(device).eval()
    load_model(restorer, checkpoint)
    encoder = PromptEncoder(embedder_checkpoint).to(device).eval()
    return restorer, encoder, device


def image_jobs(
    input_path: str | Path,
    output_path: str | Path,
    *,
    output_is_directory: bool = False,
) -> list[ImageJob]:
    """Map an input image or tree of images to output destinations.

    A directory input is searched recursively and its relative structure is
    preserved. For a file input, ``output_path`` may be an image filename or a
    directory. Set ``output_is_directory`` when the caller always needs an
    output root, such as ratio inference with multiple strength subdirectories.
    """

    source = Path(input_path)
    output = Path(output_path)
    if not source.exists():
        raise FileNotFoundError(f"Input path not found: {source}")

    if source.is_file():
        _validate_image_path(source)
        if output_is_directory or output.suffix.lower() not in IMAGE_SUFFIXES:
            destination = output / source.name
        else:
            destination = output
        _reject_overwrite(source, destination)
        return [ImageJob(source, destination)]

    if not source.is_dir():
        raise ValueError(f"Input must be an image or directory: {source}")
    if output.suffix.lower() in IMAGE_SUFFIXES:
        raise ValueError("--output must be a directory when --input is a directory")

    source_resolved = source.resolve()
    output_resolved = output.resolve()
    if output_resolved.is_relative_to(source_resolved):
        raise ValueError("--output cannot be inside --input for directory inference")

    inputs = sorted(
        path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not inputs:
        supported = ", ".join(sorted(IMAGE_SUFFIXES))
        raise ValueError(f"No supported images found under {source}; expected one of: {supported}")
    return [ImageJob(path, output / path.relative_to(source)) for path in inputs]


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return TF.to_tensor(image).unsqueeze(0).to(device)


def save_image(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(image).save(path)


def save_comparison(source: Path, restored: torch.Tensor, path: Path) -> None:
    """Save the input and restored image side by side."""

    with Image.open(source) as image:
        original = ImageOps.exif_transpose(image).convert("RGB")
    result = tensor_to_pil(restored)
    canvas = Image.new("RGB", (original.width + result.width, max(original.height, result.height)))
    canvas.paste(original, (0, 0))
    canvas.paste(result, (original.width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    output = image.squeeze(0).detach().clamp(0, 1).cpu()
    return TF.to_pil_image(output)


def report_progress(index: int, total: int, source: Path, destination: Path) -> None:
    print(f"[{index}/{total}] {source} -> {destination}", flush=True)


def _validate_image_path(path: Path) -> None:
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        supported = ", ".join(sorted(IMAGE_SUFFIXES))
        raise ValueError(f"Unsupported image extension for {path}; expected one of: {supported}")


def _reject_overwrite(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("--output would overwrite the input image; choose a different path")
