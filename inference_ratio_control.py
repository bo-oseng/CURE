#!/usr/bin/env python3
"""Restore images at one or more explicit prompt strengths."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from cure.inference_utils import (
    add_runtime_arguments,
    image_jobs,
    load_image,
    load_runtime,
    report_progress,
    resolve_input,
    save_image,
)


DEFAULT_STRENGTHS = tuple(index / 10 for index in range(11))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        help="Input image or directory (default: data/half_test/main_data/<prompt>)",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--prompt",
        required=True,
        help="Degradation to remove, for example haze or low_haze",
    )
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=DEFAULT_STRENGTHS,
        metavar="VALUE",
        help=(
            "Prompt strengths between 0 and 1. "
            "0 is identity/no restoration and 1 is full restoration "
            "(default: 0.0 through 1.0 in steps of 0.1)"
        ),
    )
    add_runtime_arguments(parser)
    return parser.parse_args(argv)


def validate_strengths(values: Sequence[float]) -> tuple[float, ...]:
    strengths = tuple(values)
    invalid = [value for value in strengths if not 0 <= value <= 1]
    if invalid:
        raise ValueError(f"Strengths must be between 0 and 1, got: {invalid}")
    if len(set(strengths)) != len(strengths):
        raise ValueError("--strengths contains duplicate values")
    return strengths


def strength_name(value: float) -> str:
    return f"strength_{value:.6f}".rstrip("0").rstrip(".")


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    strengths = validate_strengths(args.strengths)
    input_path = resolve_input(args.input, args.prompt)
    base_jobs = image_jobs(input_path, args.output, output_is_directory=True)
    restorer, encoder, device = load_runtime(
        args.checkpoint,
        args.embedder_checkpoint,
        args.device,
    )
    embeddings = [(strength, encoder.ratio([args.prompt], strength)) for strength in strengths]

    total = len(base_jobs) * len(embeddings)
    progress = 0
    output_root = Path(args.output)
    for job in base_jobs:
        image = load_image(job.source, device)
        relative_destination = job.destination.relative_to(output_root)
        for strength, embedding in embeddings:
            destination = output_root / strength_name(strength) / relative_destination
            restored = restorer(image, embedding)
            save_image(restored, destination)
            progress += 1
            report_progress(progress, total, job.source, destination)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
