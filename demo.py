#!/usr/bin/env python3
"""Restore any image or folder and save ready-to-view before/after comparisons."""

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
    save_comparison,
    save_image,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Any input image or directory")
    parser.add_argument("--output", default="outputs/demo", help="Output directory")
    parser.add_argument(
        "--prompt",
        required=True,
        help="Degradation to remove, for example haze or low_haze",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Restoration strength from 0 (identity) to 1 (full restoration)",
    )
    add_runtime_arguments(parser)
    return parser.parse_args(argv)


def validate_strength(value: float) -> float:
    if not 0 <= value <= 1:
        raise ValueError(f"--strength must be between 0 and 1, got {value}")
    return value


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    strength = validate_strength(args.strength)
    base_jobs = image_jobs(args.input, args.output, output_is_directory=True)
    restorer, encoder, device = load_runtime(
        args.checkpoint,
        args.embedder_checkpoint,
        args.device,
    )
    embedding = encoder.ratio([args.prompt], strength)

    output_root = Path(args.output)
    for index, job in enumerate(base_jobs, start=1):
        relative_destination = job.destination.relative_to(output_root)
        restored_destination = output_root / "restored" / relative_destination
        comparison_destination = (output_root / "comparison" / relative_destination).with_suffix(".png")

        restored = restorer(load_image(job.source, device), embedding)
        save_image(restored, restored_destination)
        save_comparison(job.source, restored, comparison_destination)
        report_progress(index, len(base_jobs), job.source, restored_destination)
        print(f"           comparison: {comparison_destination}", flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
