#!/usr/bin/env python3
"""Remove a two-factor degradation sequentially and save both stages."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from cure.constants import EMBEDDER_TYPES
from cure.inference_utils import (
    add_runtime_arguments,
    image_jobs,
    load_image,
    load_runtime,
    report_progress,
    resolve_input,
    save_image,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        help="Input image or directory (default: data/half_test/main_data/<source-prompt>)",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--source-prompt",
        required=True,
        help="Two-factor degradation in the input, for example low_haze",
    )
    parser.add_argument(
        "--sequence",
        nargs=2,
        metavar=("FIRST", "SECOND"),
        help="Removal order; defaults to the factor order in --source-prompt",
    )
    add_runtime_arguments(parser)
    return parser.parse_args(argv)


def restoration_sequence(source_prompt: str, sequence: Sequence[str] | None) -> tuple[str, str]:
    if source_prompt not in EMBEDDER_TYPES:
        valid = ", ".join(EMBEDDER_TYPES)
        raise ValueError(f"Unknown --source-prompt {source_prompt!r}; choose one of: {valid}")
    source_factors = source_prompt.split("_")
    if len(source_factors) != 2:
        raise ValueError("Two-stage inference requires a source prompt with exactly two factors")

    order = tuple(source_factors if sequence is None else sequence)
    if len(order) != 2 or len(set(order)) != 2 or set(order) != set(source_factors):
        raise ValueError(
            f"--sequence must contain exactly {source_factors}, each once; got {list(order)}"
        )
    return order[0], order[1]


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    first, second = restoration_sequence(args.source_prompt, args.sequence)
    input_path = resolve_input(args.input, args.source_prompt)
    base_jobs = image_jobs(input_path, args.output, output_is_directory=True)
    restorer, encoder, device = load_runtime(
        args.checkpoint,
        args.embedder_checkpoint,
        args.device,
    )
    first_embedding, second_embedding = encoder([first, second]).unbind(0)
    first_embedding = first_embedding.unsqueeze(0)
    second_embedding = second_embedding.unsqueeze(0)

    output_root = Path(args.output)
    stage1_root = output_root / f"stage1_{first}"
    stage2_root = output_root / f"stage2_{first}_then_{second}"
    total = len(base_jobs) * 2
    progress = 0
    for job in base_jobs:
        relative_destination = job.destination.relative_to(output_root)
        stage1_destination = stage1_root / relative_destination
        stage2_destination = stage2_root / relative_destination

        stage1 = restorer(load_image(job.source, device), first_embedding)
        save_image(stage1, stage1_destination)
        progress += 1
        report_progress(progress, total, job.source, stage1_destination)

        stage2 = restorer(stage1, second_embedding)
        save_image(stage2, stage2_destination)
        progress += 1
        report_progress(progress, total, job.source, stage2_destination)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
