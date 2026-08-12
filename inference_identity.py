#!/usr/bin/env python3
"""Run the restorer with the learned identity/no-restoration embedding."""

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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        help="Input image or directory; required unless --source-prompt selects half-test data",
    )
    parser.add_argument("--output", required=True, help="Output image or directory")
    parser.add_argument(
        "--source-prompt",
        help="Input degradation used only to select data/half_test/main_data/<source-prompt>",
    )
    add_runtime_arguments(parser)
    return parser.parse_args(argv)


def identity_input(input_path: str | Path | None, source_prompt: str | None) -> Path:
    if input_path is not None:
        return Path(input_path)
    if source_prompt is None:
        raise ValueError("Pass --input or --source-prompt")
    return resolve_input(None, source_prompt)


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    input_path = identity_input(args.input, args.source_prompt)
    jobs = image_jobs(input_path, args.output)
    restorer, encoder, device = load_runtime(
        args.checkpoint,
        args.embedder_checkpoint,
        args.device,
    )
    embedding = encoder.identity(1)

    for index, job in enumerate(jobs, start=1):
        restored = restorer(load_image(job.source, device), embedding)
        save_image(restored, job.destination)
        report_progress(index, len(jobs), job.source, job.destination)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
