#!/usr/bin/env python3
"""Run full-strength, one-step restoration on an image or image directory."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

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
        help="Input image or directory (default: data/half_test/main_data/<prompt>)",
    )
    parser.add_argument("--output", required=True, help="Output image or directory")
    parser.add_argument(
        "--prompt",
        required=True,
        help="Degradation to remove in one step, for example haze or low_haze",
    )
    add_runtime_arguments(parser)
    return parser.parse_args(argv)


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    input_path = resolve_input(args.input, args.prompt)
    jobs = image_jobs(input_path, args.output)
    restorer, encoder, device = load_runtime(
        args.checkpoint,
        args.embedder_checkpoint,
        args.device,
    )
    embedding = encoder([args.prompt])

    for index, job in enumerate(jobs, start=1):
        restored = restorer(load_image(job.source, device), embedding)
        save_image(restored, job.destination)
        report_progress(index, len(jobs), job.source, job.destination)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
