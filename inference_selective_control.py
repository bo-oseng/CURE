#!/usr/bin/env python3
"""Remove selected factors from a composite degradation in one step."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

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
    parser.add_argument("--output", required=True, help="Output image or directory")
    parser.add_argument(
        "--source-prompt",
        required=True,
        help="Known composite degradation in the input, for example low_haze",
    )
    parser.add_argument(
        "--remove",
        required=True,
        nargs="+",
        metavar="FACTOR",
        help="Factor or factors to remove, for example: --remove haze",
    )
    add_runtime_arguments(parser)
    return parser.parse_args(argv)


def selective_prompt(source_prompt: str, factors_to_remove: Sequence[str]) -> str:
    if source_prompt not in EMBEDDER_TYPES:
        valid = ", ".join(EMBEDDER_TYPES)
        raise ValueError(f"Unknown --source-prompt {source_prompt!r}; choose one of: {valid}")

    source_factors = source_prompt.split("_")
    if len(source_factors) < 2:
        raise ValueError("--source-prompt must contain at least two degradation factors")
    if len(set(factors_to_remove)) != len(factors_to_remove):
        raise ValueError("--remove contains duplicate factors")

    unknown = [factor for factor in factors_to_remove if factor not in source_factors]
    if unknown:
        raise ValueError(
            f"Cannot remove {unknown} from {source_prompt!r}; available factors: {source_factors}"
        )

    requested = set(factors_to_remove)
    prompt = "_".join(factor for factor in source_factors if factor in requested)
    if prompt not in EMBEDDER_TYPES:
        raise ValueError(f"The selected removal prompt {prompt!r} has no trained embedding")
    return prompt


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    prompt = selective_prompt(args.source_prompt, args.remove)
    print(f"Removing {prompt!r} from {args.source_prompt!r}", flush=True)

    input_path = resolve_input(args.input, args.source_prompt)
    jobs = image_jobs(input_path, args.output)
    restorer, encoder, device = load_runtime(
        args.checkpoint,
        args.embedder_checkpoint,
        args.device,
    )
    embedding = encoder([prompt])

    for index, job in enumerate(jobs, start=1):
        restored = restorer(load_image(job.source, device), embedding)
        save_image(restored, job.destination)
        report_progress(index, len(jobs), job.source, job.destination)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
