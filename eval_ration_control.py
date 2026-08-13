#!/usr/bin/env python3
"""Classify ratio-control outputs and measure trends across restoration strengths."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from cure.constants import EMBEDDER_TYPES
from cure.embedder_evaluation import (
    classify_paths,
    image_paths,
    load_embedder_classifier,
    metrics_dict,
    validate_type_names,
)


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RatioDirectory:
    prompt: str
    strength: float
    path: Path


def strength_from_name(name: str) -> float | None:
    if not name.startswith("strength_"):
        return None
    try:
        strength = float(name.removeprefix("strength_"))
    except ValueError:
        return None
    return strength if 0 <= strength <= 1 else None


def legacy_ratio_name(name: str) -> tuple[str, float] | None:
    prompt, separator, percentage = name.rpartition("_")
    if not separator or prompt not in EMBEDDER_TYPES or not percentage.isdigit():
        return None
    value = int(percentage)
    if not 0 <= value <= 100:
        return None
    return prompt, value / 100


def discover_ratio_directories(
    root: str | Path,
    prompts: Sequence[str] | None = None,
    strengths: Sequence[float] | None = None,
) -> list[RatioDirectory]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Ratio-control output directory not found: {root}")
    selected_prompts = set(EMBEDDER_TYPES if prompts is None else validate_type_names(prompts))
    selected_strengths = None if strengths is None else tuple(strengths)
    if selected_strengths is not None:
        invalid = [value for value in selected_strengths if not 0 <= value <= 1]
        if invalid:
            raise ValueError(f"--strengths must be between 0 and 1, got: {invalid}")

    jobs: list[RatioDirectory] = []
    prompt_roots: list[tuple[str, Path]] = []
    if root.name in selected_prompts and any(
        strength_from_name(path.name) is not None for path in root.iterdir() if path.is_dir()
    ):
        prompt_roots.append((root.name, root))
    else:
        prompt_roots.extend(
            (path.name, path)
            for path in root.iterdir()
            if path.is_dir() and path.name in selected_prompts
        )

    for prompt, prompt_root in prompt_roots:
        for path in prompt_root.iterdir():
            strength = strength_from_name(path.name) if path.is_dir() else None
            if strength is not None:
                jobs.append(RatioDirectory(prompt, strength, path))

    # Compatibility with the notebook's historical <prompt>_<percentage>
    # directories, for example low_haze_00 and low_haze_100.
    for path in root.iterdir():
        parsed = legacy_ratio_name(path.name) if path.is_dir() else None
        if parsed is not None and parsed[0] in selected_prompts:
            jobs.append(RatioDirectory(parsed[0], parsed[1], path))

    if selected_strengths is not None:
        jobs = [
            job
            for job in jobs
            if any(abs(job.strength - selected) < 1e-9 for selected in selected_strengths)
        ]
    jobs.sort(key=lambda job: (EMBEDDER_TYPES.index(job.prompt), job.strength, str(job.path)))
    if not jobs:
        raise ValueError(
            f"No ratio outputs found under {root}; expected "
            "<prompt>/strength_<value> or <prompt>_<percentage> directories"
        )
    duplicates = [(a.prompt, a.strength) for a, b in zip(jobs, jobs[1:]) if (a.prompt, a.strength) == (b.prompt, b.strength)]
    if duplicates:
        raise ValueError(f"Duplicate prompt/strength directories found: {duplicates}")
    return jobs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=PROJECT_ROOT / "outputs" / "inference" / "ratio_control",
        help="Ratio output root or a single prompt directory",
    )
    parser.add_argument("--prompts", nargs="+", metavar="PROMPT")
    parser.add_argument("--strengths", type=float, nargs="+", metavar="VALUE")
    parser.add_argument(
        "--checkpoint",
        default=PROJECT_ROOT / "checkpoints" / "OneRestore_embedder.tar",
    )
    parser.add_argument("--glove", default=PROJECT_ROOT / "assets" / "glove.6B.300d.txt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device, for example cpu, cuda, or cuda:1",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Optional maximum number of images per prompt and strength",
    )
    parser.add_argument(
        "--output-dir",
        default=PROJECT_ROOT / "outputs" / "evaluation" / "ratio_control",
        help="Directory for metrics.json and metrics.csv",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    jobs = discover_ratio_directories(args.input, args.prompts, args.strengths)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} was requested, but CUDA is unavailable")
    model = load_embedder_classifier(args.checkpoint, args.glove, device)

    rows: list[dict[str, object]] = []
    print(
        "prompt                strength  images  target_acc  target_prob  clear_prob  predominant",
        flush=True,
    )
    print(
        "--------------------  --------  ------  ----------  -----------  ----------  --------------------",
        flush=True,
    )
    for job in jobs:
        paths = image_paths(job.path, args.max_images)
        metrics = classify_paths(
            model,
            paths,
            job.prompt,
            device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        values = metrics_dict(metrics)
        row = {"prompt": job.prompt, "strength": job.strength, "directory": str(job.path), **values}
        rows.append(row)
        print(
            f"{job.prompt:<20}  {job.strength:>8.3f}  {metrics.image_count:>6}  "
            f"{metrics.accuracy:>10.4f}  {metrics.mean_target_probability:>11.4f}  "
            f"{metrics.mean_clear_probability:>10.4f}  {values['predominant_prediction']}",
            flush=True,
        )

    result: dict[str, object] = {
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "interpretation": (
            "strength is restoration strength: 0 is identity and 1 is full restoration; "
            "target_accuracy measures how often the restored image is still classified as "
            "its source degradation"
        ),
        "results": rows,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    csv_fields = (
        "prompt",
        "strength",
        "images",
        "target_accuracy",
        "mean_loss",
        "mean_target_probability",
        "mean_clear_probability",
        "predominant_prediction",
        "predominant_fraction",
        "directory",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {json_path}", flush=True)
    print(f"saved: {csv_path}", flush=True)
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
