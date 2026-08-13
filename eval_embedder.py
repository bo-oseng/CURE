#!/usr/bin/env python3
"""Evaluate the OneRestore visual embedder on degradation class folders."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=PROJECT_ROOT / "data" / "half_test" / "main_data",
        help="Root containing one directory per degradation class",
    )
    parser.add_argument(
        "--checkpoint",
        default=PROJECT_ROOT / "checkpoints" / "OneRestore_embedder.tar",
    )
    parser.add_argument("--glove", default=PROJECT_ROOT / "assets" / "glove.6B.300d.txt")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=EMBEDDER_TYPES,
        metavar="CLASS",
        help="Classes to evaluate (default: all 12 CCDD-11 classes including clear)",
    )
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
        help="Optional maximum number of images per class, useful for a smoke test",
    )
    parser.add_argument(
        "--output",
        default=PROJECT_ROOT / "outputs" / "evaluation" / "embedder" / "metrics.json",
        help="JSON metrics destination",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    classes = validate_type_names(args.classes)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} was requested, but CUDA is unavailable")
    model = load_embedder_classifier(args.checkpoint, args.glove, device)

    root = Path(args.input)
    per_class: dict[str, dict[str, object]] = {}
    total_images = 0
    total_correct = 0
    total_loss = 0.0
    print("class                 images   accuracy       loss  predominant", flush=True)
    print("--------------------  ------  ---------  ---------  --------------------", flush=True)
    for name in classes:
        paths = image_paths(root / name, args.max_images)
        metrics = classify_paths(
            model,
            paths,
            name,
            device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        values = metrics_dict(metrics)
        per_class[name] = values
        total_images += metrics.image_count
        total_correct += metrics.target_correct
        total_loss += metrics.loss_sum
        print(
            f"{name:<20}  {metrics.image_count:>6}  {metrics.accuracy:>9.4f}  "
            f"{metrics.mean_loss:>9.4f}  {values['predominant_prediction']}",
            flush=True,
        )

    result: dict[str, object] = {
        "input": str(root),
        "checkpoint": str(args.checkpoint),
        "classes": list(classes),
        "images": total_images,
        "overall_accuracy": total_correct / total_images,
        "mean_loss": total_loss / total_images,
        "per_class": per_class,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"overall: images={total_images} accuracy={result['overall_accuracy']:.4f} "
        f"loss={result['mean_loss']:.4f}",
        flush=True,
    )
    print(f"saved: {output}", flush=True)
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
