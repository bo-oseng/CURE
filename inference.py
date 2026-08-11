#!/usr/bin/env python3
"""Restore one image using a full, ratio-controlled, or sequential prompt."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from cure.checkpoint import load_model
from cure.embeddings import PromptEncoder
from cure.models import OneRestore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/CURE_restorer.tar",
    )
    parser.add_argument(
        "--embedder-checkpoint",
        default="checkpoints/OneRestore_embedder.tar",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", help="Full prompt, e.g. low_haze")
    group.add_argument("--sequence", nargs="+", help="Ordered single prompts, e.g. low haze")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not 0 <= args.strength <= 1:
        raise ValueError("--strength must be between 0 and 1")
    device = torch.device(args.device)
    model = OneRestore().to(device).eval()
    load_model(model, args.checkpoint)
    encoder = PromptEncoder(args.embedder_checkpoint).to(device).eval()
    with Image.open(args.input) as image:
        restored = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
    prompts = args.sequence if args.sequence is not None else [args.prompt]
    for prompt in prompts:
        embedding = encoder.ratio([prompt], args.strength)
        restored = model(restored, embedding)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(restored.squeeze(0).clamp(0, 1).cpu()).save(output)


if __name__ == "__main__":
    main()
