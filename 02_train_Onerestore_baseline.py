#!/usr/bin/env python3
"""Train the OneRestore baseline corresponding to legacy experiment 042."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cure.checkpoint import load_model, save_model
from cure.data import BaselineH5Dataset, sample_baseline_batch
from cure.distributed import decay_learning_rate, finish, initialize, wrap
from cure.embeddings import PromptEncoder
from cure.losses import BaselineLoss
from cure.metrics import psnr
from cure.models import OneRestore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-h5", default="datasets_h5/half_og_train.h5")
    parser.add_argument(
        "--embedder-checkpoint",
        default="checkpoints/CCDD_half_train/_embedder_model_epoch150.tar",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", default="outputs/042_baseline")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32, help="Per-process batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-decay-every", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=124)
    parser.add_argument(
        "--no-pretrained-vgg",
        action="store_true",
        help="Do not use ImageNet VGG weights (not paper-equivalent)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device, rank, world_size = initialize(args.seed)
    dataset = BaselineH5Dataset(args.train_h5)
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    prompt_encoder = PromptEncoder(args.embedder_checkpoint).to(device).eval()
    prompt_encoder.requires_grad_(False)
    restorer = OneRestore().to(device)
    optimizer = torch.optim.Adam(restorer.parameters(), lr=args.learning_rate)
    start_epoch = load_model(restorer, args.resume, optimizer=optimizer) if args.resume else 0
    restorer = wrap(restorer, device, world_size)
    objective = BaselineLoss(pretrained_vgg=not args.no_pretrained_vgg).to(device)

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        decay_learning_rate(optimizer, epoch, args.lr_decay_every)
        restorer.train()
        for step, batch in enumerate(loader, start=1):
            batch = batch.to(device, non_blocking=True)
            clean, degraded, negatives, prompts = sample_baseline_batch(
                batch, dataset.degradation_names
            )
            with torch.no_grad():
                embedding = prompt_encoder(prompts)
            output = restorer(degraded, embedding)
            loss, terms = objective(degraded, clean, negatives, output)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if rank == 0:
                print(
                    f"epoch={epoch:03d} step={step:04d}/{len(loader):04d} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"loss={loss.item():.4f} psnr={psnr(clean, output):.2f} "
                    f"contrast={terms['contrastive'].item():.4f}",
                    flush=True,
                )
        if rank == 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
            save_model(restorer, optimizer, epoch, output_dir / f"model_{epoch:03d}.tar")
    finish()


if __name__ == "__main__":
    main()
