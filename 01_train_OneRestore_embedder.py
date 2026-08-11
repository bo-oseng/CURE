#!/usr/bin/env python3
"""Train the OneRestore visual/text prompt embedder on CCDD-11 class folders."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cure.checkpoint import load_model, save_model
from cure.data import EmbeddingImageDataset
from cure.distributed import decay_learning_rate, finish, initialize, wrap
from cure.models.embedder import EmbeddingClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default="data/half_train/main_data")
    parser.add_argument("--val-dir", default="data/half_test/main_data")
    parser.add_argument("--glove", default="assets/glove.6B.300d.txt")
    parser.add_argument("--output-dir", default="outputs/embedder")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64, help="Per-process batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-decay-every", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=124)
    parser.add_argument(
        "--train-text-branch",
        action="store_true",
        help="Train GloVe/MLP parameters; the historical protocol freezes them",
    )
    parser.add_argument(
        "--no-pretrained-backbone",
        action="store_true",
        help="Do not initialize ResNet-18 with ImageNet weights",
    )
    return parser.parse_args()


@torch.no_grad()
def validate(model, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
        count += labels.numel()
    return total_loss / count, correct / count


def main() -> None:
    args = parse_args()
    device, rank, world_size = initialize(args.seed)
    train_data = EmbeddingImageDataset(args.train_dir, train=True)
    val_data = EmbeddingImageDataset(args.val_dir, train=False)
    sampler = DistributedSampler(train_data, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = EmbeddingClassifier(
        args.glove,
        pretrained_backbone=not args.no_pretrained_backbone,
        freeze_text=not args.train_text_branch,
    ).to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    start_epoch = load_model(model, args.resume, optimizer=optimizer) if args.resume else 0
    model = wrap(model, device, world_size)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        decay_learning_rate(optimizer, epoch, args.lr_decay_every)
        model.train()
        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if rank == 0:
                accuracy = (logits.argmax(1) == labels).float().mean().item()
                print(
                    f"epoch={epoch:03d} step={step:04d}/{len(train_loader):04d} "
                    f"loss={loss.item():.4f} accuracy={accuracy:.3f}",
                    flush=True,
                )
        val_loss, val_accuracy = validate(model, val_loader, device)
        if rank == 0:
            print(f"epoch={epoch:03d} val_loss={val_loss:.4f} val_accuracy={val_accuracy:.3f}")
            if epoch % args.save_every == 0 or epoch == args.epochs:
                save_model(model, optimizer, epoch, output_dir / f"embedder_{epoch:03d}.tar")
    finish()


if __name__ == "__main__":
    main()
