#!/usr/bin/env python3
"""Fine-tune OneRestore with CURE, corresponding to legacy experiment 049."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cure.checkpoint import load_model, save_model
from cure.data import CUREH5Dataset, sample_cure_batch
from cure.distributed import decay_learning_rate, finish, initialize, wrap
from cure.embeddings import PromptEncoder, align_intermediate_targets
from cure.losses import CURELoss
from cure.metrics import psnr
from cure.models import OneRestore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-h5", default="datasets_h5/half_train.h5")
    parser.add_argument(
        "--embedder-checkpoint",
        default="checkpoints/CCDD_half_train/_embedder_model_epoch150.tar",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        default="checkpoints/042_train_half_og_ccdd/OneRestore_model_301.tar",
    )
    parser.add_argument("--resume", default=None, help="Resume a CURE training checkpoint")
    parser.add_argument("--output-dir", default="outputs/049_cure")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=6, help="Per-process batch size")
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--lr-decay-every", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=124)
    parser.add_argument(
        "--fixed-component-order",
        action="store_true",
        help="Disable random low+haze / haze+low ordering (legacy debugging only)",
    )
    parser.add_argument(
        "--no-pretrained-vgg",
        action="store_true",
        help="Do not use ImageNet VGG weights (not paper-equivalent)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.ratio <= 1:
        raise ValueError("--ratio must be between 0 and 1")
    device, rank, world_size = initialize(args.seed)
    component_rng = random.Random(args.seed + rank)
    dataset = CUREH5Dataset(args.train_h5)
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
    load_model(restorer, args.baseline_checkpoint)
    start_epoch = load_model(restorer, args.resume, optimizer=optimizer) if args.resume else 0
    restorer = wrap(restorer, device, world_size)
    objective = CURELoss(pretrained_vgg=not args.no_pretrained_vgg).to(device)

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        decay_learning_rate(optimizer, epoch, args.lr_decay_every)
        restorer.train()
        for step, raw_batch in enumerate(loader, start=1):
            raw_batch = {
                name: tensor.to(device, non_blocking=True) for name, tensor in raw_batch.items()
            }
            batch = sample_cure_batch(raw_batch, dataset.degradation_names)
            prompts = batch["prompts"]
            with torch.no_grad():
                full_embedding = prompt_encoder(prompts)
                identity_embedding = prompt_encoder.identity(len(prompts))
                ratio_embedding = prompt_encoder.ratio(prompts, args.ratio)
                component_embeddings, component_names, swapped = prompt_encoder.components(
                    prompts,
                    randomize_order=not args.fixed_component_order,
                    rng=component_rng,
                )
                partial_targets = align_intermediate_targets(
                    batch["type1"], batch["type2"], swapped
                )

            output = restorer(batch["degraded"], full_embedding)
            identity_output = restorer(batch["identity_input"], identity_embedding)
            ratio_output = restorer(batch["degraded"], ratio_embedding)
            ratio_twice_output = restorer(ratio_output, ratio_embedding)
            partial_first = restorer(batch["degraded"], component_embeddings[:, 0])
            partial_second = restorer(batch["degraded"], component_embeddings[:, 1])
            sequential_first = restorer(partial_first, component_embeddings[:, 1])
            sequential_second = restorer(partial_second, component_embeddings[:, 0])
            partial_outputs = torch.stack((partial_first, partial_second), dim=1)
            sequential_outputs = torch.stack((sequential_first, sequential_second), dim=1)

            loss, terms = objective(
                degraded=batch["degraded"],
                clean=batch["clean"],
                negatives=batch["negatives"],
                output=output,
                identity_input=batch["identity_input"],
                identity_output=identity_output,
                partial_outputs=partial_outputs,
                partial_targets=partial_targets,
                sequential_outputs=sequential_outputs,
                ratio_output=ratio_output,
                ratio_target=batch["half"],
                ratio_twice_output=ratio_twice_output,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if rank == 0:
                order_example = "+".join(component_names[0])
                print(
                    f"epoch={epoch:03d} step={step:04d}/{len(loader):04d} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"loss={loss.item():.4f} psnr={psnr(batch['clean'], output):.2f} "
                    f"order={order_example} "
                    + " ".join(f"{name}={value.item():.4f}" for name, value in terms.items()),
                    flush=True,
                )
        if rank == 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
            save_model(restorer, optimizer, epoch, output_dir / f"model_{epoch:03d}.tar")
    finish()


if __name__ == "__main__":
    main()
