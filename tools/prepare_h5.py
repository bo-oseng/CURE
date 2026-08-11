#!/usr/bin/env python3
"""Build the aligned patch databases consumed by experiments 042 and 049."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Allow ``python tools/prepare_h5.py`` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure.constants import BASELINE_TYPES, RESTORATION_TYPES


def import_h5py():
    try:
        import h5py
    except (ImportError, ValueError) as error:
        raise RuntimeError("Install requirements.txt (NumPy < 2 is required).") from error
    return h5py


def read_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0


def patch_positions(height: int, width: int, size: int, stride: int):
    if height < size or width < size:
        raise ValueError(f"Image {height}x{width} is smaller than patch size {size}")
    rows = max(1, int(np.ceil((height - size) / stride + 1)))
    columns = max(1, int(np.ceil((width - size) / stride + 1)))
    for row in range(rows):
        for column in range(columns):
            yield min(row * stride, height - size), min(column * stride, width - size)


def augment(array: np.ndarray, mode: int) -> np.ndarray:
    rotations = mode // 2
    if rotations:
        array = np.rot90(array, rotations, axes=(-2, -1))
    if mode % 2:
        array = np.flip(array, axis=-2)
    return np.ascontiguousarray(array)


def crop(array: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    return array[..., top : top + size, left : left + size]


def prepare_baseline(args: argparse.Namespace) -> None:
    h5py = import_h5py()
    root = Path(args.data_root) / "main_data"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = tuple(args.types)
    files = sorted((root / "clear").glob("*.png"))
    with h5py.File(output, "w") as handle:
        handle.attrs["degradation_names"] = json.dumps(names)
        index = 0
        for clean_path in tqdm(files, desc="baseline images"):
            arrays = np.stack(
                [read_image(clean_path)]
                + [read_image(root / name / clean_path.name) for name in names]
            )
            height, width = arrays.shape[-2:]
            for top, left in patch_positions(height, width, args.patch_size, args.stride):
                patch = augment(crop(arrays, top, left, args.patch_size), random.randrange(8))
                handle.create_dataset(str(index), data=patch, compression=args.compression)
                index += 1
    print(f"Wrote {index} patches to {output}")


def prepare_cure(args: argparse.Namespace) -> None:
    h5py = import_h5py()
    data_root = Path(args.data_root)
    main_root = data_root / "main_data"
    sub_root = data_root / "sub_data"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = tuple(args.types)
    files = sorted((main_root / "clear").glob("*.png"))
    with h5py.File(output, "w") as handle:
        handle.attrs["degradation_names"] = json.dumps(names)
        index = 0
        for clean_path in tqdm(files, desc="CURE images"):
            stem = clean_path.stem
            clean = read_image(clean_path)
            identity_input = read_image(main_root / "zero" / clean_path.name)
            full_images = []
            type1_images = []
            type2_images = []
            half_images = []
            for degradation in names:
                parts = degradation.split("_")
                if len(parts) == 1:
                    type1, type2 = parts[0], "zero"
                elif len(parts) == 2:
                    type1, type2 = parts
                else:
                    raise ValueError(f"CURE training accepts at most two factors: {degradation}")
                subdir = sub_root / degradation / stem
                full_images.append(read_image(main_root / degradation / clean_path.name))
                type1_images.append(read_image(subdir / f"{stem}_{type1}_.png"))
                type2_images.append(read_image(subdir / f"{stem}_{type2}_.png"))
                half_images.append(read_image(subdir / f"{stem}_half_.png"))
            arrays = {
                "gt": clean,
                "zero": identity_input,
                "lr": np.stack(full_images),
                "type1": np.stack(type1_images),
                "type2": np.stack(type2_images),
                "half": np.stack(half_images),
            }
            height, width = clean.shape[-2:]
            for top, left in patch_positions(height, width, args.patch_size, args.stride):
                mode = random.randrange(8)
                group = handle.create_group(str(index))
                for name, array in arrays.items():
                    patch = augment(crop(array, top, left, args.patch_size), mode)
                    group.create_dataset(
                        name, data=patch, dtype=np.float32, compression=args.compression
                    )
                index += 1
    print(f"Wrote {index} patches to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode, default_output, default_types in (
        ("baseline", "datasets_h5/half_og_train.h5", BASELINE_TYPES),
        ("cure", "datasets_h5/half_train.h5", RESTORATION_TYPES),
    ):
        child = subparsers.add_parser(mode)
        child.add_argument("--data-root", default="data/half_train")
        child.add_argument("--output", default=default_output)
        child.add_argument("--patch-size", type=int, default=256)
        child.add_argument("--stride", type=int, default=200)
        child.add_argument("--types", nargs="+", default=default_types)
        child.add_argument("--compression", choices=("gzip", "lzf"), default=None)
        child.add_argument("--seed", type=int, default=124)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    if args.mode == "baseline":
        prepare_baseline(args)
    else:
        prepare_cure(args)


if __name__ == "__main__":
    main()
