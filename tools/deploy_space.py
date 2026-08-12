#!/usr/bin/env python3
"""Create or update the minimal CURE Gradio repository on HF Spaces."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPACE_FILES = {
    "app.py": PROJECT_ROOT / "app.py",
    "README.md": PROJECT_ROOT / "hf_space" / "README.md",
    "requirements.txt": PROJECT_ROOT / "hf_space" / "requirements.txt",
    "cure/__init__.py": PROJECT_ROOT / "cure" / "__init__.py",
    "cure/checkpoint.py": PROJECT_ROOT / "cure" / "checkpoint.py",
    "cure/constants.py": PROJECT_ROOT / "cure" / "constants.py",
    "cure/embeddings.py": PROJECT_ROOT / "cure" / "embeddings.py",
    "cure/models/__init__.py": PROJECT_ROOT / "cure" / "models" / "__init__.py",
    "cure/models/onerestore.py": PROJECT_ROOT / "cure" / "models" / "onerestore.py",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face Space ID, for example ses7720/CURE-Demo",
    )
    parser.add_argument("--private", action="store_true", help="Create a private Space")
    parser.add_argument(
        "--commit-message",
        default="Update CURE Gradio demo",
        help="Commit message used on the Space repository",
    )
    return parser.parse_args(argv)


def deployment_operations() -> list[CommitOperationAdd]:
    missing = [str(path) for path in SPACE_FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Space deployment files are missing: {missing}")
    return [
        CommitOperationAdd(path_in_repo=destination, path_or_fileobj=source)
        for destination, source in SPACE_FILES.items()
    ]


def main() -> None:
    args = parse_args()
    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=args.private,
        exist_ok=True,
    )
    commit = api.create_commit(
        repo_id=args.repo_id,
        repo_type="space",
        operations=deployment_operations(),
        commit_message=args.commit_message,
    )
    print(f"Space source updated: {commit.repo_url}")
    print(f"Demo URL: https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
