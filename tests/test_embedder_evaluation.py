from pathlib import Path

import pytest

from cure.embedder_evaluation import image_paths, validate_type_names
from eval_ration_control import (
    discover_ratio_directories,
    legacy_ratio_name,
    strength_from_name,
)


def test_validate_type_names() -> None:
    assert validate_type_names(["low", "low_haze"]) == ("low", "low_haze")
    with pytest.raises(ValueError, match="Unknown degradation"):
        validate_type_names(["blur"])


def test_image_paths_are_recursive_and_limited(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "b.png").touch()
    (tmp_path / "nested" / "a.jpg").touch()
    (tmp_path / "ignore.txt").touch()

    assert image_paths(tmp_path) == [tmp_path / "b.png", tmp_path / "nested" / "a.jpg"]
    assert image_paths(tmp_path, max_images=1) == [tmp_path / "b.png"]


def test_strength_directory_names() -> None:
    assert strength_from_name("strength_0.8") == 0.8
    assert strength_from_name("strength_1") == 1
    assert strength_from_name("stage_0.8") is None
    assert legacy_ratio_name("low_haze_20") == ("low_haze", 0.2)


def test_discover_current_ratio_layout(tmp_path: Path) -> None:
    for name in ("strength_1", "strength_0", "strength_0.5"):
        (tmp_path / "low_haze" / name).mkdir(parents=True)

    jobs = discover_ratio_directories(tmp_path, prompts=["low_haze"])

    assert [(job.prompt, job.strength) for job in jobs] == [
        ("low_haze", 0),
        ("low_haze", 0.5),
        ("low_haze", 1),
    ]


def test_discover_legacy_ratio_layout_and_filter(tmp_path: Path) -> None:
    (tmp_path / "haze_00").mkdir()
    (tmp_path / "haze_50").mkdir()
    (tmp_path / "haze_100").mkdir()

    jobs = discover_ratio_directories(tmp_path, prompts=["haze"], strengths=[0.5])

    assert len(jobs) == 1
    assert jobs[0].prompt == "haze"
    assert jobs[0].strength == 0.5
