from pathlib import Path

import pytest

from cure.inference_utils import DEFAULT_TEST_DATA, image_jobs, resolve_input
from demo import validate_strength
from inference_identity import identity_input
from inference_ratio_control import strength_name, validate_strengths
from inference_selective_control import selective_prompt
from inference_twostage import restoration_sequence


def test_selective_prompt_preserves_source_order() -> None:
    assert selective_prompt("low_haze", ["haze"]) == "haze"
    assert selective_prompt("low_haze_rain", ["rain", "low"]) == "low_rain"


def test_selective_prompt_rejects_unavailable_factor() -> None:
    with pytest.raises(ValueError, match="Cannot remove"):
        selective_prompt("low_haze", ["snow"])


def test_ratio_strength_validation_and_names() -> None:
    assert validate_strengths([0, 0.8, 1]) == (0, 0.8, 1)
    assert strength_name(0.8) == "strength_0.8"
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_strengths([1.1])


def test_directory_jobs_preserve_relative_paths(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    nested = input_dir / "low_haze" / "sample.png"
    nested.parent.mkdir(parents=True)
    nested.touch()

    jobs = image_jobs(input_dir, output_dir)

    assert jobs[0].source == nested
    assert jobs[0].destination == output_dir / "low_haze" / "sample.png"


def test_default_input_uses_prompt_directory() -> None:
    assert resolve_input(None, "low_haze") == DEFAULT_TEST_DATA / "low_haze"


def test_identity_input_can_use_explicit_or_half_test_path(tmp_path: Path) -> None:
    assert identity_input(tmp_path, None) == tmp_path
    assert identity_input(None, "low_haze") == DEFAULT_TEST_DATA / "low_haze"
    with pytest.raises(ValueError, match="--input or --source-prompt"):
        identity_input(None, None)


def test_two_stage_order_defaults_and_can_be_reversed() -> None:
    assert restoration_sequence("low_haze", None) == ("low", "haze")
    assert restoration_sequence("low_haze", ["haze", "low"]) == ("haze", "low")
    with pytest.raises(ValueError, match="must contain exactly"):
        restoration_sequence("low_haze", ["low", "snow"])


def test_demo_strength_validation() -> None:
    assert validate_strength(0.75) == 0.75
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_strength(-0.1)
