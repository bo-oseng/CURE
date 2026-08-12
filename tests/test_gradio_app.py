from PIL import Image
import pytest

import app
from tools.deploy_space import SPACE_FILES, deployment_operations


def test_space_deployment_contains_only_runtime_files() -> None:
    assert set(SPACE_FILES) == {
        "app.py",
        "README.md",
        "requirements.txt",
        "cure/__init__.py",
        "cure/checkpoint.py",
        "cure/constants.py",
        "cure/embeddings.py",
        "cure/models/__init__.py",
        "cure/models/onerestore.py",
    }
    assert all(path.is_file() for path in SPACE_FILES.values())
    assert len(deployment_operations()) == len(SPACE_FILES)


def test_selective_prompt_preserves_source_factor_order() -> None:
    assert app._selective_prompt("low_haze_rain", ["rain", "low"]) == "low_rain"
    with pytest.raises(app.gr.Error, match="하나 이상"):
        app._selective_prompt("low_haze", [])


def test_two_stage_orders_cover_both_directions() -> None:
    assert app._order_choices("low_haze") == ("low → haze", "haze → low")


def test_image_preparation_limits_the_longest_side() -> None:
    image = Image.new("RGB", (app.MAX_IMAGE_SIDE * 2, app.MAX_IMAGE_SIDE), "white")
    tensor, message = app._prepare_image(image, app.torch.device("cpu"))
    assert tensor.shape == (1, 3, app.MAX_IMAGE_SIDE // 2, app.MAX_IMAGE_SIDE)
    assert "축소" in message
