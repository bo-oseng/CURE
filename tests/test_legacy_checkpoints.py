from pathlib import Path

import pytest

from cure.checkpoint import load_model
from cure.embeddings import PromptEncoder
from cure.models import OneRestore

BASELINE = Path("checkpoints/042_train_half_og_ccdd/OneRestore_model_301.tar")
EMBEDDER = Path("checkpoints/CCDD_half_train/_embedder_model_epoch150.tar")


@pytest.mark.skipif(not BASELINE.exists(), reason="local legacy checkpoint is not linked")
def test_baseline_checkpoint_loads_strictly() -> None:
    load_model(OneRestore(), BASELINE, strict=True)


@pytest.mark.skipif(not EMBEDDER.exists(), reason="local embedder checkpoint is not linked")
def test_text_encoder_loads_legacy_checkpoint() -> None:
    encoder = PromptEncoder(EMBEDDER)
    assert encoder(["clear", "low_haze"]).shape == (2, 324)
