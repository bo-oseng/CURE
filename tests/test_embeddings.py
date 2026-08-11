import random

import torch

from cure.embeddings import align_intermediate_targets, decompose_prompts


class AlwaysSwap(random.Random):
    def random(self) -> float:
        return 0.0


def test_single_prompt_pairs_with_identity() -> None:
    pairs, swapped = decompose_prompts(["low"])
    assert pairs == [("low", "identity")]
    assert swapped.tolist() == [False]


def test_composite_prompt_can_reverse_order() -> None:
    pairs, swapped = decompose_prompts(["low_haze"], randomize_order=True, rng=AlwaysSwap())
    assert pairs == [("haze", "low")]
    assert swapped.tolist() == [True]


def test_intermediate_targets_follow_component_order() -> None:
    low_remaining = torch.tensor([[[[1.0]]]])
    haze_remaining = torch.tensor([[[[2.0]]]])
    targets = align_intermediate_targets(low_remaining, haze_remaining, torch.tensor([False]))
    assert targets[:, 0].item() == 2.0  # remove low -> haze remains
    assert targets[:, 1].item() == 1.0  # remove haze -> low remains

    swapped = align_intermediate_targets(low_remaining, haze_remaining, torch.tensor([True]))
    assert swapped[:, 0].item() == 1.0
    assert swapped[:, 1].item() == 2.0


def test_triple_prompt_is_evaluation_only() -> None:
    try:
        decompose_prompts(["low_haze_rain"])
    except ValueError as error:
        assert "evaluation-only" in str(error)
    else:
        raise AssertionError("triple prompt should have been rejected")
