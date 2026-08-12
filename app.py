#!/usr/bin/env python3
"""Gradio interface for CURE controllable image restoration."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

from cure.checkpoint import load_model
from cure.constants import EMBEDDER_TYPES
from cure.embeddings import PromptEncoder
from cure.models import OneRestore

try:
    import spaces as hf_spaces
except ImportError:  # The package is injected by the ZeroGPU runtime on Spaces.
    hf_spaces = None


MODEL_REPO = os.environ.get("CURE_MODEL_REPO", "ses7720/CURE")
MODEL_REVISION = os.environ.get("CURE_MODEL_REVISION", "main")
PROJECT_ROOT = Path(__file__).resolve().parent
MAX_IMAGE_SIDE = max(64, int(os.environ.get("CURE_MAX_IMAGE_SIDE", "1024")))

RESTORATION_PROMPTS = tuple(name for name in EMBEDDER_TYPES if name != "clear")
COMPOSITE_PROMPTS = tuple(name for name in RESTORATION_PROMPTS if "_" in name)
TWO_FACTOR_PROMPTS = tuple(name for name in COMPOSITE_PROMPTS if len(name.split("_")) == 2)
RATIO_STRENGTHS = tuple(f"{value / 10:.1f}" for value in range(11))
INFERENCE_LOCK = threading.Lock()


def zerogpu(duration: int):
    """Use a ZeroGPU allocation on Spaces and remain a no-op for local runs."""

    if hf_spaces is None:
        return lambda function: function
    return hf_spaces.GPU(duration=duration)


@dataclass(frozen=True)
class Runtime:
    restorer: OneRestore
    encoder: PromptEncoder
    device: torch.device


def _checkpoint_path(filename: str, environment_name: str) -> Path:
    """Prefer an explicit/local checkpoint and otherwise use the Hub cache."""

    override = os.environ.get(environment_name)
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{environment_name} does not point to a file: {path}")
        return path

    local = PROJECT_ROOT / "checkpoints" / filename
    if local.is_file():
        return local

    return Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
            revision=MODEL_REVISION,
            token=os.environ.get("HF_TOKEN"),
        )
    )


@lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    """Load the restorer and prompt encoder once for all Gradio tabs."""

    device_name = os.environ.get(
        "CURE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device_name!r} was requested, but CUDA is unavailable")

    device = torch.device(device_name)
    restorer = OneRestore().to(device).eval()
    load_model(restorer, _checkpoint_path("CURE_restorer.tar", "CURE_CHECKPOINT"))
    encoder = PromptEncoder(
        _checkpoint_path("OneRestore_embedder.tar", "CURE_EMBEDDER_CHECKPOINT")
    ).to(device).eval()
    return Runtime(restorer=restorer, encoder=encoder, device=device)


def _prepare_image(image: Image.Image | None, device: torch.device) -> tuple[torch.Tensor, str]:
    if image is None:
        raise gr.Error("먼저 입력 이미지를 업로드해 주세요.")

    image = ImageOps.exif_transpose(image).convert("RGB")
    original_size = image.size
    largest_side = max(original_size)
    if largest_side > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / largest_side
        resized_size = tuple(max(1, round(side * scale)) for side in original_size)
        image = image.resize(resized_size, Image.Resampling.LANCZOS)

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    if image.size == original_size:
        size_message = f"입력 해상도 {original_size[0]}×{original_size[1]}"
    else:
        size_message = (
            f"입력 해상도 {original_size[0]}×{original_size[1]}를 "
            f"{image.width}×{image.height}로 축소"
        )
    return tensor, size_message


def _to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        image.squeeze(0)
        .detach()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _status(runtime: Runtime, size_message: str, operation: str) -> str:
    return f"{operation} 완료 · {size_message} · device={runtime.device}"


@zerogpu(duration=60)
def run_main(image: Image.Image | None, prompt: str) -> tuple[Image.Image, str]:
    runtime = get_runtime()
    tensor, size_message = _prepare_image(image, runtime.device)
    with INFERENCE_LOCK, torch.inference_mode():
        restored = runtime.restorer(tensor, runtime.encoder([prompt]))
    return _to_pil(restored), _status(runtime, size_message, f"{prompt} 전체 복원")


@zerogpu(duration=180)
def run_ratio(
    image: Image.Image | None,
    prompt: str,
    selected_strengths: Sequence[str] | None,
    progress: gr.Progress = gr.Progress(),
) -> tuple[list[tuple[Image.Image, str]], str]:
    if not selected_strengths:
        raise gr.Error("비교할 strength를 하나 이상 선택해 주세요.")

    strengths = tuple(float(value) for value in selected_strengths)
    if any(not 0 <= value <= 1 for value in strengths):
        raise gr.Error("strength는 0과 1 사이여야 합니다.")

    runtime = get_runtime()
    tensor, size_message = _prepare_image(image, runtime.device)
    results: list[tuple[Image.Image, str]] = []
    with INFERENCE_LOCK, torch.inference_mode():
        for index, strength in enumerate(strengths):
            progress(index / len(strengths), desc=f"strength={strength:g}")
            embedding = runtime.encoder.ratio([prompt], strength)
            restored = runtime.restorer(tensor, embedding)
            results.append((_to_pil(restored), f"strength={strength:g}"))
    progress(1.0, desc="완료")
    values = ", ".join(f"{value:g}" for value in strengths)
    return results, _status(runtime, size_message, f"{prompt} ratio [{values}]")


def _selective_prompt(source_prompt: str, factors: Sequence[str] | None) -> str:
    source_factors = source_prompt.split("_")
    if not factors:
        raise gr.Error("제거할 degradation factor를 하나 이상 선택해 주세요.")
    if any(factor not in source_factors for factor in factors):
        raise gr.Error("선택한 factor가 source degradation에 포함되어 있지 않습니다.")

    selected = set(factors)
    prompt = "_".join(factor for factor in source_factors if factor in selected)
    if prompt not in EMBEDDER_TYPES:
        raise gr.Error(f"학습된 prompt embedding이 없습니다: {prompt}")
    return prompt


def selective_factor_update(source_prompt: str) -> dict:
    factors = source_prompt.split("_")
    return gr.update(choices=factors, value=[factors[-1]])


@zerogpu(duration=60)
def run_selective(
    image: Image.Image | None,
    source_prompt: str,
    factors: Sequence[str] | None,
) -> tuple[Image.Image, str]:
    prompt = _selective_prompt(source_prompt, factors)
    runtime = get_runtime()
    tensor, size_message = _prepare_image(image, runtime.device)
    with INFERENCE_LOCK, torch.inference_mode():
        restored = runtime.restorer(tensor, runtime.encoder([prompt]))
    return _to_pil(restored), _status(
        runtime, size_message, f"{source_prompt}에서 {prompt} 선택 제거"
    )


@zerogpu(duration=60)
def run_identity(image: Image.Image | None) -> tuple[Image.Image, str]:
    runtime = get_runtime()
    tensor, size_message = _prepare_image(image, runtime.device)
    with INFERENCE_LOCK, torch.inference_mode():
        restored = runtime.restorer(tensor, runtime.encoder.identity(1))
    return _to_pil(restored), _status(runtime, size_message, "identity/no-restoration")


def _order_choices(source_prompt: str) -> tuple[str, str]:
    first, second = source_prompt.split("_")
    return f"{first} → {second}", f"{second} → {first}"


def two_stage_order_update(source_prompt: str) -> dict:
    choices = _order_choices(source_prompt)
    return gr.update(choices=choices, value=choices[0])


@zerogpu(duration=120)
def run_twostage(
    image: Image.Image | None,
    source_prompt: str,
    order: str,
) -> tuple[Image.Image, Image.Image, str]:
    choices = _order_choices(source_prompt)
    if order not in choices:
        raise gr.Error(f"올바른 복원 순서를 선택해 주세요: {choices}")
    first, second = (part.strip() for part in order.split("→"))

    runtime = get_runtime()
    tensor, size_message = _prepare_image(image, runtime.device)
    with INFERENCE_LOCK, torch.inference_mode():
        embeddings = runtime.encoder([first, second])
        stage1 = runtime.restorer(tensor, embeddings[0].unsqueeze(0))
        stage2 = runtime.restorer(stage1, embeddings[1].unsqueeze(0))
    return (
        _to_pil(stage1),
        _to_pil(stage2),
        _status(runtime, size_message, f"{first} 제거 후 {second} 제거"),
    )


def _image_input() -> gr.Image:
    return gr.Image(
        label="Input image",
        sources=["upload", "clipboard"],
        type="pil",
        image_mode="RGB",
    )


def _image_output(label: str) -> gr.Image:
    return gr.Image(label=label, type="pil", format="png", interactive=False)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="CURE · Controllable Image Restoration") as interface:
        gr.Markdown(
            """
            # CURE: Controllable Unified Image Restoration

            Restore an image in one step, control restoration strength, remove only selected
            degradation factors, test identity behavior, or choose a two-stage restoration order.
            The public demo limits the longest input side to 1024 px to control GPU memory usage.
            """
        )

        with gr.Tab("Main · one step"):
            with gr.Row():
                main_input = _image_input()
                main_output = _image_output("Restored image")
            main_prompt = gr.Dropdown(
                RESTORATION_PROMPTS,
                value="low_haze",
                label="Degradation to remove",
            )
            main_button = gr.Button("Restore", variant="primary")
            main_status = gr.Textbox(label="Run information", interactive=False)
            main_button.click(
                run_main,
                [main_input, main_prompt],
                [main_output, main_status],
                api_name="main",
            )

        with gr.Tab("Ratio control"):
            ratio_input = _image_input()
            with gr.Row():
                ratio_prompt = gr.Dropdown(
                    RESTORATION_PROMPTS,
                    value="low_haze",
                    label="Degradation to remove",
                )
                ratio_strengths = gr.CheckboxGroup(
                    RATIO_STRENGTHS,
                    value=["0.0", "0.5", "1.0"],
                    label="Strengths (0 = identity, 1 = full)",
                )
            ratio_button = gr.Button("Compare strengths", variant="primary")
            ratio_gallery = gr.Gallery(
                label="Ratio-controlled results",
                columns=3,
                object_fit="contain",
                show_download_button=True,
            )
            ratio_status = gr.Textbox(label="Run information", interactive=False)
            ratio_button.click(
                run_ratio,
                [ratio_input, ratio_prompt, ratio_strengths],
                [ratio_gallery, ratio_status],
                api_name="ratio_control",
            )

        with gr.Tab("Selective control"):
            with gr.Row():
                selective_input = _image_input()
                selective_output = _image_output("Selectively restored image")
            with gr.Row():
                selective_source = gr.Dropdown(
                    COMPOSITE_PROMPTS,
                    value="low_haze",
                    label="Known source degradation",
                )
                selective_factors = gr.CheckboxGroup(
                    ["low", "haze"],
                    value=["haze"],
                    label="Factors to remove",
                )
            selective_source.change(
                selective_factor_update,
                selective_source,
                selective_factors,
            )
            selective_button = gr.Button("Remove selected factors", variant="primary")
            selective_status = gr.Textbox(label="Run information", interactive=False)
            selective_button.click(
                run_selective,
                [selective_input, selective_source, selective_factors],
                [selective_output, selective_status],
                api_name="selective_control",
            )

        with gr.Tab("Identity"):
            gr.Markdown(
                "Run the learned identity/no-restoration condition. This is useful for checking "
                "how closely the restorer preserves its input."
            )
            with gr.Row():
                identity_input = _image_input()
                identity_output = _image_output("Identity output")
            identity_button = gr.Button("Run identity", variant="primary")
            identity_status = gr.Textbox(label="Run information", interactive=False)
            identity_button.click(
                run_identity,
                identity_input,
                [identity_output, identity_status],
                api_name="identity",
            )

        with gr.Tab("Two stage"):
            with gr.Row():
                twostage_input = _image_input()
                stage1_output = _image_output("Stage 1")
                stage2_output = _image_output("Stage 2 · final")
            with gr.Row():
                twostage_source = gr.Dropdown(
                    TWO_FACTOR_PROMPTS,
                    value="low_haze",
                    label="Two-factor source degradation",
                )
                initial_orders = _order_choices("low_haze")
                twostage_order = gr.Radio(
                    initial_orders,
                    value=initial_orders[0],
                    label="Removal order",
                )
            twostage_source.change(
                two_stage_order_update,
                twostage_source,
                twostage_order,
            )
            twostage_button = gr.Button("Run two stages", variant="primary")
            twostage_status = gr.Textbox(label="Run information", interactive=False)
            twostage_button.click(
                run_twostage,
                [twostage_input, twostage_source, twostage_order],
                [stage1_output, stage2_output, twostage_status],
                api_name="two_stage",
            )

        gr.Markdown(
            "[Project page](https://bo-oseng.github.io/CURE/) · "
            "[Code](https://github.com/bo-oseng/CURE) · "
            "[Model weights](https://huggingface.co/ses7720/CURE)"
        )

    return interface.queue(max_size=8, default_concurrency_limit=1)


demo = build_demo()


if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))
