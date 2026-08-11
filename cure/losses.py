"""Losses for the 042 baseline and the 049 CURE fine-tuning stage."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    values = [math.exp(-((x - window_size // 2) ** 2) / (2 * sigma**2)) for x in range(window_size)]
    window = torch.tensor(values)
    return window / window.sum()


def _window(window_size: int, channels: int, reference: torch.Tensor) -> torch.Tensor:
    one_dimensional = _gaussian(window_size, 1.5).unsqueeze(1)
    two_dimensional = one_dimensional @ one_dimensional.transpose(0, 1)
    return (
        two_dimensional.float()
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(channels, 1, window_size, window_size)
        .contiguous()
        .to(device=reference.device, dtype=reference.dtype)
    )


def ssim(
    image1: torch.Tensor,
    image2: torch.Tensor,
    *,
    window_size: int = 11,
    value_range: float = 1.0,
    full: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    real_size = min(window_size, image1.shape[-2], image1.shape[-1])
    window = _window(real_size, image1.shape[1], image1)
    mean1 = F.conv2d(image1, window, groups=image1.shape[1])
    mean2 = F.conv2d(image2, window, groups=image2.shape[1])
    mean1_sq = mean1.square()
    mean2_sq = mean2.square()
    mean12 = mean1 * mean2
    variance1 = F.conv2d(image1.square(), window, groups=image1.shape[1]) - mean1_sq
    variance2 = F.conv2d(image2.square(), window, groups=image2.shape[1]) - mean2_sq
    covariance = F.conv2d(image1 * image2, window, groups=image1.shape[1]) - mean12
    c1 = (0.01 * value_range) ** 2
    c2 = (0.03 * value_range) ** 2
    contrast_numerator = 2 * covariance + c2
    contrast_denominator = variance1 + variance2 + c2
    contrast = (contrast_numerator / contrast_denominator).mean()
    similarity = (
        (2 * mean12 + c1) * contrast_numerator / ((mean1_sq + mean2_sq + c1) * contrast_denominator)
    ).mean()
    return (similarity, contrast) if full else similarity


def ms_ssim(image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
    """MS-SSIM matching the historical OneRestore implementation."""

    weights = image1.new_tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    similarities = []
    contrasts = []
    for _ in range(len(weights)):
        similarity, contrast = ssim(image1, image2, full=True)
        similarities.append(similarity)
        contrasts.append(contrast)
        image1 = F.avg_pool2d(image1, 2)
        image2 = F.avg_pool2d(image2, 2)
    similarities = (torch.stack(similarities) + 1) / 2
    contrasts = (torch.stack(contrasts) + 1) / 2
    powered_similarity = similarities**weights
    powered_contrast = contrasts**weights
    # This expression intentionally preserves the baseline implementation.
    return torch.prod(powered_contrast[:-1] * powered_similarity[-1])


class ContrastiveRestorationLoss(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        self.features = models.vgg16(weights=weights).features[:16]
        self.features.requires_grad_(False)
        self.layers = {"3", "8", "15"}

    def _extract(self, image: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for name, layer in self.features._modules.items():
            image = layer(image)
            if name in self.layers:
                outputs.append(image)
        return outputs

    def forward(
        self,
        degraded: torch.Tensor,
        target: torch.Tensor,
        negatives: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        degraded_features = self._extract(degraded)
        target_features = self._extract(target)
        output_features = self._extract(output)
        negative_features = [
            self._extract(negatives[:, index]) for index in range(negatives.shape[1])
        ]
        total = output.new_zeros(())
        denominator_count = len(negative_features) + 1
        for level, (output_feature, target_feature, degraded_feature) in enumerate(
            zip(output_features, target_features, degraded_features, strict=True)
        ):
            positive = F.l1_loss(output_feature, target_feature.detach())
            denominator = F.l1_loss(output_feature, degraded_feature.detach())
            denominator += sum(
                F.l1_loss(output_feature, features[level].detach())
                for features in negative_features
            )
            total += positive / (denominator / denominator_count + 1e-7)
        return total / len(output_features)


def reconstruction_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    *,
    pixel: str = "smooth_l1",
    smooth_weight: float = 0.6,
    ssim_weight: float = 0.3,
) -> torch.Tensor:
    if pixel == "smooth_l1":
        pixel_loss = F.smooth_l1_loss(output, target)
    elif pixel == "l1":
        pixel_loss = F.l1_loss(output, target)
    else:
        raise ValueError(f"Unsupported pixel loss: {pixel}")
    return smooth_weight * pixel_loss + ssim_weight * (1 - ms_ssim(output, target))


class BaselineLoss(nn.Module):
    def __init__(
        self,
        weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
        *,
        pretrained_vgg: bool = True,
    ) -> None:
        super().__init__()
        self.smooth_weight, self.ssim_weight, self.contrast_weight = weights
        self.contrastive = ContrastiveRestorationLoss(pretrained=pretrained_vgg)

    def forward(
        self,
        degraded: torch.Tensor,
        target: torch.Tensor,
        negatives: torch.Tensor,
        output: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        smooth = F.smooth_l1_loss(output, target)
        structural = 1 - ms_ssim(output, target)
        contrast = self.contrastive(degraded, target, negatives, output)
        total = (
            self.smooth_weight * smooth
            + self.ssim_weight * structural
            + self.contrast_weight * contrast
        )
        return total, {
            "baseline": total.detach(),
            "smooth_l1": smooth.detach(),
            "ms_ssim": structural.detach(),
            "contrastive": contrast.detach(),
        }


class CURELoss(nn.Module):
    """Baseline loss plus the four objectives proposed in CURE."""

    def __init__(
        self,
        weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
        *,
        pretrained_vgg: bool = True,
    ) -> None:
        super().__init__()
        self.smooth_weight, self.ssim_weight, _ = weights
        self.baseline = BaselineLoss(weights, pretrained_vgg=pretrained_vgg)

    def forward(
        self,
        *,
        degraded: torch.Tensor,
        clean: torch.Tensor,
        negatives: torch.Tensor,
        output: torch.Tensor,
        identity_input: torch.Tensor,
        identity_output: torch.Tensor,
        partial_outputs: torch.Tensor,
        partial_targets: torch.Tensor,
        sequential_outputs: torch.Tensor,
        ratio_output: torch.Tensor,
        ratio_target: torch.Tensor,
        ratio_twice_output: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        baseline, _ = self.baseline(degraded, clean, negatives, output)

        identity = reconstruction_loss(
            identity_output,
            identity_input,
            pixel="l1",
            smooth_weight=self.smooth_weight,
            ssim_weight=self.ssim_weight,
        )
        ratio = reconstruction_loss(
            ratio_output,
            ratio_target,
            smooth_weight=self.smooth_weight,
            ssim_weight=self.ssim_weight,
        ) + reconstruction_loss(
            ratio_twice_output,
            clean,
            smooth_weight=self.smooth_weight,
            ssim_weight=self.ssim_weight,
        )
        intermediate = sum(
            reconstruction_loss(
                partial_outputs[:, index],
                partial_targets[:, index],
                smooth_weight=self.smooth_weight,
                ssim_weight=self.ssim_weight,
            )
            for index in range(2)
        )
        first, second = sequential_outputs.unbind(dim=1)
        permutation = (
            F.smooth_l1_loss(first, clean)
            + F.smooth_l1_loss(second, clean)
            + F.smooth_l1_loss(first, second)
        )
        total = baseline + identity + ratio + intermediate + permutation
        return total, {
            "baseline": baseline.detach(),
            "identity": identity.detach(),
            "ratio": ratio.detach(),
            "intermediate": intermediate.detach(),
            "permutation": permutation.detach(),
        }
