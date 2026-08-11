import math

import torch


def psnr(target: torch.Tensor, output: torch.Tensor, value_range: float = 1.0) -> float:
    mse = torch.mean((target - output) ** 2).item()
    return float("inf") if mse == 0 else 10 * math.log10(value_range**2 / mse)
