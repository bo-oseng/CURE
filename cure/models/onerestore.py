"""OneRestore backbone used by both the 042 baseline and CURE.

Module attribute names intentionally match the original implementation so that
the released legacy checkpoints can be loaded without remapping model keys.
"""

import math
import numbers

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F


def _to_3d(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def _to_4d(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=height, w=width)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("LayerNorm expects a one-dimensional shape")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(variance + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("LayerNorm expects a one-dimensional shape")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(variance + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, layer_norm_type: str) -> None:
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), height, width)


class Cross_Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool, q_dim: int = 324) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.query_size = int(math.sqrt(q_dim))
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Linear(q_dim, q_dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(
            dim * 2,
            dim * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        q = self.q(query)
        k, value = self.kv_dwconv(self.kv(x)).chunk(2, dim=1)
        k = F.interpolate(
            k,
            size=(self.query_size, self.query_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        q = repeat(
            q,
            "b l -> b head c l",
            head=self.num_heads,
            c=self.dim // self.num_heads,
        )
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        value = rearrange(
            value,
            "b (head c) h w -> b head c (h w)",
            head=self.num_heads,
        )
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attention = (q @ k.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)
        out = attention @ value
        out = rearrange(
            out,
            "b head c (h w) -> b (head c) h w",
            head=self.num_heads,
            h=height,
            w=width,
        )
        return self.project_out(out)


class Self_Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        q, k, value = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        value = rearrange(
            value,
            "b (head c) h w -> b head c (h w)",
            head=self.num_heads,
        )
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = (q @ k.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)
        out = attention @ value
        out = rearrange(
            out,
            "b head c (h w) -> b (head c) h w",
            head=self.num_heads,
            h=height,
            w=width,
        )
        return self.project_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool) -> None:
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.cross_attn = Cross_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.self_attn = Self_Attention(dim, num_heads, bias)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        x = x + self.cross_attn(self.norm1(x), query)
        x = x + self.self_attn(self.norm2(x))
        return x + self.ffn(self.norm3(x))


class ResidualBlock(nn.Module):
    def __init__(self, channel: int, norm: bool = False) -> None:
        del norm  # Kept in the signature for legacy compatibility.
        super().__init__()
        self.el = TransformerBlock(
            channel,
            num_heads=8,
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",
        )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        return self.el(x, embedding)


class encoder(nn.Module):
    def __init__(self, channel: int) -> None:
        super().__init__()
        self.el = ResidualBlock(channel)
        self.em = ResidualBlock(channel * 2)
        self.es = ResidualBlock(channel * 4)
        self.ess = ResidualBlock(channel * 8)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_eltem = nn.Conv2d(channel, channel * 2, 1, bias=False)
        self.conv_emtes = nn.Conv2d(channel * 2, channel * 4, 1, bias=False)
        self.conv_estess = nn.Conv2d(channel * 4, channel * 8, 1, bias=False)
        # Unused by the original forward pass, but present in released checkpoints.
        self.conv_esstesss = nn.Conv2d(channel * 8, channel * 16, 1, bias=False)

    def forward(
        self, x: torch.Tensor, embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        elout = self.el(x, embedding)
        emout = self.em(self.conv_eltem(self.maxpool(elout)), embedding)
        esout = self.es(self.conv_emtes(self.maxpool(emout)), embedding)
        essout = self.ess(self.conv_estess(self.maxpool(esout)), embedding)
        return elout, emout, esout, essout


class backbone(nn.Module):
    def __init__(self, channel: int) -> None:
        super().__init__()
        self.s1 = ResidualBlock(channel * 8)
        self.s2 = ResidualBlock(channel * 8)

    def forward(
        self,
        x: torch.Tensor,
        embedding: torch.Tensor,
        share1_noise: torch.Tensor | None = None,
        share2_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        share1 = self.s1(x, embedding if share1_noise is None else share1_noise)
        return self.s2(share1, embedding if share2_noise is None else share2_noise)


class decoder(nn.Module):
    def __init__(self, channel: int) -> None:
        super().__init__()
        self.dss = ResidualBlock(channel * 8)
        self.ds = ResidualBlock(channel * 4)
        self.dm = ResidualBlock(channel * 2)
        self.dl = ResidualBlock(channel)
        self.conv_dsstds = nn.Conv2d(channel * 8, channel * 4, 1, bias=False)
        self.conv_dstdm = nn.Conv2d(channel * 4, channel * 2, 1, bias=False)
        self.conv_dmtdl = nn.Conv2d(channel * 2, channel, 1, bias=False)

    @staticmethod
    def _upsample(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_ss: torch.Tensor,
        x_s: torch.Tensor,
        x_m: torch.Tensor,
        x_l: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        dssout = self.dss(x + x_ss, embedding)
        dsout = self.ds(self.conv_dsstds(self._upsample(dssout, x_s)) + x_s, embedding)
        dmout = self.dm(self.conv_dstdm(self._upsample(dsout, x_m)) + x_m, embedding)
        return self.dl(self.conv_dmtdl(self._upsample(dmout, x_l)) + x_l, embedding)


class OneRestore(nn.Module):
    def __init__(self, channel: int = 32) -> None:
        super().__init__()
        self.norm = lambda x: (x - 0.5) / 0.5
        self.denorm = lambda x: (x + 1) / 2
        self.in_conv = nn.Conv2d(3, channel, 1, bias=False)
        self.encoder = encoder(channel)
        self.middle = backbone(channel)
        self.decoder = decoder(channel)
        self.out_conv = nn.Conv2d(channel, 3, 1, bias=False)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        x_l, x_m, x_s, x_ss = self.encoder(self.in_conv(self.norm(x)), embedding)
        x_mid = self.middle(x_ss, embedding)
        x_out = self.decoder(x_mid, x_ss, x_s, x_m, x_l, embedding)
        return self.denorm(self.out_conv(x_out) + x)
