"""BF16-to-NVFP4 quantization (`nvfp4_maxabs`) for block-scaled weight matrices.

For one logical matrix `W[N,K]` the method chooses the FP32 weight divisor
`d_w = RN_fp32(2688 / max|W|)` (2688 = 6 * 448, the largest E2M1 magnitude times the largest
E4M3FN value), so the matrix maximum maps to the top of both grids. Each K group of 16 values then
gets the E4M3FN scale word nearest `RN_fp32(max|group| * d_w / 6)`, ties to even, and each value
the E2M1 word nearest `W * d_w / scale`, ties to even, saturating at 6. The represented weight is
`e2m1(code) * e4m3fn(scale) / d_w` (tensor-formats.md 3.3). A zero matrix takes `d_w = 1` and
zero words; a group whose scale rounds to zero takes zero codes. Codes are packed two per byte,
the even-K value in the low nibble.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .fp8_row import _round_e4m3fn_rne

GROUP = 16
_E2M1_MAX = 6.0
_E4M3FN_MAX = 448.0
_E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E4M3FN_POSITIVE = torch.arange(0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn).double().numpy()


@dataclass(frozen=True, slots=True)
class Nvfp4Words:
    codes: torch.Tensor  # uint8 [N, K/2], even K in the low nibble
    scales: torch.Tensor  # uint8 [N, K/16], natural (unswizzled) E4M3FN words
    divisor: torch.Tensor  # float32 scalar d_w


def _round_e2m1_rne(magnitude: np.ndarray) -> np.ndarray:
    """Nearest E2M1 magnitude word for values in [0, 6], ties to the even word."""

    upper = np.searchsorted(_E2M1_MAGNITUDES, magnitude, side="left").clip(0, 7)
    lower = np.maximum(upper - 1, 0)
    lower_distance = magnitude - _E2M1_MAGNITUDES[lower]
    upper_distance = _E2M1_MAGNITUDES[upper] - magnitude
    choose_upper = (upper_distance < lower_distance) | (
        (upper_distance == lower_distance) & ((upper & 1) == 0)
    )
    return np.where(choose_upper, upper, lower).astype(np.uint8)


def quantize_bf16_nvfp4(weight: torch.Tensor) -> Nvfp4Words:
    """Quantize one BF16 `[N,K]` matrix, K a multiple of 16, to NVFP4 words."""

    if weight.dtype != torch.bfloat16 or weight.dim() != 2:
        raise TypeError("NVFP4 quantization source must be a rank-two BF16 tensor")
    n, k = weight.shape
    if n <= 0 or k <= 0 or k % (2 * GROUP):
        raise ValueError("NVFP4 quantization needs positive N and K a multiple of 32")
    values = weight.detach().to(device="cpu", dtype=torch.float32).numpy().astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("NVFP4 quantization source contains NaN or infinity")

    maximum = float(np.abs(values).max())
    if maximum == 0.0:
        return Nvfp4Words(
            torch.zeros((n, k // 2), dtype=torch.uint8),
            torch.zeros((n, k // GROUP), dtype=torch.uint8),
            torch.tensor(1.0, dtype=torch.float32),
        )
    divisor = np.float32(_E2M1_MAX * _E4M3FN_MAX / maximum)
    if not np.isfinite(divisor) or divisor <= 0.0:
        raise ValueError("NVFP4 weight divisor is not finite and positive")

    groups = values.reshape(n, k // GROUP, GROUP)
    target = (np.abs(groups).max(axis=2) * float(divisor) / _E2M1_MAX).astype(np.float32)
    scale_words = _round_e4m3fn_rne(np.minimum(target, np.float32(_E4M3FN_MAX)))
    scale = _E4M3FN_POSITIVE[scale_words]

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = groups * float(divisor) / scale[:, :, None]
    normalized[scale == 0.0] = 0.0
    words = _round_e2m1_rne(np.minimum(np.abs(normalized), _E2M1_MAX))
    words |= np.where(np.signbit(groups), 0x8, 0).astype(np.uint8)
    words = words.reshape(n, k)
    packed = words[:, 0::2] | (words[:, 1::2] << 4)
    return Nvfp4Words(
        torch.from_numpy(np.ascontiguousarray(packed)),
        torch.from_numpy(np.ascontiguousarray(scale_words)),
        torch.tensor(float(divisor), dtype=torch.float32),
    )


_E2M1_SIGNED = np.concatenate((_E2M1_MAGNITUDES, -_E2M1_MAGNITUDES))


def represented_nvfp4(words: Nvfp4Words) -> np.ndarray:
    """The exact binary64 values the words represent, `[N,K]`."""

    packed = words.codes.numpy()
    codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    scale = np.repeat(_E4M3FN_POSITIVE[words.scales.numpy()], GROUP, axis=1)
    return _E2M1_SIGNED[codes] * scale / float(words.divisor)
