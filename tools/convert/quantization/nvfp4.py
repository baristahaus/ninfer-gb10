"""Round-to-nearest NVFP4 quantization of BF16 expert banks.

Each expert gets one global scale ``amax / (6 * 448)`` and stores the FP32 divisor ``1 / global``;
each 16-value group along K gets an E4M3FN scale ``group_amax / (6 * global)`` rounded to nearest,
and each value the nearest E2M1 code (ties to even) of ``value / (scale / divisor)``, the step the
runtime decodes.
"""

from __future__ import annotations

from typing import Iterator

import torch

from tools.artifact.codecs.nvfp4 import swizzle_nvfp4_scales
from tools.artifact.layouts import align_up

_E2M1_MAX = 6.0
_E4M3_MAX = 448.0
# Boundaries between consecutive E2M1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6. A tie rounds to the
# even code, so it goes up exactly where the upper neighbour has an even index.
_BOUNDARIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIE_UP = (False, True, False, True, False, True, False)


def _e2m1_codes(scaled: torch.Tensor) -> torch.Tensor:
    magnitude = scaled.abs()
    index = torch.zeros_like(magnitude, dtype=torch.uint8)
    for boundary, tie_up in zip(_BOUNDARIES, _TIE_UP):
        index += (magnitude >= boundary if tie_up else magnitude > boundary).to(torch.uint8)
    return index | (scaled < 0).to(torch.uint8) << 3


def quantize_expert(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return packed codes ``[N,K/2]``, natural E4M3FN scales ``[N,K/16]`` and the divisor."""

    if weight.dim() != 2 or weight.shape[1] % 16:
        raise ValueError("NVFP4 expert quantization requires [N,K] with K divisible by 16")
    values = weight.double()
    amax = values.abs().max()
    if not bool(amax > 0):
        raise ValueError("NVFP4 expert has no nonzero values")
    global_scale = amax / (_E2M1_MAX * _E4M3_MAX)
    divisor = global_scale.reciprocal().float()
    groups = values.reshape(values.shape[0], -1, 16)
    group_scale = (groups.abs().amax(dim=2) / (_E2M1_MAX * global_scale)).clamp(max=_E4M3_MAX)
    scales = group_scale.float().to(torch.float8_e4m3fn)
    # Codes are chosen against the step the runtime decodes, scale / divisor.
    step = scales.double() / divisor.double()
    scaled = torch.where(step[..., None] > 0, groups / step[..., None], torch.zeros_like(groups))
    codes = _e2m1_codes(scaled.reshape(values.shape))
    packed = codes[:, 0::2] | codes[:, 1::2] << 4
    return packed, scales.view(torch.uint8), divisor.reshape(())


def quantize_expert_bank(bank: torch.Tensor, device: torch.device) -> Iterator[bytes]:
    """Yield the ``expert_block_scale_k16_m128x4_v1`` payload of a BF16 ``[E,N,K]`` bank."""

    experts, n, k = bank.shape
    codes, scales = [], []
    divisors = torch.empty(experts, dtype=torch.float32)
    for expert in range(experts):
        packed, natural, divisor = quantize_expert(bank[expert].to(device))
        codes.append(packed.cpu())
        scales.append(swizzle_nvfp4_scales(natural.cpu(), (n, k)))
        divisors[expert] = divisor.cpu()
    code_bytes = experts * n * k // 2
    for packed in codes:
        yield packed.numpy().tobytes()
    yield bytes(align_up(code_bytes, 256) - code_bytes)
    for swizzled in scales:
        yield swizzled.numpy().tobytes()
    yield divisors.numpy().tobytes()
