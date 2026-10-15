"""Exact MXFP8 code and UE8M0 scale-plane packing and decoding.

``dense_block_scale_k32_mxfp8_v1`` stores verbatim E4M3FN weight words in a
row-major code plane followed by one unsigned 8-bit biased-exponent scale word
per K-axis group of 32. No numerical conversion happens in either direction.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..formats import decode_e4m3fn_word
from ..layouts import dense_block_scale_geometry
from ._tensor_bytes import (
    Payload,
    _exact_uint8_matrix,
    _payload_length,
    _payload_tensor,
)


def encode_mxfp8(
    codes: torch.Tensor, scale_words: torch.Tensor, shape: Sequence[int]
) -> bytes:
    """Encode exact source E4M3FN words and UE8M0 exponent words."""

    geometry = dense_block_scale_geometry("mxfp8", shape)
    code_plane = _exact_uint8_matrix(
        codes, (geometry.n, geometry.k), "MXFP8 codes"
    )
    scales = _exact_uint8_matrix(
        scale_words,
        (geometry.n, geometry.groups_per_row),
        "MXFP8 scale words",
    )
    payload = bytearray(geometry.payload_bytes)
    payload[: geometry.code_plane_bytes] = code_plane.numpy().tobytes()
    payload[
        geometry.scale_plane_offset : geometry.scale_plane_offset
        + geometry.scale_plane_bytes
    ] = scales.numpy().tobytes()
    return bytes(payload)


def decode_mxfp8_words(
    payload: Payload, shape: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode exact E4M3FN code and UE8M0 scale words from a payload."""

    geometry = dense_block_scale_geometry("mxfp8", shape)
    if _payload_length(payload) != geometry.payload_bytes:
        raise ValueError(
            f"MXFP8 payload has {_payload_length(payload)} bytes, "
            f"expected {geometry.payload_bytes}"
        )
    raw = _payload_tensor(payload, torch.device("cpu"))
    codes = raw[: geometry.code_plane_bytes].clone().reshape(
        geometry.n, geometry.k
    )
    scales = raw[
        geometry.scale_plane_offset : geometry.scale_plane_offset
        + geometry.scale_plane_bytes
    ].clone().reshape(geometry.n, geometry.groups_per_row)
    return codes, scales


def represented_mxfp8(codes: torch.Tensor, scale_words: torch.Tensor) -> torch.Tensor:
    """Exact float64 represented weights of MXFP8 codes and UE8M0 scale words.

    Reference implementation for oracles: ``E4M3(word) * 2^(e - 127)``, every
    product computed exactly in binary64 from the decoded words.
    """

    codes = codes.detach().cpu().reshape(-1)
    scales = scale_words.detach().cpu().reshape(-1)
    if scales.numel() * 32 != codes.numel():
        raise ValueError("MXFP8 scale words must cover groups of 32 codes")
    table = torch.tensor(
        [decode_e4m3fn_word(word) for word in range(256)], dtype=torch.float64
    )
    values = table[codes.long()]
    exponents = torch.pow(2.0, scales.double() - 127.0)
    return values * exponents.repeat_interleave(32)
