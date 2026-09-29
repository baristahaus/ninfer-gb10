"""Exact oracle for the `nvfp4_maxabs` BF16-to-NVFP4 method.

The oracle evaluates the method's definition value by value in Python binary64, searching every
E4M3FN and E2M1 word for the nearest one (ties to the even word), and checks the method's words
exactly. It shares no arithmetic with the vectorized implementation.
"""

from __future__ import annotations

import math
import struct

import torch

from tools.artifact.codecs.nvfp4 import decode_nvfp4_words, encode_nvfp4
from tools.artifact.formats import decode_e2m1_word, decode_e4m3fn_word
from tools.convert.quantization.nvfp4 import quantize_bf16_nvfp4

_E4M3FN = [(word, decode_e4m3fn_word(word)) for word in range(0x7F)]
_E2M1 = [(word, decode_e2m1_word(word)) for word in range(8)]


def _fp32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _nearest(target: float, table: list[tuple[int, float]]) -> int:
    best_word, best_distance = None, math.inf
    for word, value in table:
        distance = abs(value - target)
        if distance < best_distance or (distance == best_distance and word % 2 == 0):
            best_word, best_distance = word, distance
    return best_word


def _oracle(matrix: list[list[float]]) -> tuple[list[list[int]], list[list[int]], float]:
    maximum = max(abs(value) for row in matrix for value in row)
    if maximum == 0.0:
        return ([[0] * len(row) for row in matrix],
                [[0] * (len(row) // 16) for row in matrix], 1.0)
    divisor = _fp32(6.0 * 448.0 / maximum)
    words, scales = [], []
    for row in matrix:
        row_words, row_scales = [], []
        for begin in range(0, len(row), 16):
            group = row[begin : begin + 16]
            target = _fp32(max(abs(value) for value in group) * divisor / 6.0)
            scale_word = _nearest(min(target, 448.0), _E4M3FN)
            scale = decode_e4m3fn_word(scale_word)
            row_scales.append(scale_word)
            for value in group:
                magnitude = 0.0 if scale == 0.0 else min(abs(value) * divisor / scale, 6.0)
                code = _nearest(magnitude, _E2M1)
                row_words.append(code | (8 if math.copysign(1.0, value) < 0 else 0))
        words.append(row_words)
        scales.append(row_scales)
    return words, scales, divisor


def _unpack(codes: torch.Tensor) -> list[list[int]]:
    rows = []
    for row in codes.tolist():
        values = []
        for byte in row:
            values.extend((byte & 0xF, byte >> 4))
        rows.append(values)
    return rows


def _check(source: torch.Tensor) -> None:
    quantized = quantize_bf16_nvfp4(source)
    words, scales, divisor = _oracle(source.float().tolist())
    assert _unpack(quantized.codes) == words
    assert quantized.scales.tolist() == scales
    assert quantized.divisor.item() == divisor


def test_random_matrix_matches_the_oracle_exactly() -> None:
    generator = torch.Generator().manual_seed(7)
    columns = torch.logspace(-4, 1, 64)  # groups spanning four decades of magnitude
    source = (torch.randn(8, 64, generator=generator) * columns).bfloat16()
    _check(source)


def test_ties_signed_zero_zero_groups_and_saturation() -> None:
    row = [6.0, 0.25, -0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.0, 0.0, 4.0, -6.0, 0.5, 3.0, 1.5, 2.0]
    tiny = [1.0e-30] + [0.0] * 15  # its scale rounds to zero: codes are zero
    source = torch.tensor([row + tiny, [0.0] * 32], dtype=torch.bfloat16)
    _check(source)
    assert quantize_bf16_nvfp4(torch.zeros(1, 32, dtype=torch.bfloat16)).divisor.item() == 1.0


def test_words_encode_as_a_valid_nvfp4_matrix() -> None:
    source = torch.randn(128, 64, generator=torch.Generator().manual_seed(3)).bfloat16()
    quantized = quantize_bf16_nvfp4(source)
    payload = encode_nvfp4(quantized.codes, quantized.scales, quantized.divisor, (128, 64))
    codes, scales, divisor = decode_nvfp4_words(payload, (128, 64))
    assert torch.equal(codes, quantized.codes)
    assert torch.equal(scales, quantized.scales)
    assert divisor.item() == quantized.divisor.item()
    represented = torch.tensor(
        [
            [decode_e2m1_word(word) * decode_e4m3fn_word(scale_row[column // 16])
             / divisor.item() for column, word in enumerate(word_row)]
            for word_row, scale_row in zip(_unpack(codes), scales.tolist())
        ]
    )
    error = (represented - source.float()).square().sum().sqrt() / source.float().square().sum().sqrt()
    assert error.item() < 0.12  # E2M1 with 16-value groups on Gaussian data: about 0.09
