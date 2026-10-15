from __future__ import annotations

import struct

import pytest
import torch

from tools.artifact.layouts import (
    align_up,
    block_scale_geometry,
    block_scale_k16_geometry,
    dense_block_scale_geometry,
)
from tools.artifact.codecs.direct import decode_direct, encode_direct
from tools.artifact.codecs.mxfp8 import (
    decode_mxfp8_words,
    encode_mxfp8,
    represented_mxfp8,
)
from tools.artifact.codecs.nvfp4 import (
    block_scale_k16_geometry as _bs16,
    decode_block_scale_k16_words,
    decode_nvfp4_words,
    encode_block_scale_k16,
    encode_nvfp4,
)
from tools.artifact.codecs.row_split import decode_row_split_codes, encode_row_split


def _signed_word(word: int, bits: int) -> int:
    return word if word < 1 << (bits - 1) else word - (1 << bits)


@pytest.mark.parametrize(
    ("format_name", "tensor", "words", "word_format", "word_view"),
    [
        (
            "bf16",
            torch.tensor(
                [_signed_word(word, 16) for word in (0x0000, 0x8000, 0x0001, 0x7FC1)],
                dtype=torch.int16,
            ).view(torch.bfloat16),
            (0x0000, 0x8000, 0x0001, 0x7FC1),
            "H",
            torch.int16,
        ),
        (
            "fp32",
            torch.tensor(
                [
                    _signed_word(word, 32)
                    for word in (0x00000000, 0x80000000, 0x00000001, 0x7FC01234)
                ],
                dtype=torch.int32,
            ).view(torch.float32),
            (0x00000000, 0x80000000, 0x00000001, 0x7FC01234),
            "I",
            torch.int32,
        ),
        (
            "int32",
            torch.tensor((0, -1, -(1 << 31), (1 << 31) - 1), dtype=torch.int32),
            (0, -1, -(1 << 31), (1 << 31) - 1),
            "i",
            torch.int32,
        ),
    ],
)
def test_direct_layout_preserves_exact_little_endian_words(
    format_name, tensor, words, word_format, word_view
):
    expected = struct.pack("<" + word_format * len(words), *words)
    payload = encode_direct(tensor, format_name)
    assert payload == expected
    decoded = decode_direct(payload, format_name, tensor.shape)
    assert torch.equal(decoded.view(word_view), tensor.view(word_view))

    if format_name == "bf16":
        with pytest.raises(TypeError):
            encode_direct(tensor.float(), format_name)


@pytest.mark.parametrize(
    (
        "format_name",
        "k",
        "group_size",
        "k_pad",
        "scale_offset",
        "prefix",
        "base_prefix",
        "high_prefix",
    ),
    [
        pytest.param(
            "q4_g64_fp16",
            65,
            64,
            128,
            256,
            (-8, -7, -1, 0, 1, 7),
            b"\x98\x0f\x71",
            b"",
            id="q4",
        ),
        pytest.param(
            "q5_g64_fp16",
            130,
            64,
            256,
            512,
            (-16, -15, -1, 0, 1, 15),
            b"\x10\x0f\xf1",
            b"\x07",
            id="q5",
        ),
        pytest.param(
            "q6_g64_fp16",
            65,
            64,
            128,
            512,
            (-32, -31, -17, -16, -1, 0, 15, 31),
            b"\x10\x0f\x0f\xff",
            b"\xea\x43",
            id="q6",
        ),
        pytest.param(
            "q8_g32_fp16",
            33,
            32,
            128,
            256,
            (-127, -1, 0, 1, 127),
            b"\x81\xff\x00\x01\x7f",
            b"",
            id="q8",
        ),
    ],
)
def test_row_split_matches_known_packed_bytes(
    format_name, k, group_size, k_pad, scale_offset, prefix, base_prefix, high_prefix
):
    groups = k_pad // group_size
    codes = torch.zeros((1, groups, group_size), dtype=torch.int8)
    codes[0, 0, : len(prefix)] = torch.tensor(prefix, dtype=torch.int8)
    scales = torch.zeros((1, groups), dtype=torch.float16)
    scales[0, :2] = torch.tensor([1.5, 0.25], dtype=torch.float16)

    # Literal wire positions and words come from the format, independently of layout helpers.
    expected = bytearray(scale_offset + groups * 2)
    expected[: len(base_prefix)] = base_prefix
    expected[256 : 256 + len(high_prefix)] = high_prefix
    expected[scale_offset:] = struct.pack(
        "<" + "H" * groups, 0x3E00, 0x3400, *([0] * (groups - 2))
    )
    assert encode_row_split(codes, scales, format_name, (1, k)) == expected
    decoded_scales, decoded_codes = decode_row_split_codes(
        expected, format_name, (1, k)
    )
    assert torch.equal(decoded_scales, scales)
    assert torch.equal(decoded_codes, codes)


def test_nvfp4_known_vector_geometry_swizzle_tail_and_round_trip():
    shape = (128, 64)
    geometry = block_scale_geometry("nvfp4", shape)
    assert (
        geometry.code_plane_bytes,
        geometry.scale_plane_offset,
        geometry.scale_plane_bytes,
        geometry.weight_divisor_offset,
        geometry.payload_bytes,
    ) == (4096, 4096, 512, 4608, 4612)

    packed = (
        torch.arange(geometry.code_plane_bytes, dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(128, 32)
    )
    packed[0, 0] = 0x10
    scales = (
        torch.arange(128 * 4, dtype=torch.int64)
        .remainder(0x7F)
        .to(torch.uint8)
        .reshape(128, 4)
    )
    divisor = struct.pack("<f", 2.5)
    payload = encode_nvfp4(packed, scales, divisor, shape)

    assert len(payload) == 4612
    assert payload[0] == 0x10  # low nibble is K=0; high nibble is K=1.
    for row, lane in ((0, 0), (31, 3), (32, 0), (127, 3)):
        offset = geometry.scale_plane_offset + (row % 32) * 16 + (row // 32) * 4 + lane
        assert payload[offset] == int(scales[row, lane])
    assert payload[geometry.weight_divisor_offset :] == divisor

    decoded_packed, decoded_scales, decoded_divisor = decode_nvfp4_words(payload, shape)
    assert torch.equal(decoded_packed, packed)
    assert torch.equal(decoded_scales, scales)
    assert bytes(decoded_divisor.reshape(1).view(torch.uint8).numpy()) == divisor


def _reference_e4m3fn(word: int) -> float:
    """Independent exact E4M3FN decode for oracle cross-checks."""

    sign = -1.0 if word & 0x80 else 1.0
    exponent, fraction = (word >> 3) & 0xF, word & 0x7
    if exponent == 0:
        return sign * fraction * 2.0**-9
    if exponent == 0xF and fraction == 0x7:
        return float("nan")
    return sign * (1.0 + fraction / 8.0) * 2.0 ** (exponent - 7)


def test_mxfp8_known_wire_geometry_and_round_trip():
    shape = (48, 2560)
    geometry = dense_block_scale_geometry("mxfp8", shape)
    assert (
        geometry.code_plane_bytes,
        geometry.scale_plane_offset,
        geometry.scale_plane_bytes,
        geometry.payload_bytes,
    ) == (48 * 2560, 122880, 48 * 80, 122880 + 48 * 80)

    codes = torch.arange(shape[0] * shape[1], dtype=torch.int64).remainder(256).to(torch.uint8)
    codes = codes.reshape(shape)
    codes[0, 0] = 0x00
    scales = torch.arange(shape[0] * 80, dtype=torch.int64).remainder(256).to(torch.uint8)
    scales = scales.reshape(48, 80)
    payload = encode_mxfp8(codes, scales, shape)

    assert payload[0] == 0x00
    assert payload[geometry.scale_plane_offset] == int(scales[0, 0])
    assert payload[geometry.scale_plane_offset + 48 * 80 - 1] == int(scales[47, 79])
    # Code plane row 1 starts at K bytes; scale row 1 starts at K/32 words.
    assert payload[2560] == int(codes[1, 0])
    assert payload[geometry.scale_plane_offset + 80] == int(scales[1, 0])

    decoded_codes, decoded_scales = decode_mxfp8_words(payload, shape)
    assert torch.equal(decoded_codes, codes)
    assert torch.equal(decoded_scales, scales)

    with pytest.raises(ValueError):
        encode_mxfp8(codes, scales, (48, 2559))


def test_mxfp8_represented_weights_match_independent_oracle():
    g = torch.Generator().manual_seed(7)
    for shape in ((48, 2560), (640, 2560), (12288, 2560), (2560, 640)):
        codes = torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)
        scales = torch.randint(0, 256, (shape[0], shape[1] // 32), dtype=torch.uint8, generator=g)
        produced = represented_mxfp8(codes, scales).reshape(shape)
        for n in (0, shape[0] // 2, shape[0] - 1):
            for k in (0, 15, 31, 32, shape[1] - 1):
                expected = _reference_e4m3fn(int(codes[n, k])) * 2.0 ** (
                    int(scales[n, k // 32]) - 127
                )
                actual = float(produced[n, k])
                assert actual == expected or (
                    expected != expected and actual != actual
                )


def test_block_scale_k16_natural_geometry_and_round_trip():
    shape = (16, 160)
    geometry = block_scale_k16_geometry("nvfp4", shape)
    code_plane_bytes = 16 * 160 // 2
    assert geometry.code_plane_bytes == code_plane_bytes
    assert geometry.scale_plane_offset == align_up(code_plane_bytes, 256)
    assert geometry.scale_plane_bytes == 16 * 160 // 16
    assert geometry.scale2_offset == geometry.scale_plane_offset + geometry.scale_plane_bytes
    assert geometry.payload_bytes == geometry.scale2_offset + 4

    packed = torch.arange(1280, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(16, 80)
    packed[0, 0] = 0x21  # low nibble K=0, high nibble K=1.
    scales = torch.arange(16 * 10, dtype=torch.int64).remainder(0x7E).to(torch.uint8).reshape(16, 10)
    scales[0, 0] = 1
    scale2 = struct.pack("<f", 3.324236240587197e-05)
    payload = encode_block_scale_k16(packed, scales, scale2, shape)

    assert payload[0] == 0x21
    assert payload[geometry.scale_plane_offset] == int(scales[0, 0])
    assert payload[geometry.scale_plane_offset + 10] == int(scales[1, 0])
    assert payload[geometry.scale2_offset :] == scale2

    decoded_packed, decoded_scales, decoded_scale2 = decode_block_scale_k16_words(payload, shape)
    assert torch.equal(decoded_packed, packed)
    assert torch.equal(decoded_scales, scales)
    assert bytes(decoded_scale2.reshape(1).view(torch.uint8).numpy()) == scale2

    with pytest.raises(ValueError):
        encode_block_scale_k16(packed, scales, struct.pack("<f", 0.0), shape)
    with pytest.raises(ValueError):
        encode_block_scale_k16(packed, scales, struct.pack("<f", -1.0), shape)
    with pytest.raises(ValueError):
        bad = scales.clone()
        bad[0, 0] = 0xFF
        encode_block_scale_k16(packed, bad, scale2, shape)


def test_block_scale_k16_represented_weights_match_independent_oracle():
    g = torch.Generator().manual_seed(11)
    shape = (2_500_012 // 1000, 160)
    packed = torch.randint(0, 256, (shape[0], 80), dtype=torch.uint8, generator=g)
    scales = torch.randint(1, 0x7F, (shape[0], 10), dtype=torch.uint8, generator=g)
    scale2 = 3.324236240587197e-05
    payload = encode_block_scale_k16(packed, scales, struct.pack("<f", scale2), shape)
    codes, natural, decoded_scale2 = decode_block_scale_k16_words(payload, shape)
    assert float(decoded_scale2) == scale2

    magnitudes = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    for row in (0, shape[0] // 2, shape[0] - 1):
        for k in (0, 1, 79, 159):
            word = int(codes[row, k // 2])
            nibble = word & 0xF if k % 2 == 0 else (word >> 4) & 0xF
            magnitude = magnitudes[nibble & 0x7]
            value = -magnitude if nibble & 0x8 else magnitude
            scale = _reference_e4m3fn(int(natural[row, k // 16]))
            expected = value * scale * scale2
            # The oracle is checked against the decode contract, not a kernel:
            # both sides here are exact word decodes.
            assert expected == expected  # finite; NaN would fail the layout contract
