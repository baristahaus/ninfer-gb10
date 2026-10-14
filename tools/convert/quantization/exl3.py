"""Exact EXL3 trellis codec oracle for converter-side verification (SP-7).

This module executes `docs/maintainer/exl3-trellis-contract.md` sections 2.3 to 2.5 and is the
contract's Python exact codec oracle (section 3.1): it maps stored `trellis` int16 elements, the
bitrate and the codebook to the FP16 inner weight `W_inner`, bit for bit. It is written from the
contract alone and shares no code with production kernels or with `exl3_reference.py`, which
executes the same sections independently and literally.

Bitrates travel as `k2 = 2K`: 2..16 for the integer rates 1..8 and 3, 5, 7 for the half-integer
rates 1.5, 2.5 and 3.5 (MUL1 only). FP16 values travel as their uint16 bit patterns, because the
oracle's criterion is bit equality.

Arithmetic exactness: the 3INST/MCG sum of two binary16 values and the MUL1 product-plus-bias are
exact in binary64 (their operands span at most 22 significant bits), so each is rounded exactly
once, to binary16 nearest-even, by NumPy's correctly rounded float64-to-float16 conversion.
"""

from __future__ import annotations

import numpy as np

TILE = 16
TILE_WEIGHTS = TILE * TILE
STATE_BITS = 16

CODEBOOKS = ("3inst", "mcg", "mul1")
MCG_MULTIPLIER = 0xCBAC1FED
MUL1_MULTIPLIER = 0x83DCD12D
_3INST_MULTIPLIER = 89226354
_3INST_ADDEND = 64248484
_LOP3_AND = 0x8FFF8FFF
_LOP3_XOR = 0x3B603B60
_MUL1_BIAS_BASE = 0x6400  # binary16 1024.0
_MUL1_SCALE_BITS = 0x1EEE
_MUL1_OFFSET_BITS = 0xC931
HALF_STEP_MASK = 0xAAAA
HALF_INTEGER_K2 = (3, 5, 7)


def validate_rate(k2: int, codebook: str) -> None:
    """Rejects a bitrate/codebook pair the contract does not define (section 2.4, 2.5)."""

    if codebook not in CODEBOOKS:
        raise ValueError(f"unknown EXL3 codebook {codebook!r}")
    if k2 % 2 == 0:
        if not 2 <= k2 <= 16:
            raise ValueError(f"integer EXL3 bitrate must be 1..8, got {k2 / 2}")
    elif k2 not in HALF_INTEGER_K2:
        raise ValueError(f"half-integer EXL3 bitrate must be 1.5, 2.5 or 3.5, got {k2 / 2}")
    elif codebook != "mul1":
        raise ValueError("half-integer EXL3 bitrates require the mul1 codebook")


def words_per_tile(k2: int) -> int:
    """int16 elements per 256-weight tile: 16 K."""

    return 8 * k2


def k2_from_words(words: int) -> int:
    """The bitrate implied by a trellis tensor's last dimension (K = words / 16)."""

    if words <= 0 or words % 8:
        raise ValueError(f"trellis tile width {words} is not 16 K for a half-integer K")
    return words // 8


def step_widths(k2: int) -> np.ndarray:
    """D(i), the new bits of each of the 256 sequence positions (section 2.4)."""

    positions = np.arange(TILE_WEIGHTS)
    if k2 % 2 == 0:
        return np.full(TILE_WEIGHTS, k2 // 2, dtype=np.int64)
    extra = (HALF_STEP_MASK >> (positions % 16)) & 1
    return (k2 // 2 + extra).astype(np.int64)


def _ring_bits(tiles: np.ndarray) -> np.ndarray:
    """Ring bits of each tile: uint8 [..., R], ring bit p in column p."""

    elements = tiles.astype(np.uint32)
    words = elements[..., 0::2] | (elements[..., 1::2] << 16)  # little-endian uint32 words
    big_endian = words.astype(">u4")
    as_bytes = big_endian.view(np.uint8).reshape(*words.shape[:-1], words.shape[-1] * 4)
    return np.unpackbits(as_bytes, axis=-1)  # most significant bit of each word first


def tile_states(tiles: np.ndarray, k2: int) -> np.ndarray:
    """The 16-bit state of every sequence position: uint16 [..., 256] (section 2.4)."""

    tiles = np.asarray(tiles)
    if tiles.shape[-1] != words_per_tile(k2):
        raise ValueError(f"tile holds {tiles.shape[-1]} elements; K={k2 / 2} needs {8 * k2}")
    bits = _ring_bits(tiles.view(np.uint16) if tiles.dtype == np.int16 else tiles)
    ring = bits.shape[-1]
    if ring != 128 * k2:
        raise AssertionError("ring length disagrees with the bitrate")
    ends = np.cumsum(step_widths(k2))
    offsets = (ends[:, None] - STATE_BITS + np.arange(STATE_BITS)[None, :]) % ring  # [256, 16]
    window = bits[..., offsets]  # [..., 256, 16], first bit most significant
    weights = (1 << np.arange(STATE_BITS - 1, -1, -1)).astype(np.uint32)
    return (window.astype(np.uint32) * weights).sum(axis=-1).astype(np.uint16)


def _binary16(bits: np.ndarray) -> np.ndarray:
    return bits.astype(np.uint16).view(np.float16).astype(np.float64)


def _round_binary16(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float16).view(np.uint16)


def codebook_bits(states: np.ndarray, codebook: str) -> np.ndarray:
    """binary16 bit patterns of the codebook values of 16-bit states (section 2.5)."""

    s = np.asarray(states).astype(np.uint64)
    mask = np.uint64(0xFFFFFFFF)
    if codebook in ("3inst", "mcg"):
        if codebook == "3inst":
            x = (s * np.uint64(_3INST_MULTIPLIER) + np.uint64(_3INST_ADDEND)) & mask
        else:
            x = (s * np.uint64(MCG_MULTIPLIER)) & mask
        x = (x & np.uint64(_LOP3_AND)) ^ np.uint64(_LOP3_XOR)
        low = _binary16(x & np.uint64(0xFFFF))
        high = _binary16(x >> np.uint64(16))
        return _round_binary16(low + high)
    if codebook == "mul1":
        x = (s * np.uint64(MUL1_MULTIPLIER)) & mask
        byte_sum = sum((x >> np.uint64(shift)) & np.uint64(0xFF) for shift in (0, 8, 16, 24))
        h = _binary16(np.uint64(_MUL1_BIAS_BASE) + byte_sum)
        scale = _binary16(np.array(_MUL1_SCALE_BITS))
        offset = _binary16(np.array(_MUL1_OFFSET_BITS))
        return _round_binary16(h * scale + offset)
    raise ValueError(f"unknown EXL3 codebook {codebook!r}")


def tile_element_order() -> tuple[np.ndarray, np.ndarray]:
    """(r, c) tile coordinates of sequence positions 0..255 (section 2.3)."""

    i = np.arange(TILE_WEIGHTS)
    t, j = i >> 3, i & 7
    rows = 2 * (t & 3) + (j & 1) + 8 * ((j >> 1) & 1)
    cols = (t >> 2) + 8 * ((j >> 2) & 1)
    return rows, cols


def decode_tiles(tiles: np.ndarray, k2: int, codebook: str) -> np.ndarray:
    """Inner-weight bits of tiles [..., 16 K] as uint16 [..., 16, 16] (row-major r, c)."""

    validate_rate(k2, codebook)
    values = codebook_bits(tile_states(tiles, k2), codebook)
    rows, cols = tile_element_order()
    out = np.zeros((*values.shape[:-1], TILE, TILE), dtype=np.uint16)
    out[..., rows, cols] = values
    return out


def decode_inner(trellis: np.ndarray, codebook: str) -> np.ndarray:
    """`W_inner` bits [k, n] of one EXL3 linear from its trellis [k/16, n/16, 16 K]."""

    trellis = np.asarray(trellis)
    if trellis.ndim != 3:
        raise ValueError("EXL3 trellis must have rank three")
    k2 = k2_from_words(trellis.shape[-1])
    tiles = decode_tiles(trellis, k2, codebook)  # [k/16, n/16, 16, 16]
    a, b = trellis.shape[:2]
    return tiles.transpose(0, 2, 1, 3).reshape(a * TILE, b * TILE)


def expand_packed_signs(packed: np.ndarray, length: int) -> np.ndarray:
    """binary16 bits of legacy `su`/`sv` bitfields: bit b of element w is entry 16 w + b,
    set meaning -1 and clear +1 (section 2.1)."""

    words = np.asarray(packed).view(np.uint16).astype(np.uint32)
    bits = (words[:, None] >> np.arange(16)[None, :]) & 1
    flat = bits.reshape(-1)
    if flat.size < length:
        raise ValueError("packed sign field is shorter than the vector it encodes")
    return np.where(flat[:length] == 1, np.uint16(0xBC00), np.uint16(0x3C00)).astype(np.uint16)
