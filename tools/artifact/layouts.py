"""Persistent tensor layouts shared by NInfer converters and reference models.

This module maps already-selected numeric words to their registered byte layout.
It deliberately does not quantize floating-point source weights or assign model
roles to tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
import operator
from types import MappingProxyType
from typing import Sequence


from .formats import (
    DirectFormat,
    Exl3Format,
    Fp8RowFormat,
    Nvfp4Format,
    NumericFormat,
    QuantFormat,
    get_format,
)

PLANE_ALIGNMENT = 256
K_ALIGNMENT = 128


@dataclass(frozen=True, slots=True)
class Layout:
    name: str
    alignment: int
    formats: frozenset[str]


@dataclass(frozen=True, slots=True)
class RowSplitGeometry:
    n: int
    k: int
    k_pad: int
    groups_per_row: int
    base_bytes_per_group: int
    high_bytes_per_group: int
    base_row_bytes: int
    high_row_bytes: int
    scale_row_bytes: int
    base_offset: int
    base_bytes: int
    high_offset: int
    high_bytes: int
    scale_offset: int
    scale_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class BlockScaleGeometry:
    n: int
    k: int
    groups_per_row: int
    k_tiles: int
    code_plane_bytes: int
    scale_plane_offset: int
    scale_plane_bytes: int
    weight_divisor_offset: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class RowScaleGeometry:
    n: int
    k: int
    code_plane_bytes: int
    scale_plane_offset: int
    scale_plane_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class Exl3BankGeometry:
    """Plane geometry of one exl3_expert_bank (SP-7 contract section 4).

    The trellis plane packs each expert's extent at its own 256-byte-aligned size,
    so mixed bitrates give unequal extents; the experts plane is the per-expert
    directory that resolves them. The codebook is the bank's object scalar.
    """

    experts: int
    n: int
    k: int
    k2: tuple[int, ...]
    codebook: str
    directory_offset: int
    directory_bytes: int
    trellis_offset: int
    trellis_extents: tuple[int, ...]
    trellis_bytes: int
    suh_offset: int
    suh_bytes: int
    svh_offset: int
    svh_bytes: int
    codebook_offset: int
    payload_bytes: int

    def directory_entries(self) -> tuple[tuple[int, int, int], ...]:
        """Per expert: (trellis byte offset, extent bytes, 2K), in expert id order."""

        entries = []
        offset = self.trellis_offset
        for expert, extent in enumerate(self.trellis_extents):
            entries.append((offset, extent, self.k2[expert]))
            offset += extent
        return tuple(entries)


@dataclass(frozen=True, slots=True)
class Exl3DenseGeometry:
    """Plane geometry of one exl3_dense_v1 (SP-7 contract section 4, dense linear).

    One linear, one bitrate: the trellis plane sits at offset 0 with no
    directory, and the codebook id closes the object.
    """

    n: int
    k: int
    k2: int
    codebook: str
    trellis_offset: int
    trellis_bytes: int
    suh_offset: int
    suh_bytes: int
    svh_offset: int
    svh_bytes: int
    codebook_offset: int
    payload_bytes: int


CONTIGUOUS_LE_V1 = Layout("contiguous_le_v1", 256, frozenset(("bf16", "fp32", "int32", "fp8_e4m3fn")))
ROW_SPLIT_K128_V1 = Layout(
    "row_split_k128_v1",
    256,
    frozenset(("q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16")),
)
BLOCK_SCALE_K16_M128X4_V1 = Layout(
    "block_scale_k16_m128x4_v1",
    256,
    frozenset(("nvfp4",)),
)
ROW_SCALE_V1 = Layout(
    "row_scale_v1",
    256,
    frozenset(("fp8_e4m3fn_row_bf16",)),
)

EXPERT_BLOCK_SCALE_K16_M128X4_V1 = Layout(
    "expert_block_scale_k16_m128x4_v1", 256, frozenset(("nvfp4",)))

EXL3_EXPERT_BANK_V1 = Layout("exl3_expert_bank", 256, frozenset(("exl3",)))

EXL3_DENSE_V1 = Layout("exl3_dense_v1", 256, frozenset(("exl3",)))

LAYOUTS = MappingProxyType(
    {
        layout.name: layout
        for layout in (
            CONTIGUOUS_LE_V1,
            ROW_SPLIT_K128_V1,
            BLOCK_SCALE_K16_M128X4_V1,
            ROW_SCALE_V1,
            EXPERT_BLOCK_SCALE_K16_M128X4_V1,
            EXL3_EXPERT_BANK_V1,
            EXL3_DENSE_V1,
        )
    }
)


def align_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("align_up requires a nonnegative value and positive alignment")
    return (value + alignment - 1) // alignment * alignment


def get_layout(name: str) -> Layout:
    try:
        return LAYOUTS[name]
    except KeyError:
        raise ValueError(f"unknown tensor layout: {name!r}") from None


def _format(value: str | NumericFormat) -> NumericFormat:
    if isinstance(value, str):
        return get_format(value)
    registered = get_format(value.name)
    if value != registered:
        raise ValueError(f"numeric format does not match registered {value.name!r}")
    return registered


def _layout(value: str | Layout) -> Layout:
    if isinstance(value, str):
        return get_layout(value)
    registered = get_layout(value.name)
    if value != registered:
        raise ValueError(f"layout does not match registered {value.name!r}")
    return registered


def _shape(value: Sequence[int], *, rank: int | None = None) -> tuple[int, ...]:
    dims = []
    for dim in value:
        if isinstance(dim, bool):
            raise ValueError("shape dimensions must be positive integers")
        try:
            item = operator.index(dim)
        except TypeError:
            raise ValueError("shape dimensions must be positive integers") from None
        if item <= 0:
            raise ValueError("shape dimensions must be positive integers")
        dims.append(item)
    result = tuple(dims)
    if rank is not None and len(result) != rank:
        raise ValueError(f"layout requires rank {rank}, got rank {len(result)}")
    return result


def row_split_geometry(
    format: str | QuantFormat, shape: Sequence[int]
) -> RowSplitGeometry:
    spec = _format(format)
    if not isinstance(spec, QuantFormat):
        raise ValueError("row_split_k128_v1 requires a grouped quantized format")
    n, k = _shape(shape, rank=2)
    k_pad = align_up(k, K_ALIGNMENT)
    groups_per_row = k_pad // spec.group_size
    base_bytes_per_group = spec.group_size if spec.bits == 8 else spec.group_size // 2
    high_bytes_per_group = (
        0 if spec.bits in (4, 8) else spec.group_size * (spec.bits - 4) // 8
    )
    base_row_bytes = groups_per_row * base_bytes_per_group
    high_row_bytes = groups_per_row * high_bytes_per_group
    scale_row_bytes = groups_per_row * 2
    base_bytes = n * base_row_bytes
    high_bytes = n * high_row_bytes
    scale_bytes = n * scale_row_bytes
    high_offset = align_up(base_bytes, PLANE_ALIGNMENT)
    scale_offset = high_offset + align_up(high_bytes, PLANE_ALIGNMENT)
    return RowSplitGeometry(
        n=n,
        k=k,
        k_pad=k_pad,
        groups_per_row=groups_per_row,
        base_bytes_per_group=base_bytes_per_group,
        high_bytes_per_group=high_bytes_per_group,
        base_row_bytes=base_row_bytes,
        high_row_bytes=high_row_bytes,
        scale_row_bytes=scale_row_bytes,
        base_offset=0,
        base_bytes=base_bytes,
        high_offset=high_offset,
        high_bytes=high_bytes,
        scale_offset=scale_offset,
        scale_bytes=scale_bytes,
        payload_bytes=scale_offset + scale_bytes,
    )


def block_scale_geometry(
    format: str | Nvfp4Format, shape: Sequence[int]
) -> BlockScaleGeometry:
    spec = _format(format)
    if not isinstance(spec, Nvfp4Format):
        raise ValueError("block_scale_k16_m128x4_v1 requires NVFP4")
    n, k = _shape(shape, rank=2)
    if n % 128 != 0 or k % 64 != 0:
        raise ValueError(
            "block_scale_k16_m128x4_v1 requires N divisible by 128 "
            "and K divisible by 64"
        )
    code_plane_bytes = n * k // 2
    scale_plane_offset = align_up(code_plane_bytes, PLANE_ALIGNMENT)
    scale_plane_bytes = n * k // spec.group_size
    weight_divisor_offset = scale_plane_offset + scale_plane_bytes
    return BlockScaleGeometry(
        n=n,
        k=k,
        groups_per_row=k // spec.group_size,
        k_tiles=k // 64,
        code_plane_bytes=code_plane_bytes,
        scale_plane_offset=scale_plane_offset,
        scale_plane_bytes=scale_plane_bytes,
        weight_divisor_offset=weight_divisor_offset,
        payload_bytes=weight_divisor_offset + 4,
    )


def row_scale_geometry(
    format: str | Fp8RowFormat, shape: Sequence[int]
) -> RowScaleGeometry:
    spec = _format(format)
    if not isinstance(spec, Fp8RowFormat):
        raise ValueError("row_scale_v1 requires a row-scaled FP8 format")
    n, k = _shape(shape, rank=2)
    code_plane_bytes = n * k
    scale_plane_offset = align_up(code_plane_bytes, PLANE_ALIGNMENT)
    scale_plane_bytes = n * 2
    return RowScaleGeometry(
        n=n,
        k=k,
        code_plane_bytes=code_plane_bytes,
        scale_plane_offset=scale_plane_offset,
        scale_plane_bytes=scale_plane_bytes,
        payload_bytes=scale_plane_offset + scale_plane_bytes,
    )


EXL3_CODEBOOK_IDS = MappingProxyType({"3inst": 0, "mcg": 1, "mul1": 2})
EXL3_MIN_K2 = 2
EXL3_MAX_K2 = 16
EXL3_HALF_INTEGER_K2 = frozenset((3, 5, 7))


def validate_exl3_rate(k2: int, codebook: str) -> None:
    """Section 2.4/2.5 bitrate admission: 2K in 2..16, half-integer rates MUL1 only."""

    if type(k2) is not int:
        raise ValueError("EXL3 bitrate 2K must be an integer")
    if codebook not in EXL3_CODEBOOK_IDS:
        raise ValueError(f"unknown EXL3 codebook {codebook!r}")
    if k2 % 2 == 0:
        if not EXL3_MIN_K2 <= k2 <= EXL3_MAX_K2:
            raise ValueError(f"EXL3 integer bitrate 2K must be 2..16, got {k2}")
        return
    if k2 not in EXL3_HALF_INTEGER_K2:
        raise ValueError(f"EXL3 half-integer bitrate 2K must be 3, 5 or 7, got {k2}")
    if codebook != "mul1":
        raise ValueError(f"half-integer EXL3 bitrates require the mul1 codebook, got {k2}")


def exl3_bank_geometry(
    n: int, k: int, k2: Sequence[int], codebook: str
) -> Exl3BankGeometry:
    """Plane geometry of one EXL3 expert bank from its per-expert bitrates.

    Expert e's trellis extent is k*n*K_e/8 bytes, zero-padded to a multiple of 256;
    extents pack in expert id order (contract section 4). k and n are multiples of
    128; the suh/svh planes are fixed-stride FP16 [E, k] / [E, n].
    """

    if type(n) is not int or type(k) is not int:
        raise ValueError("EXL3 bank dimensions must be integers")
    if n <= 0 or k <= 0 or n % 128 or k % 128:
        raise ValueError("EXL3 bank requires N and K divisible by 128")
    rates = tuple(k2)
    if not rates:
        raise ValueError("EXL3 bank requires at least one expert")
    for rate in rates:
        validate_exl3_rate(rate, codebook)
    experts = len(rates)
    directory_bytes = experts * 16
    trellis_offset = align_up(directory_bytes, PLANE_ALIGNMENT)
    extents = tuple(align_up(n * k * rate // 16, PLANE_ALIGNMENT) for rate in rates)
    trellis_bytes = sum(extents)
    suh_offset = trellis_offset + trellis_bytes
    suh_bytes = experts * k * 2
    svh_offset = align_up(suh_offset + suh_bytes, PLANE_ALIGNMENT)
    svh_bytes = experts * n * 2
    codebook_offset = svh_offset + svh_bytes
    return Exl3BankGeometry(
        experts=experts,
        n=n,
        k=k,
        k2=rates,
        codebook=codebook,
        directory_offset=0,
        directory_bytes=directory_bytes,
        trellis_offset=trellis_offset,
        trellis_extents=extents,
        trellis_bytes=trellis_bytes,
        suh_offset=suh_offset,
        suh_bytes=suh_bytes,
        svh_offset=svh_offset,
        svh_bytes=svh_bytes,
        codebook_offset=codebook_offset,
        payload_bytes=codebook_offset + 4,
    )


def exl3_dense_geometry(n: int, k: int, k2: int, codebook: str) -> "Exl3DenseGeometry":
    """Plane geometry of one exl3_dense_v1 from its single bitrate.

    The trellis plane sits at offset 0 (no directory plane); its extent is
    k*n*K/8 bytes, zero-padded to a multiple of 256; the suh/svh planes are
    fixed-stride FP16 [k] / [n]; the final four bytes are the codebook id.
    """

    if type(n) is not int or type(k) is not int or type(k2) is not int:
        raise ValueError("EXL3 dense dimensions and bitrate must be integers")
    if n <= 0 or k <= 0 or n % 128 or k % 128:
        raise ValueError("EXL3 dense linear requires N and K divisible by 128")
    validate_exl3_rate(k2, codebook)
    trellis_bytes = align_up(n * k * k2 // 16, PLANE_ALIGNMENT)
    suh_offset = trellis_bytes
    suh_bytes = k * 2
    svh_offset = suh_offset + suh_bytes
    svh_bytes = n * 2
    codebook_offset = svh_offset + svh_bytes
    return Exl3DenseGeometry(
        n=n,
        k=k,
        k2=k2,
        codebook=codebook,
        trellis_offset=0,
        trellis_bytes=trellis_bytes,
        suh_offset=suh_offset,
        suh_bytes=suh_bytes,
        svh_offset=svh_offset,
        svh_bytes=svh_bytes,
        codebook_offset=codebook_offset,
        payload_bytes=codebook_offset + 4,
    )


def encoded_size(
    layout: str | Layout,
    format: str | NumericFormat,
    shape: Sequence[int],
    *,
    payload_bytes: int | None = None,
) -> int:
    layout_spec = _layout(layout)
    numeric_spec = _format(format)
    if numeric_spec.name not in layout_spec.formats:
        raise ValueError(
            f"layout {layout_spec.name!r} does not accept format {numeric_spec.name!r}"
        )
    if layout_spec is EXL3_EXPERT_BANK_V1 or layout_spec is EXL3_DENSE_V1:
        if not isinstance(numeric_spec, Exl3Format):
            raise ValueError(f"{layout_spec.name} requires the EXL3 format")
        # Mixed per-expert bitrates (bank) and the dense linear's single bitrate make the
        # payload size data-dependent: it is the sum of the trellis extents, resolved by the
        # converter's plane geometry and validated against the encoded object at load
        # (contract section 4).
        if type(payload_bytes) is not int or payload_bytes < 256 + 4:
            raise ValueError(
                f"{layout_spec.name} size is data-dependent; supply the object's exact "
                "payload bytes"
            )
        _shape(shape, rank=2)
        return payload_bytes
    if layout_spec is EXPERT_BLOCK_SCALE_K16_M128X4_V1:
        e, n, k = _shape(shape, rank=3)
        if n % 128 or k % 64:
            raise ValueError("expert bank requires N%128=0 and K%64=0")
        return align_up(e*n*k//2, 256) + e*n*k//16 + e*4
    if layout_spec is CONTIGUOUS_LE_V1:
        if not isinstance(numeric_spec, DirectFormat):
            raise ValueError("contiguous_le_v1 requires a direct format")
        dims = _shape(shape)
        if len(dims) > 16:
            raise ValueError("contiguous_le_v1 supports rank 0 through 16")
        return prod(dims) * numeric_spec.word_bytes
    if layout_spec is ROW_SPLIT_K128_V1:
        if not isinstance(numeric_spec, QuantFormat):
            raise ValueError("row_split_k128_v1 requires a grouped quantized format")
        return row_split_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is BLOCK_SCALE_K16_M128X4_V1:
        if not isinstance(numeric_spec, Nvfp4Format):
            raise ValueError("block_scale_k16_m128x4_v1 requires NVFP4")
        return block_scale_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is ROW_SCALE_V1:
        if not isinstance(numeric_spec, Fp8RowFormat):
            raise ValueError("row_scale_v1 requires a row-scaled FP8 format")
        return row_scale_geometry(numeric_spec, shape).payload_bytes
    raise ValueError(f"unsupported tensor layout: {layout_spec.name!r}")
