"""Method requests/results and the built-in adapters to quantization or encoded import.

Methods own input traversal, chunking, and auxiliary values. Numerical algorithms
return codes/scales; artifact.tensor_output owns their physical byte placement.
User methods accept the same PrepareRequest and return a PreparedMethod.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from math import prod
import struct
from typing import Callable, Mapping

import numpy as np
import torch

from tools.artifact.formats import (
    DirectFormat,
    QuantFormat,
    get_format,
    valid_positive_fp32_word,
)
from tools.artifact.schema import TensorSpec
from tools.artifact.tensor_output import TensorOutput

from .quantization.fp8_row import quantize_bf16_rows
from .quantization.groupwise import quantize_matrix
from .sources.exllamav3 import Exl3ExpertSource
from .sources.logical import EncodedRows, LogicalSource

from tools.artifact.layouts import (
    EXL3_CODEBOOK_IDS,
    exl3_bank_geometry,
    validate_exl3_rate,
)
from tools.convert.quantization.exl3 import decode_tiles

EXL3_TILE = 16

UseKey = tuple[str, str]
AuxiliaryKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class MethodInput:
    parameter: str
    source: LogicalSource
    uses: tuple[UseKey, ...]


@dataclass(frozen=True, slots=True)
class AuxiliaryValue:
    format: str
    shape: tuple[int, ...]
    data: bytes

    @classmethod
    def activation_divisor(cls, value: bytes | float) -> AuxiliaryValue:
        raw = value if isinstance(value, bytes) else struct.pack("<f", value)
        if len(raw) != 4 or not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
            raise ValueError("activation input divisor must be positive finite FP32")
        return cls("fp32", (), raw)


@dataclass(frozen=True, slots=True)
class PreparedMethod:
    produce: Callable[[TensorOutput], None]
    auxiliaries: Mapping[AuxiliaryKey, AuxiliaryValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PrepareRequest:
    target: TensorSpec
    inputs: tuple[MethodInput, ...]
    policies: Mapping[UseKey, str]
    parameters: Mapping[str, object]
    device: str = "cuda"
    rows_per_chunk: int = 512
    auxiliary_overrides: Mapping[AuxiliaryKey, AuxiliaryValue] = field(
        default_factory=dict
    )
    source_offsets: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        values = [0]
        for item in self.inputs:
            values.append(values[-1] + prod(item.source.shape))
        object.__setattr__(self, "source_offsets", tuple(values))

    @property
    def source(self) -> LogicalSource:
        if len(self.inputs) != 1:
            raise ValueError("this method has multiple inputs; use request.inputs")
        return self.inputs[0].source

    def require_input_shape(self, shape: tuple[int, ...]) -> None:
        if self.source.shape != tuple(shape):
            raise ValueError(f"source shape {self.source.shape} differs from {shape}")

    def job(self, *, produce, auxiliaries=None) -> PreparedMethod:
        return PreparedMethod(produce, {} if auxiliaries is None else dict(auxiliaries))

    def values(self, begin: int, end: int) -> torch.Tensor:
        """Read the ordered logical inputs as one C-order parent element sequence."""
        pieces = []
        if not 0 <= begin <= end <= self.source_offsets[-1]:
            raise ValueError(f"{self.target.id}: requested values exceed method inputs")
        index = bisect_right(self.source_offsets, begin) - 1
        while begin < end:
            high = min(end, self.source_offsets[index + 1])
            pieces.append(
                self.inputs[index].source.values(
                    begin - self.source_offsets[index],
                    high - self.source_offsets[index],
                )
            )
            begin = high
            index += 1
        if not pieces:
            return torch.empty(0)
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    def encoded_rows(self, begin: int, end: int) -> EncodedRows:
        pieces = []
        cursor = 0
        for item in self.inputs:
            source = item.source
            if len(source.shape) != 2 or source.shape[1] != self.target.shape[1]:
                raise ValueError(
                    f"{item.parameter}: encoded grouping requires complete rows"
                )
            low, high = max(begin, cursor), min(end, cursor + source.shape[0])
            if low < high:
                if source.read_encoded is None:
                    raise ValueError(f"{item.parameter}: encoded rows are unavailable")
                pieces.append(source.read_encoded(low - cursor, high - cursor))
            cursor += source.shape[0]
        if not pieces or end > cursor:
            raise ValueError("encoded method input range is invalid")
        first = pieces[0]
        if any(
            (p.format, p.weight_divisor) != (first.format, first.weight_divisor)
            for p in pieces
        ):
            raise ValueError("encoded inputs cannot share one parent format/divisor")
        if len(pieces) == 1:
            return first
        return EncodedRows(
            first.format,
            torch.cat([p.codes for p in pieces]),
            torch.cat([p.scales for p in pieces]),
            first.weight_divisor,
        )


Method = Callable[[PrepareRequest], PreparedMethod]
_DIRECT_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "int32": torch.int32}


def _preflight(request: PrepareRequest, *, values: bool = True) -> None:
    if request.parameters:
        raise ValueError(
            f"{request.target.id}: this method accepts no numerical parameters"
        )
    if prod(request.target.shape) != sum(
        prod(item.source.shape) for item in request.inputs
    ):
        raise ValueError(
            f"{request.target.id}: parent and logical source counts differ"
        )
    if type(request.rows_per_chunk) is not int or request.rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    if values:
        for item in request.inputs:
            item.source.values(0, 1)


def cast_direct(request: PrepareRequest) -> PreparedMethod:
    """Convert values at the explicit target BF16/FP32/INT32 boundary."""
    if not isinstance(get_format(request.target.format), DirectFormat):
        raise ValueError("cast_direct requires a direct target format")
    _preflight(request)
    dtype = _DIRECT_DTYPES[request.target.format]
    chunk = request.rows_per_chunk * (
        request.target.shape[-1] if len(request.target.shape) > 1 else 1
    )
    elements = prod(request.target.shape)

    def produce(output):
        for begin in range(0, elements, chunk):
            values = request.values(begin, min(elements, begin + chunk))
            if dtype == torch.int32 and (
                bool((values < -(1 << 31)).any())
                or bool((values > (1 << 31) - 1).any())
            ):
                raise ValueError(
                    "int32 conversion source is outside the representable range"
                )
            values = values.to(dtype=dtype)
            output.write_values(begin, values)

    return request.job(produce=produce)


def grouped_absmax(request: PrepareRequest) -> PreparedMethod:
    """Use the existing grouped max-abs, FP16-scale and code-rounding algorithm."""
    if (
        not isinstance(get_format(request.target.format), QuantFormat)
        or len(request.target.shape) != 2
    ):
        raise ValueError("grouped_absmax requires a grouped-integer matrix target")
    _preflight(request)
    n, k = request.target.shape

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = request.values(begin * k, end * k).reshape(end - begin, k)
            if not values.dtype.is_floating_point:
                raise TypeError(
                    "grouped_absmax source must provide floating-point values"
                )
            encoded = quantize_matrix(
                values, request.target.format, device=request.device
            )
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def fp8_row_maxabs(request: PrepareRequest) -> PreparedMethod:
    """Round inputs to BF16, then quantize to FP8 codes with BF16 row scales."""
    if request.target.format != "fp8_e4m3fn_row_bf16" or len(request.target.shape) != 2:
        raise ValueError("fp8_row_maxabs requires the row-scaled FP8 matrix format")
    _preflight(request)
    n, k = request.target.shape

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = (
                request.values(begin * k, end * k)
                .reshape(end - begin, k)
                .to(torch.bfloat16)
            )
            encoded = quantize_bf16_rows(values)
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def import_encoded(request: PrepareRequest) -> PreparedMethod:
    """Preserve the current FP8/NVFP4 source codes, scales and weight divisor."""
    if (
        request.target.format not in ("nvfp4", "fp8_e4m3fn_row_bf16")
        or len(request.target.shape) != 2
    ):
        raise ValueError("import_encoded requires a known encoded matrix target")
    _preflight(request, values=False)
    auxiliaries = {}
    weight_divisor = None
    for item in request.inputs:
        source = item.source
        if source.read_encoded is None:
            raise ValueError(f"{item.parameter}: encoded rows are unavailable")
        first = source.read_encoded(0, 1)
        if first.format != request.target.format:
            raise ValueError(
                f"{item.parameter}: source {first.format} differs from target {request.target.format}"
            )
        if first.format == "nvfp4":
            if weight_divisor is None:
                weight_divisor = first.weight_divisor
            elif first.weight_divisor != weight_divisor:
                raise ValueError(
                    "NVFP4 weight divisors differ; choose separate parents or a conversion method"
                )
            for parameter, input_name in item.uses:
                key = (parameter, input_name, "activation_input_divisor")
                if key in request.auxiliary_overrides:
                    auxiliaries[key] = request.auxiliary_overrides[key]
                elif request.policies[(parameter, input_name)] == "AllowA4":
                    if source.input_divisor is None:
                        raise ValueError(
                            f"{parameter}: supply an activation divisor for AllowA4"
                        )
                    auxiliaries[key] = AuxiliaryValue.activation_divisor(
                        source.input_divisor()
                    )
    n = request.target.shape[0]
    chunk = (
        max(128, request.rows_per_chunk // 128 * 128)
        if request.target.format == "nvfp4"
        else request.rows_per_chunk
    )

    def produce(output):
        for begin in range(0, n, chunk):
            words = request.encoded_rows(begin, min(n, begin + chunk))
            output.write_codes(begin, words.codes, words.scales, words.weight_divisor)

    return request.job(produce=produce, auxiliaries=auxiliaries)


def import_exl3_bank(request: PrepareRequest) -> PreparedMethod:
    """Repack one (layer, projection) EXL3 expert bank verbatim (contract 6.1 step 3).

    Every input is one expert's EXL3 linear (an `Exl3ExpertSource`); the parent is
    the bank `[E*n, k]`. Trellis bytes and suh/svh are copied unchanged, packed
    experts are expanded, and the emitted planes are verified against the source
    tensors with the exact codec oracle before the produced bytes are written.
    """

    if request.target.format != "exl3" or request.target.layout != "exl3_expert_bank":
        raise ValueError("import_exl3_bank requires the EXL3 expert bank target")
    if request.parameters:
        raise ValueError("import_exl3_bank accepts no method parameters")
    sources = []
    for item in request.inputs:
        if not isinstance(item.source, Exl3ExpertSource):
            raise ValueError(
                f"{item.parameter}: import_exl3_bank requires EXL3 expert sources"
            )
        sources.append(item.source)
    if not sources:
        raise ValueError("an EXL3 bank requires at least one expert")
    codebooks = {source.linear.codebook for source in sources}
    if len(codebooks) != 1:
        raise ValueError(
            f"a bank's codebook must be uniform; got {sorted(codebooks)}"
        )
    codebook = sources[0].linear.codebook
    n, k = sources[0].shape
    if any(source.shape != (n, k) for source in sources):
        raise ValueError("a bank's experts must share one [n, k] shape")
    if prod(request.target.shape) != len(sources) * n * k:
        raise ValueError("bank parent and logical expert shapes disagree")
    rates = tuple(source.linear.k2 for source in sources)
    for rate in rates:
        validate_exl3_rate(rate, codebook)
    geometry = exl3_bank_geometry(n, k, rates, codebook)
    if (geometry.experts * n, k) != tuple(request.target.shape):
        raise ValueError("bank parent shape is not the stacked expert rows [E*n, k]")

    # Directory-aware load validation (contract 4) checked at emission: offsets
    # increase in expert order, are 256-byte aligned, and match the extents.
    offsets = [entry[0] for entry in geometry.directory_entries()]
    if offsets != sorted(offsets) or any(
        offset % 256 or extent % 256
        for offset, extent in zip(offsets, geometry.trellis_extents)
    ):
        raise ValueError("trellis extents violate the load validation")

    def verify_expert(expert: int, extent: memoryview) -> None:
        """Contract 6.1 step 4: sampled tiles decode bit-identically from the emitted
        extent and the source trellis; every emitted suh/svh row equals its source."""

        source = sources[expert]
        tiles_n, tiles_k = k // EXL3_TILE, n // EXL3_TILE
        positions = [(0, 0), (tiles_n - 1, tiles_k - 1)]
        generator = torch.Generator().manual_seed(
            20261014 + 7919 * expert + 104729 * tiles_n * tiles_k
        )
        picks = torch.randint(0, tiles_n * tiles_k, (14,), generator=generator)
        positions += [(
            int(pick) // tiles_k, int(pick) % tiles_k
        ) for pick in picks.tolist()]
        words = rates[expert] * 8
        emitted = torch.frombuffer(
            bytearray(extent[: tiles_n * tiles_k * words * 2]), dtype=torch.int16
        ).reshape(tiles_n, tiles_k, words)
        emitted_tiles = decode_tiles(emitted, rates[expert], codebook)
        source_tiles = decode_tiles(source.linear.trellis, rates[expert], codebook)
        for tile_n, tile_k in positions:
            if not np.array_equal(
                emitted_tiles[tile_n, tile_k], source_tiles[tile_n, tile_k]
            ):
                raise ValueError(
                    f"expert {expert} tile ({tile_n},{tile_k}) differs from its source"
                )

    def produce(output):
        directory = bytearray(geometry.directory_bytes)
        for expert, (offset, extent, rate) in enumerate(geometry.directory_entries()):
            struct.pack_into("<QIBBBB", directory, expert * 16, offset, extent, rate, 0, 0, 0)
        output.write_bytes(geometry.directory_offset, bytes(directory))
        if geometry.trellis_offset > geometry.directory_bytes:
            output.write_bytes(
                geometry.directory_bytes,
                bytes(geometry.trellis_offset - geometry.directory_bytes),
            )
        suh = torch.empty((geometry.experts, k), dtype=torch.float16)
        svh = torch.empty((geometry.experts, n), dtype=torch.float16)
        for expert, source in enumerate(sources):
            offset, extent, _ = geometry.directory_entries()[expert]
            data = source.linear.trellis.numpy().tobytes()
            if len(data) != n * k * rates[expert] // 16:
                raise ValueError(f"expert {expert}: trellis bytes disagree with its rate")
            packed = data + bytes(extent - len(data))
            verify_expert(expert, memoryview(packed))
            output.write_bytes(offset, packed)
            suh[expert] = source.linear.suh
            svh[expert] = source.linear.svh
        if not torch.equal(
            suh, torch.stack([source.linear.suh for source in sources])
        ) or not torch.equal(
            svh, torch.stack([source.linear.svh for source in sources])
        ):
            raise ValueError("emitted suh/svh rows differ from their sources")
        output.write_bytes(geometry.suh_offset, suh.numpy().tobytes())
        suh_end = geometry.suh_offset + geometry.suh_bytes
        if geometry.svh_offset > suh_end:
            output.write_bytes(suh_end, bytes(geometry.svh_offset - suh_end))
        output.write_bytes(geometry.svh_offset, svh.numpy().tobytes())
        output.write_bytes(
            geometry.codebook_offset,
            EXL3_CODEBOOK_IDS[codebook].to_bytes(4, "little"),
        )

    return request.job(produce=produce)


METHODS: dict[str, Method] = {
    "cast_direct": cast_direct,
    "grouped_absmax": grouped_absmax,
    "fp8_row_maxabs": fp8_row_maxabs,
    "import_encoded": import_encoded,
    "import_exl3_bank": import_exl3_bank,
}
