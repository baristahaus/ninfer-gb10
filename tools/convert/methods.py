"""Method requests/results and the built-in adapters to quantization or encoded import.

Methods own input traversal, chunking, and auxiliary values. Numerical algorithms
return codes/scales; artifact.tensor_output owns their physical byte placement.
User methods accept the same PrepareRequest and return a PreparedMethod.
"""

from __future__ import annotations

from bisect import bisect_right
from concurrent.futures import Executor
from dataclasses import dataclass, field
from math import isfinite, prod
import struct
from typing import Callable, Mapping

import torch

from tools.artifact.formats import (
    DirectFormat,
    QuantFormat,
    get_format,
    valid_positive_fp32_word,
)
from tools.artifact.schema import TensorSpec
from tools.artifact.tensor_output import TensorOutput

import numpy as np

from tools.artifact.layouts import (
    EXL3_CODEBOOK_IDS,
    exl3_bank_geometry,
    exl3_dense_geometry,
    validate_exl3_rate,
)

from .quantization.exl3 import decode_tiles
from .quantization.fp8_row import quantize_bf16_rows
from .quantization.groupwise import quantize_matrix
from .quantization.stored_scale import quantize_stored_scale
from .sources.exllamav3 import Exl3ExpertSource
from .sources.logical import EncodedRows, LogicalSource

UseKey = tuple[str, str]
AuxiliaryKey = tuple[str, str, str]
CHUNK_BYTES = 16 * 1024 * 1024
SOURCE_CACHE_BYTES = 2 * 1024 * 1024 * 1024


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
    rows_per_chunk: int | None = None
    auxiliary_overrides: Mapping[AuxiliaryKey, AuxiliaryValue] = field(
        default_factory=dict
    )
    source_cache_bytes: int = SOURCE_CACHE_BYTES
    reader: Executor | None = None
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

    def job(self, *, produce, auxiliaries=None) -> PreparedMethod:
        return PreparedMethod(produce, {} if auxiliaries is None else dict(auxiliaries))

    def chunk_rows(self, row_bytes: int, *, alignment: int = 1) -> int:
        rows = self.rows_per_chunk or max(1, CHUNK_BYTES // row_bytes)
        return max(alignment, rows // alignment * alignment)

    def read_chunks(self, read: Callable, count: int, chunk: int):
        """Keep one ordered source read ahead of computation, bounded by one chunk."""
        pending = None
        for begin in range(0, count, chunk):
            end = min(count, begin + chunk)
            if self.reader is None:
                yield begin, end, read(begin, end)
                continue
            if pending is None:
                pending = self.reader.submit(read, begin, end)
            value = pending.result()
            pending = (
                self.reader.submit(read, end, min(count, end + chunk))
                if end < count
                else None
            )
            yield begin, end, value

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
        k = self.target.shape[-1]
        if not 0 <= begin < end <= self.source_offsets[-1] // k:
            raise ValueError("encoded method input range is invalid")
        index = bisect_right(self.source_offsets, begin * k) - 1
        while begin < end:
            item = self.inputs[index]
            source = item.source
            if len(source.shape) != 2 or source.shape[1] != k:
                raise ValueError(
                    f"{item.parameter}: encoded grouping requires complete rows"
                )
            cursor = self.source_offsets[index] // k
            high = min(end, cursor + source.shape[0])
            if source.read_encoded is None:
                raise ValueError(f"{item.parameter}: encoded rows are unavailable")
            pieces.append(source.read_encoded(begin - cursor, high - cursor))
            if pieces[-1].format != self.target.format:
                raise ValueError(
                    f"{item.parameter}: encoded rows differ from target format"
                )
            begin = high
            index += 1
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
    if request.rows_per_chunk is not None and (
        type(request.rows_per_chunk) is not int or request.rows_per_chunk <= 0
    ):
        raise ValueError("rows_per_chunk must be positive")
    if type(request.source_cache_bytes) is not int or request.source_cache_bytes < 0:
        raise ValueError("source_cache_bytes must be nonnegative")
    if values:
        for item in request.inputs:
            item.source.values(0, 1)


def cast_direct(request: PrepareRequest) -> PreparedMethod:
    """Convert values at the explicit target BF16/FP32/INT32 boundary."""
    if not isinstance(get_format(request.target.format), DirectFormat):
        raise ValueError("cast_direct requires a direct target format")
    _preflight(request)
    dtype = _DIRECT_DTYPES[request.target.format]
    k = request.target.shape[-1] if len(request.target.shape) > 1 else 1
    chunk = request.chunk_rows(k * dtype.itemsize) * k
    elements = prod(request.target.shape)

    def produce(output):
        for begin, _, values in request.read_chunks(request.values, elements, chunk):
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
    chunk = request.chunk_rows(k * 4)

    def produce(output):
        for begin, end, values in request.read_chunks(request.values, n * k, chunk * k):
            values = values.reshape((end - begin) // k, k)
            if not values.dtype.is_floating_point:
                raise TypeError(
                    "grouped_absmax source must provide floating-point values"
                )
            encoded = quantize_matrix(
                values, request.target.format, device=request.device
            )
            output.write_codes(begin // k, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def fp8_row_maxabs(request: PrepareRequest) -> PreparedMethod:
    """Round inputs to BF16, then quantize to FP8 codes with BF16 row scales."""
    if request.target.format != "fp8_e4m3fn_row_bf16" or len(request.target.shape) != 2:
        raise ValueError("fp8_row_maxabs requires the row-scaled FP8 matrix format")
    _preflight(request)
    n, k = request.target.shape
    chunk = request.chunk_rows(k * 4)

    def produce(output):
        for begin, end, values in request.read_chunks(request.values, n * k, chunk * k):
            values = (
                values.reshape((end - begin) // k, k)
                .to(device=request.device)
                .to(torch.bfloat16)
            )
            encoded = quantize_bf16_rows(values)
            output.write_codes(begin // k, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def import_encoded(request: PrepareRequest) -> PreparedMethod:
    """Preserve FP8/NVFP4 words, with independent divisors for bank matrices."""
    block_fp8 = request.target.format == "fp8_e4m3fn_block128x128_bf16"
    if (
        request.target.format
        not in ("nvfp4", "fp8_e4m3fn_row_bf16", "fp8_e4m3fn_block128x128_bf16")
        or len(request.target.shape) not in (2, 3)
        or (
            len(request.target.shape) == 3
            and request.target.format == "fp8_e4m3fn_row_bf16"
        )
    ):
        raise ValueError("import_encoded requires an encoded matrix or bank target")
    _preflight(request, values=False)
    n, k = request.target.shape[-2:]
    matrices = request.target.shape[0] if len(request.target.shape) == 3 else 1
    auxiliaries = {}
    divisors = {}
    cursor = 0
    for item in request.inputs:
        source = item.source
        if len(source.shape) != 2 or source.shape[1] != k:
            raise ValueError(
                f"{item.parameter}: encoded grouping requires complete rows"
            )
        if block_fp8 and source.shape[0] % 128:
            raise ValueError(
                f"{item.parameter}: FP8 source must contain whole 128-row blocks"
            )
        if source.read_encoded is None:
            raise ValueError(f"{item.parameter}: encoded rows are unavailable")
        encoding = source.encoding()
        if encoding != request.target.format:
            raise ValueError(
                f"{item.parameter}: source {encoding} differs from target {request.target.format}"
            )
        if encoding == "nvfp4":
            if source.weight_divisor is None:
                raise ValueError(
                    f"{item.parameter}: NVFP4 source needs a weight divisor"
                )
            divisor = source.weight_divisor()
            for matrix in range(cursor // n, (cursor + source.shape[0] - 1) // n + 1):
                if matrix in divisors and divisors[matrix] != divisor:
                    raise ValueError("NVFP4 weight divisors differ within one matrix")
                divisors[matrix] = divisor
            for parameter, input_name in item.uses:
                key = (parameter, input_name, "activation_input_divisor")
                if key in request.auxiliary_overrides:
                    value = request.auxiliary_overrides[key]
                elif request.policies[(parameter, input_name)] == "AllowA4":
                    if source.input_divisor is None:
                        raise ValueError(
                            f"{parameter}: supply an activation divisor for AllowA4"
                        )
                    value = AuxiliaryValue.activation_divisor(source.input_divisor())
                else:
                    continue
                if key in auxiliaries and auxiliaries[key] != value:
                    raise ValueError(
                        f"{parameter}: activation divisors differ within one Use"
                    )
                auxiliaries[key] = value
        cursor += source.shape[0]
    chunk = request.chunk_rows(
        k, alignment=128 if request.target.format == "nvfp4" or block_fp8 else 1
    )

    def produce(output):
        for matrix in range(matrices):
            read = lambda begin, end: request.encoded_rows(
                matrix * n + begin, matrix * n + end
            )
            for begin, _, words in request.read_chunks(read, n, chunk):
                output.write_codes(
                    begin,
                    words.codes,
                    words.scales,
                    words.weight_divisor,
                    matrix=matrix,
                )

    return request.job(produce=produce, auxiliaries=auxiliaries)


def stored_scale_mse(request: PrepareRequest) -> PreparedMethod:
    """Two-pass object amax and exhaustive G16 stored-scale MSE encoding."""
    if (
        request.target.format not in ("nf4_g16_e4m3fn", "nvfp4")
        or len(request.target.shape) != 2
    ):
        raise ValueError("stored_scale_mse requires an NF4 or NVFP4 matrix")
    _preflight(request)
    n, k = request.target.shape
    chunk = request.chunk_rows(
        k * 4, alignment=128 if request.target.format == "nvfp4" else 1
    )

    def produce(output):
        maximum = torch.zeros((), dtype=torch.float32, device=request.device)
        cached = [] if n * k * 4 <= request.source_cache_bytes else None
        for begin, end, values in request.read_chunks(request.values, n * k, chunk * k):
            values = values.to(request.device).float()
            maximum = torch.maximum(maximum, values.abs().amax())
            if cached is not None:
                cached.append((begin, end, values))
        maximum = maximum.item()
        if not isfinite(maximum):
            raise ValueError("stored-scale source contains NaN or infinity")
        limit = 448.0 if request.target.format == "nf4_g16_e4m3fn" else 2688.0
        divisor = struct.pack("<f", limit / maximum if maximum else 1.0)
        if not valid_positive_fp32_word(struct.unpack("<I", divisor)[0]):
            raise ValueError("amax produces an invalid FP32 object divisor")
        blocks = (
            cached
            if cached is not None
            else request.read_chunks(request.values, n * k, chunk * k)
        )
        for begin, end, values in blocks:
            values = values.reshape((end - begin) // k, k)
            encoded = quantize_stored_scale(
                values, request.target.format, divisor, device=request.device
            )
            output.write_codes(begin // k, encoded.codes, encoded.scales, divisor)

    return request.job(produce=produce)


_LAYER_CACHE: dict[str, dict] = {}

EXL3_TILE = 16


def pack_exl3_bank_quantized(request: PrepareRequest) -> PreparedMethod:
    """Emit one (layer, projection) EXL3 expert bank from the bank quantize campaign.

    Every input is one expert's logical source in expert-id order (all sharing one
    [n, k]); the method parameters carry `{"layer_file": path, "projection": name,
    "k2": rate}` pointing at the `convert_model_banks` layer file. The bank planes
    re-verify sampled tiles against the campaign trellis before writing (the import
    route's exact-oracle discipline); deep verification against the BF16 source is
    the real-check tool's job.
    """

    if request.target.format != "exl3" or request.target.layout != "exl3_expert_bank":
        raise ValueError("pack_exl3_bank_quantized requires the EXL3 expert bank target")
    layer_file = request.parameters.get("layer_file")
    projection = request.parameters.get("projection")
    k2 = request.parameters.get("k2")
    if not isinstance(layer_file, str) or not isinstance(projection, str) or k2 is None:
        raise ValueError(
            "pack_exl3_bank_quantized requires layer_file, projection and k2 parameters")
    k2 = int(k2)
    if not request.inputs:
        raise ValueError("an EXL3 bank requires at least one expert")
    shapes = [item.source.shape for item in request.inputs]
    if any(len(shape) != 2 or shape != shapes[0] for shape in shapes):
        raise ValueError("a bank's experts must share one [n, k] logical shape")
    n, k = shapes[0]
    experts = len(request.inputs)
    if (geometry_experts := prod(request.target.shape) // (n * k)) != experts:
        raise ValueError("bank parent and logical expert shapes disagree")
    codebook = "mcg"
    validate_exl3_rate(k2, codebook)
    geometry = exl3_bank_geometry(n, k, [k2] * experts, codebook)
    if (geometry.experts * n, k) != tuple(request.target.shape):
        raise ValueError("bank parent shape is not the stacked expert rows [E*n, k]")

    results = _layer_file(torch.load, layer_file)[projection]
    if not isinstance(results, list) or len(results) != experts:
        raise ValueError(f"{layer_file}:{projection}: expert count disagrees with the group")
    trellis_planes = []
    suh = torch.empty((experts, k), dtype=torch.float16)
    svh = torch.empty((experts, n), dtype=torch.float16)
    for expert, result in enumerate(results):
        stored_n, stored_k = result["weight_shape"]
        if (stored_n, stored_k) != (n, k):
            raise ValueError(
                f"{layer_file}:{projection}: expert {expert} shape "
                f"{(stored_n, stored_k)} differs from the logical {(n, k)}")
        trellis = result["trellis"].detach().cpu().contiguous()
        if trellis.dtype != torch.int16 or trellis.shape != (k // 16, n // 16, 8 * k2):
            raise ValueError(
                f"{layer_file}:{projection}: expert {expert} trellis shape disagrees "
                f"with the rate")
        trellis_planes.append(trellis)
        suh[expert] = result["suh"].detach().cpu().to(torch.float16)
        svh[expert] = result["svh"].detach().cpu().to(torch.float16)

    def verify_expert(expert: int, extent: memoryview) -> None:
        tiles_n, tiles_k = k // EXL3_TILE, n // EXL3_TILE
        positions = [(0, 0), (tiles_n - 1, tiles_k - 1)]
        generator = torch.Generator().manual_seed(
            20261014 + 7919 * expert + 104729 * tiles_n * tiles_k)
        picks = torch.randint(0, tiles_n * tiles_k, (14,), generator=generator)
        positions += [(int(pick) // tiles_k, int(pick) % tiles_k) for pick in picks.tolist()]
        words = k2 * 8
        emitted = torch.frombuffer(
            bytearray(extent[: tiles_n * tiles_k * words * 2]), dtype=torch.int16
        ).reshape(tiles_n, tiles_k, words)
        emitted_tiles = decode_tiles(emitted, k2, codebook)
        source_tiles = decode_tiles(trellis_planes[expert], k2, codebook)
        for tile_n, tile_k in positions:
            if not np.array_equal(emitted_tiles[tile_n, tile_k], source_tiles[tile_n, tile_k]):
                raise ValueError(
                    f"expert {expert} tile ({tile_n},{tile_k}) differs from its source")

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
        for expert, (offset, extent, _rate) in enumerate(geometry.directory_entries()):
            data = trellis_planes[expert].numpy().tobytes()
            if len(data) != n * k * k2 // 16:
                raise ValueError(f"expert {expert}: trellis bytes disagree with the rate")
            packed = data + bytes(extent - len(data))
            verify_expert(expert, memoryview(packed))
            output.write_bytes(offset, packed)
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


pack_exl3_bank_quantized.quantized_bank = True


def _exl3_bank_spec_bytes(items, parameters) -> int:
    """The bank payload is data-dependent: resolve it from the group's shape contract.

    `items` are the recipe's (parameter, source) pairs; a MethodInput also fits through
    its `source` attribute.
    """

    def _shape(item):
        # The recipe passes (parameter, Selection) pairs; a MethodInput fits directly.
        value = item[1] if isinstance(item, tuple) else item
        return value.source.shape

    shapes = [_shape(item) for item in items]
    if any(len(shape) != 2 or shape != shapes[0] for shape in shapes):
        raise ValueError("a bank's experts must share one [n, k] logical shape")
    n, k = shapes[0]
    codebook = "mcg"
    k2 = int(parameters["k2"])
    validate_exl3_rate(k2, codebook)
    return exl3_bank_geometry(n, k, [k2] * len(items), codebook).payload_bytes


pack_exl3_bank_quantized.spec_bytes = _exl3_bank_spec_bytes


def _layer_file(loader, path: str) -> dict:
    """Layer files are immutable campaign outputs; one assembly reads each many times."""

    cached = _LAYER_CACHE.get(path)
    if cached is None:
        cached = _LAYER_CACHE[path] = loader(path, map_location="cpu", weights_only=False)
    return cached




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



def _exl3_import_spec_bytes(items, parameters) -> int:
    """Resolve the verbatim bank payload from the experts' own rates."""

    def _source(item):
        value = item[1] if isinstance(item, tuple) else item
        return value.source

    sources = [_source(item) for item in items]
    n, k = sources[0].shape
    rates = tuple(source.linear.k2 for source in sources)
    return exl3_bank_geometry(n, k, rates, sources[0].linear.codebook).payload_bytes


import_exl3_bank.spec_bytes = _exl3_import_spec_bytes


def pack_exl3_dense(request: PrepareRequest) -> PreparedMethod:
    """Emit one dense linear as an `exl3_dense_v1` object from a convert_model result.

    The method parameters carry `{"layer_file": path, "tensor": name}` pointing at the
    quantizer's layer file (the SP-7 quantize route's output); the request's logical source
    (the BF16 checkpoint tensor) validates the [n, k] orientation only. The packed payload is
    byte-identical to `pack_dense`'s, and the emitted planes are re-checked against the layer
    file's own metrics (shape, bitrate, codebook, suh/svh extents) before writing. Deep
    per-tensor verification against the composed weight is the real-check tool's job
    (`exl3_real_check.py` + the conformance binary's argv mode), not the campaign writer's.
    """

    if request.target.format != "exl3" or request.target.layout != "exl3_dense_v1":
        raise ValueError("pack_exl3_dense requires the exl3_dense_v1 target")
    layer_file = request.parameters.get("layer_file")
    tensor = request.parameters.get("tensor")
    if not isinstance(layer_file, str) or not isinstance(tensor, str):
        raise ValueError("pack_exl3_dense requires layer_file and tensor parameters")
    if len(request.inputs) != 1:
        raise ValueError("pack_exl3_dense requires exactly one logical source")
    n, k = request.target.shape

    result = _layer_file(torch.load, layer_file)[tensor]
    stored_n, stored_k = result["weight_shape"]
    if (stored_n, stored_k) != (n, k):
        raise ValueError(
            f"{layer_file}:{tensor}: converted shape {(stored_n, stored_k)} differs from "
            f"the logical parameter {(n, k)}"
        )
    k2 = int(result["k2"])
    codebook = "mcg"
    geometry = exl3_dense_geometry(n, k, k2, codebook)
    if geometry.payload_bytes < 256 + 4:
        raise ValueError("pack_exl3_dense: degenerate payload")

    trellis = result["trellis"].detach().cpu().contiguous()
    if trellis.dtype != torch.int16 or trellis.ndim != 3:
        raise ValueError(f"{layer_file}:{tensor}: trellis must be int16 [k/16, n/16, 8*k2]")
    if trellis.shape != (k // 16, n // 16, 8 * k2):
        raise ValueError(
            f"{layer_file}:{tensor}: trellis shape {tuple(trellis.shape)} disagrees with "
            f"[{k // 16}, {n // 16}, {8 * k2}]"
        )
    suh = result["suh"].detach().cpu().to(torch.float16).contiguous()
    svh = result["svh"].detach().cpu().to(torch.float16).contiguous()
    if tuple(suh.shape) != (k,) or tuple(svh.shape) != (n,):
        raise ValueError(f"{layer_file}:{tensor}: suh/svh planes disagree with the trellis")

    def produce(output):
        trellis_bytes = trellis.numpy().tobytes()
        if len(trellis_bytes) != geometry.trellis_bytes:
            raise ValueError(f"{layer_file}:{tensor}: trellis bytes disagree with the bitrate")
        output.write_bytes(0, trellis_bytes)
        padded = geometry.suh_offset - len(trellis_bytes)
        if padded:
            output.write_bytes(len(trellis_bytes), bytes(padded))
        output.write_bytes(geometry.suh_offset, suh.numpy().tobytes())
        output.write_bytes(geometry.svh_offset, svh.numpy().tobytes())
        output.write_bytes(
            geometry.codebook_offset, EXL3_CODEBOOK_IDS[codebook].to_bytes(4, "little")
        )

    return request.job(produce=produce)


def _exl3_dense_spec_bytes(items, parameters) -> int:
    """Resolve the dense payload from the layer file's own stored shape and rate."""

    result = _layer_file(torch.load, parameters["layer_file"])[parameters["tensor"]]
    n, k = result["weight_shape"]
    return exl3_dense_geometry(n, k, int(result["k2"]), "mcg").payload_bytes


pack_exl3_dense.spec_bytes = _exl3_dense_spec_bytes


def pack_exl3_dense_group(request: PrepareRequest) -> PreparedMethod:
    """Emit one fused dense EXL3 object from a group of logical parameters (SP-7).

    The fused quantize campaign stores each mixer/FFN group as ONE exl3 linear in the
    binding row order (attention: query, key, gate, value; GDN: query, key, value, z;
    MLP: gate, up); the model's packing group binds each parameter to its row range of
    this object. The method parameters carry `{"layer_file": path, "tensor": name}` of
    the fused group tensor; the request's inputs are the group's logical sources in
    order, and their stacked shape must match the stored [n, k]. Deep per-tensor
    verification against the composed weight stays with the real-check tool.
    """

    if request.target.format != "exl3" or request.target.layout != "exl3_dense_v1":
        raise ValueError("pack_exl3_dense_group requires the exl3_dense_v1 target")
    layer_file = request.parameters.get("layer_file")
    tensor = request.parameters.get("tensor")
    if not isinstance(layer_file, str) or not isinstance(tensor, str):
        raise ValueError("pack_exl3_dense_group requires layer_file and tensor parameters")
    if not request.inputs:
        raise ValueError("pack_exl3_dense_group requires the group's logical sources")
    n = sum(prod(item.source.shape[:1]) for item in request.inputs)
    k = prod(request.inputs[0].source.shape[1:2]) if request.inputs[0].source.shape else 0
    for item in request.inputs:
        shape = item.source.shape
        if len(shape) != 2 or shape[1] != request.inputs[0].source.shape[1]:
            raise ValueError("pack_exl3_dense_group: group members must share one [., k] shape")

    result = _layer_file(torch.load, layer_file)[tensor]
    stored_n, stored_k = result["weight_shape"]
    if (stored_n, stored_k) != (n, k):
        raise ValueError(
            f"{layer_file}:{tensor}: converted shape {(stored_n, stored_k)} differs from "
            f"the fused group {(n, k)}"
        )
    k2 = int(result["k2"])
    codebook = "mcg"
    geometry = exl3_dense_geometry(n, k, k2, codebook)

    trellis = result["trellis"].detach().cpu().contiguous()
    if trellis.dtype != torch.int16 or trellis.shape != (k // 16, n // 16, 8 * k2):
        raise ValueError(f"{layer_file}:{tensor}: trellis shape disagrees with the fused group")
    suh = result["suh"].detach().cpu().to(torch.float16).contiguous()
    svh = result["svh"].detach().cpu().to(torch.float16).contiguous()
    if tuple(suh.shape) != (k,) or tuple(svh.shape) != (n,):
        raise ValueError(f"{layer_file}:{tensor}: suh/svh planes disagree with the fused group")

    def produce(output):
        trellis_bytes = trellis.numpy().tobytes()
        if len(trellis_bytes) != geometry.trellis_bytes:
            raise ValueError(f"{layer_file}:{tensor}: trellis bytes disagree with the bitrate")
        output.write_bytes(0, trellis_bytes)
        padded = geometry.suh_offset - len(trellis_bytes)
        if padded:
            output.write_bytes(len(trellis_bytes), bytes(padded))
        output.write_bytes(geometry.suh_offset, suh.numpy().tobytes())
        output.write_bytes(geometry.svh_offset, svh.numpy().tobytes())
        output.write_bytes(
            geometry.codebook_offset, EXL3_CODEBOOK_IDS[codebook].to_bytes(4, "little")
        )

    return request.job(produce=produce)


def _exl3_dense_group_spec_bytes(items, parameters) -> int:
    """Resolve the fused group payload from the layer file's own stored shape."""

    def _rows(item):
        value = item[1] if isinstance(item, tuple) else item
        return prod(value.source.shape[:1])

    n = sum(_rows(item) for item in items)
    value = items[0][1] if isinstance(items[0], tuple) else items[0]
    k = value.source.shape[1]
    result = _layer_file(torch.load, parameters["layer_file"])[parameters["tensor"]]
    stored_n, stored_k = result["weight_shape"]
    if (stored_n, stored_k) != (n, k):
        raise ValueError(
            f"{parameters['layer_file']}:{parameters['tensor']}: converted shape "
            f"{(stored_n, stored_k)} differs from the group {(n, k)}"
        )
    return exl3_dense_geometry(n, k, int(result["k2"]), "mcg").payload_bytes


pack_exl3_dense_group.spec_bytes = _exl3_dense_group_spec_bytes


METHODS: dict[str, Method] = {
    "cast_direct": cast_direct,
    "grouped_absmax": grouped_absmax,
    "fp8_row_maxabs": fp8_row_maxabs,
    "import_encoded": import_encoded,
    "import_exl3_bank": import_exl3_bank,
    "pack_exl3_dense": pack_exl3_dense,
    "pack_exl3_dense_group": pack_exl3_dense_group,
    "pack_exl3_bank_quantized": pack_exl3_bank_quantized,
    "stored_scale_mse": stored_scale_mse,
}
