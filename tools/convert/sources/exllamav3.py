"""The exllamav3 EXL3 checkpoint as a converter source (SP-7 contract section 6.1).

Reads and validates one (layer, expert, projection) EXL3 linear at a time:
its `trellis` int16 tensor, its `suh`/`svh` (or legacy packed `su`/`sv`), and its
optional `mcg`/`mul1` codebook marker. The checkpoint's `config.json` must carry
`quantization_config.quant_method = "exl3"` at the contract's pinned version.
Nothing here quantizes or decodes; the exact codec oracle lives in
`tools/convert/quantization/exl3.py` and is used by the bank method's verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch

from tools.artifact.layouts import EXL3_CODEBOOK_IDS, validate_exl3_rate
from .safetensors import SafetensorsSource, TensorInfo

PINNED_EXL3_VERSION = "1.6.1"
MCG_MARKER = 0xCBAC1FED
MUL1_MARKER = 0x83DCD12D


def marker_as_int32(marker: int) -> int:
    """A marker's u32 bit pattern as the signed int32 that stores it."""

    return marker - (1 << 32) if marker >= (1 << 31) else marker


# Two sampled tiles per bank side plus the inner pair bound the import's own checks.
_TILE = 16


@dataclass(frozen=True, slots=True)
class Exl3Linear:
    """One source EXL3 linear: the emitted bank's verbatim inputs."""

    trellis: torch.Tensor          # int16 [k/16, n/16, 16K]
    suh: torch.Tensor              # FP16 [k], expanded from legacy `su` when packed
    svh: torch.Tensor              # FP16 [n], expanded from legacy `sv` when packed
    k2: int                        # 2K, from the trellis shape
    codebook: str                  # 3inst | mcg | mul1, from the marker
    label: str

    @property
    def k(self) -> int:
        return self.trellis.shape[0] * _TILE

    @property
    def n(self) -> int:
        return self.trellis.shape[1] * _TILE


def _check_architecture(exl3_config: dict, base_config: dict) -> None:
    """Section 6.1 step 5: the EXL3 checkpoint must match the BF16 source's architecture."""

    def resolve(config: dict) -> dict:
        if "text_config" in config:
            return config["text_config"]
        return config

    source, reference = resolve(exl3_config), resolve(base_config)
    fields = (
        "hidden_size",
        "num_hidden_layers",
        "num_experts",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "num_experts_per_tok",
        "layer_types",
    )
    for field in fields:
        if field not in source or field not in reference:
            continue
        if source[field] != reference[field]:
            raise ValueError(
                f"EXL3 checkpoint {field}={source[field]!r} differs from the "
                f"BF16 source's {reference[field]!r}"
            )


def check_quantization_config(config: dict) -> None:
    """Section 6.1 step 1: quant_method `exl3` at the pinned version."""

    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise ValueError("EXL3 checkpoint has no quantization_config")
    if quantization.get("quant_method") != "exl3":
        raise ValueError(
            f"quant_method {quantization.get('quant_method')!r} is not exl3"
        )
    version = quantization.get("version")
    if version != PINNED_EXL3_VERSION:
        raise ValueError(
            f"EXL3 checkpoint version {version!r} is not the pinned "
            f"{PINNED_EXL3_VERSION!r}"
        )


def _marker_codebook(store: SafetensorsSource, name: str) -> str:
    """The marker names select the codebook; absent markers mean 3INST (section 2.1)."""

    for suffix, codebook, marker in (
        ("mcg", "mcg", MCG_MARKER),
        ("mul1", "mul1", MUL1_MARKER),
    ):
        full = f"{name}.{suffix}"
        if not store.has(full):
            continue
        info: TensorInfo = store.describe(full)
        if prod(info.shape) != 1 or info.dtype != "I32":
            raise ValueError(f"{full}: codebook marker must be one int32 scalar")
        value = store.read_flat(full).item()
        if value != marker_as_int32(marker):
            raise ValueError(
                f"{full}: marker {value & 0xFFFFFFFF:#x} must equal its multiplier {marker:#x}"
            )
        return codebook
    return "3inst"


def _packed_signs(packed: torch.Tensor, length: int) -> torch.Tensor:
    """Legacy `su`/`sv` int16 bitfields to ±1 FP16 (section 2.1): bit b of element w
    gives entry 16w+b; bit set is -1, clear +1."""

    words = packed.to(torch.int32) & 0xFFFF
    signs = torch.where(words == 0, 1.0, -1.0)
    expanded = signs.repeat_interleave(16)[:length]
    if expanded.numel() != length:
        raise ValueError("packed sign field is shorter than the vector it encodes")
    return expanded.to(torch.float16)


def read_exl3_linear(
    store: SafetensorsSource, name: str, *, expected: tuple[int, int] | None = None
) -> Exl3Linear:
    """Read and validate one `{projection}` EXL3 linear (its suffixed tensors)."""

    trellis_info = store.describe(f"{name}.trellis")
    if trellis_info.dtype != "I16" or trellis_info.shape == ():
        raise ValueError(f"{name}.trellis must be a nonempty int16 tensor")
    trellis = store.read_flat(f"{name}.trellis").reshape(trellis_info.shape)
    if len(trellis_info.shape) != 3:
        raise ValueError(f"{name}.trellis must be [k/16, n/16, 16K]")
    rows, columns, words = trellis_info.shape
    if words % 8 or not 2 <= words // 8 <= 16:
        raise ValueError(f"{name}.trellis width {words} is not 16K for K in 1..8")
    k2 = words // 8
    k, n = rows * _TILE, columns * _TILE
    if k % 128 or n % 128:
        raise ValueError(f"{name}: EXL3 dimensions {k}x{n} are not multiples of 128")

    codebook = _marker_codebook(store, name)
    validate_exl3_rate(k2, codebook)

    if store.has(f"{name}.suh"):
        suh = store.read_flat(f"{name}.suh")
        if suh.dtype != torch.float16 or suh.numel() != k:
            raise ValueError(f"{name}.suh must be FP16 [{k}]")
    else:
        packed = store.read_flat(f"{name}.su")
        if packed.dtype != torch.int16 or packed.numel() != (k + 15) // 16:
            raise ValueError(f"{name}.su must be the packed int16 sign field of [{k}]")
        suh = _packed_signs(packed, k)
    if store.has(f"{name}.svh"):
        svh = store.read_flat(f"{name}.svh")
        if svh.dtype != torch.float16 or svh.numel() != n:
            raise ValueError(f"{name}.svh must be FP16 [{n}]")
    else:
        packed = store.read_flat(f"{name}.sv")
        if packed.dtype != torch.int16 or packed.numel() != (n + 15) // 16:
            raise ValueError(f"{name}.sv must be the packed int16 sign field of [{n}]")
        svh = _packed_signs(packed, n)
    if bool(torch.isnan(suh.float()).any()) or bool(torch.isnan(svh.float()).any()):
        raise ValueError(f"{name}: suh/svh must be finite")

    if expected is not None and (k, n) != tuple(expected):
        raise ValueError(
            f"{name}: EXL3 linear is {k}x{n}, the bank requires {expected[0]}x{expected[1]}"
        )
    return Exl3Linear(
        trellis.contiguous(),
        suh.contiguous(),
        svh.contiguous(),
        k2,
        codebook,
        f"{store.path}:{name}",
    )


@dataclass(frozen=True, slots=True)
class Exl3ExpertSource:
    """One expert's EXL3 linear as a recipe-level logical source.

    The shape is the expert's semantic weight `[n, k]` (the dequantized matrix's
    shape), matching the BF16 binding's parameter shape so the existing grouping,
    binding-fragment and tiering row conventions hold unchanged. The bank method
    consumes the handles directly; this source provides no value reads.
    """

    shape: tuple[int, ...]
    label: str
    linear: Exl3Linear

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or self.shape != (self.linear.n, self.linear.k):
            raise ValueError("EXL3 expert source shape must be [n, k] of its linear")

    @property
    def codebook_id(self) -> int:
        return EXL3_CODEBOOK_IDS[self.linear.codebook]


def exl3_expert_source(
    store: SafetensorsSource,
    tensor_name: str,
    *,
    expected: tuple[int, int] | None = None,
) -> Exl3ExpertSource:
    """Read one expert's EXL3 linear and wrap it as a recipe-level source."""

    linear = read_exl3_linear(store, tensor_name, expected=expected)
    return Exl3ExpertSource((linear.n, linear.k), linear.label, linear)


def exl3_provenance(store: SafetensorsSource) -> dict:
    """Descriptive provenance for the conversion report (contract section 4)."""

    quantization = store.config.get("quantization_config", {})
    return {
        "exl3_version": quantization.get("version"),
        "checkpoint": str(store.path),
    }
