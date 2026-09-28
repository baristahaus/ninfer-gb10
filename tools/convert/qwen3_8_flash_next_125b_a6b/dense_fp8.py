"""Re-encode the Qwen3.8 Flash-Next 125B-A6B dense projections as row-scaled FP8.

The input is the BF16-projection artifact written by `convert` (or its v3 upgrade); its BF16 words
are the checkpoint's words, so no source checkpoint is needed. The projections whose Ops accept
`fp8_e4m3fn_row_bf16` are quantized from those exact words with `fp8_row_maxabs` rounding: one BF16
row multiplier (the round-to-nearest-even BF16 value of the row's max |w| / 448) and E4M3FN codes
rounded to nearest even. They are the GDN query/key/value, output-gate and output projections, the
QSA packed query/gate and output projections, the text HyperConnection down and up projections,
the text shared-expert projections and the output head. Each keeps its object id, except the
shared expert's gate and up projections: their rows are stacked into one [1280,2560] parent
(`...shared_expert.gate_up_proj.weight`, gate rows first) that the fused SwiGLU consumes whole,
and their bindings become its two row ranges. Rows are quantized independently, so stacking does
not change any code or multiplier. Every Use, component and other binding is unchanged. Every
other object, including the MTP layer, the routed experts and the PLE table, is copied byte for
byte.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence

import numpy as np
import torch

from tools.artifact.codecs.fp8_row import encode_fp8_row_scaled
from tools.artifact.reader import Artifact
from tools.artifact.schema import (
    ResourceObject,
    ResourceSpec,
    TensorObject,
    TensorSpec,
    binding_parts,
)
from tools.artifact.writer import ArtifactWriter
from tools.convert.quantization.fp8_row import RowScaledFp8Words, quantize_bf16_rows

from . import inventory

FP8_ROW = "fp8_e4m3fn_row_bf16"
ROW_SCALE = "row_scale_v1"
OUTPUT_BASENAME = "qwen3_8_flash_next_125b_a6b_nvfp4_fp8_projections.ninfer"
RECIPE_ID = "qwen3_8_flash_next_125b_a6b_nvfp4_fp8_projections-v3"
# Row chunks of about this many elements are quantized concurrently; the NumPy rounding releases
# the GIL, so the pool scales with the host's cores.
QUANTIZE_CHUNK_ELEMENTS = 1 << 21
_E4M3FN_VALUES = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).double().numpy()


class PackedParent(NamedTuple):
    """One FP8 parent whose rows stack whole BF16 logical matrices in order."""

    parameters: tuple[str, ...]
    shape: tuple[int, int]


def _hyper_connection(prefix: str) -> dict[str, tuple[int, int]]:
    return {
        prefix + "input_mix_weight_down.weight": (320, 10240),
        prefix + "input_mix_weight_up.weight": (10240, 320),
    }


def dense_fp8_parameters() -> dict[str, tuple[int, int]]:
    """Logical text projections re-encoded in place by this recipe, with their exact shapes."""

    text = "model.language_model."
    parameters: dict[str, tuple[int, int]] = {}
    for layer in inventory.LAYERS:
        prefix = f"{text}layers.{layer}."
        if layer in inventory.FULL_ATTENTION_LAYERS:
            parameters[prefix + "self_attn.q_proj.weight"] = (12288, 2560)
            parameters[prefix + "self_attn.o_proj.weight"] = (2560, 6144)
        else:
            parameters[prefix + "linear_attn.in_proj_qkv.weight"] = (10240, 2560)
            parameters[prefix + "linear_attn.in_proj_z.weight"] = (6144, 2560)
            parameters[prefix + "linear_attn.out_proj.weight"] = (2560, 6144)
        for connection in ("attn_hyper_connection.", "mlp_hyper_connection."):
            parameters.update(_hyper_connection(prefix + connection))
        parameters[prefix + "mlp.shared_expert.down_proj.weight"] = (2560, 640)
    parameters.update(_hyper_connection(text + "hyper_connection_mixer."))
    parameters["lm_head.weight"] = (248320, 2560)
    return parameters


def dense_fp8_packed_parents() -> dict[str, PackedParent]:
    """New FP8 parents by object id: each text layer's shared-expert gate rows, then up rows."""

    packed: dict[str, PackedParent] = {}
    for layer in inventory.LAYERS:
        prefix = f"model.language_model.layers.{layer}.mlp.shared_expert."
        packed[prefix + "gate_up_proj.weight"] = PackedParent(
            (prefix + "gate_proj.weight", prefix + "up_proj.weight"), (1280, 2560)
        )
    return packed


def _binding_objects(binding: object, objects: Mapping[str, object], label: str) -> set[str]:
    return {object_id for object_id, _, _ in binding_parts(binding, objects, label)}


def _whole_bf16_object(artifact: Artifact, name: str, shape: tuple[int, int]) -> str:
    """The id of the one whole contiguous BF16 object that parameter `name` binds."""

    binding = artifact.directory.bindings.get(name)
    if binding is None:
        raise ValueError(f"missing logical parameter {name}")
    parts = binding_parts(binding, artifact.by_id, name)
    object_id, begin, end = parts[0]
    obj = artifact.object(object_id)
    if (
        len(parts) != 1
        or not isinstance(obj, TensorObject)
        or obj.format != "bf16"
        or obj.layout != "contiguous_le_v1"
        or tuple(obj.shape) != shape
        or (begin, end) != (0, shape[0] * shape[1])
    ):
        raise ValueError(f"{name}: expected one whole contiguous BF16 {list(shape)} object")
    return object_id


def _select_objects(
    artifact: Artifact,
    parameters: Mapping[str, tuple[int, int]],
    packed: Mapping[str, PackedParent],
) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[str, ...]]]:
    """Resolve the re-encoded parameters to whole BF16 objects that no other consumer references.

    Returns the objects re-encoded in place with their shapes, and each packed parent's source
    objects in row order.
    """

    directory = artifact.directory
    selected: dict[str, tuple[int, int]] = {}
    sources: dict[str, tuple[str, ...]] = {}
    claimed: set[str] = set()

    def claim(name: str, object_id: str) -> None:
        if object_id in claimed:
            raise ValueError(f"{name}: object {object_id} is shared by two selected parameters")
        claimed.add(object_id)

    for name, shape in parameters.items():
        object_id = _whole_bf16_object(artifact, name, shape)
        claim(name, object_id)
        selected[object_id] = shape
    for packed_id, parent in packed.items():
        if packed_id in artifact.by_id:
            raise ValueError(f"packed parent {packed_id} collides with an existing object")
        rows = parent.shape[0] // len(parent.parameters)
        if rows * len(parent.parameters) != parent.shape[0]:
            raise ValueError(f"packed parent {packed_id} does not split into equal row blocks")
        ids = []
        for name in parent.parameters:
            object_id = _whole_bf16_object(artifact, name, (rows, parent.shape[1]))
            claim(name, object_id)
            ids.append(object_id)
        sources[packed_id] = tuple(ids)
    names = set(parameters) | {name for parent in packed.values() for name in parent.parameters}
    for name, binding in directory.bindings.items():
        if name not in names and _binding_objects(binding, artifact.by_id, name) & claimed:
            raise ValueError(f"{name} shares a re-encoded object with a BF16 consumer")
    for use in directory.uses:
        for role, binding in use.get("auxiliaries", {}).items():
            label = f"{use['parameter']}@{use['input']}/{role}"
            if _binding_objects(binding, artifact.by_id, label) & claimed:
                raise ValueError(f"{label} references a re-encoded object")
    return selected, sources


def _float64(bf16: torch.Tensor) -> np.ndarray:
    """Exact binary64 values of contiguous BF16 words."""

    words = bf16.view(torch.int16).numpy().view(np.uint16).astype(np.uint32) << np.uint32(16)
    return words.view(np.float32).astype(np.float64)


def _quantize_rows(values: torch.Tensor) -> tuple[RowScaledFp8Words, float, float, float]:
    """Quantize BF16 rows; return the words and the represented-weight error sums."""

    words = quantize_bf16_rows(values)
    exact = _float64(values)
    error = _E4M3FN_VALUES[words.codes.numpy()] * _float64(words.scales)[:, None] - exact
    return (
        words,
        float(np.square(error).sum()),
        float(np.square(exact).sum()),
        float(np.abs(error).max()),
    )


def _fp8_payload(
    artifact: Artifact, object_ids: Sequence[str], shape: tuple[int, int], pool: Executor
) -> tuple[bytes, dict]:
    """Quantize the row-stacked BF16 objects in concurrent row chunks; return payload and error."""

    n, k = shape
    raw = bytearray(b"".join(artifact.read_object(object_id) for object_id in object_ids))
    values = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(n, k)
    rows = max(1, QUANTIZE_CHUNK_ELEMENTS // k)
    chunks = list(pool.map(_quantize_rows, (values[r : r + rows] for r in range(0, n, rows))))
    codes = torch.cat([words.codes for words, _, _, _ in chunks])
    scales = torch.cat([words.scales for words, _, _, _ in chunks])
    squared_error = sum(chunk[1] for chunk in chunks)
    squared_value = sum(chunk[2] for chunk in chunks)
    relative = math.sqrt(squared_error / squared_value) if squared_value > 0.0 else 0.0
    return encode_fp8_row_scaled(codes, scales, shape), {
        "shape": list(shape),
        "relative_rms_error": relative,
        "max_abs_error": max(chunk[3] for chunk in chunks),
    }


def reencode(
    source_path: str | Path,
    out_path: str | Path,
    parameters: Mapping[str, tuple[int, int]],
    *,
    recipe_id: str,
    packed: Mapping[str, PackedParent] | None = None,
) -> dict:
    """Write a copy of `source_path` whose selected BF16 matrices are row-scaled FP8.

    `parameters` are re-encoded in place; each `packed` parent replaces its source objects, at
    the position of the first, and its parameters bind consecutive row ranges of it.
    """

    started = time.perf_counter()
    output = Path(out_path)
    packed = dict(packed or {})
    with Artifact(source_path) as artifact:
        directory = artifact.directory
        selected, sources = _select_objects(artifact, parameters, packed)
        first_source = {ids[0]: packed_id for packed_id, ids in sources.items()}
        later_sources = {object_id for ids in sources.values() for object_id in ids[1:]}
        # One step per written object: (output id, source ids, FP8 shape or None to copy).
        steps: list[tuple[str, tuple[str, ...], tuple[int, int] | None]] = []
        specs: list[TensorSpec | ResourceSpec] = []
        for obj in directory.objects:
            if obj.id in later_sources:
                continue
            if isinstance(obj, ResourceObject):
                specs.append(ResourceSpec(obj.id, obj.bytes, obj.encoding))
                steps.append((obj.id, (obj.id,), None))
            elif obj.id in first_source:
                packed_id = first_source[obj.id]
                shape = packed[packed_id].shape
                specs.append(TensorSpec(packed_id, shape, FP8_ROW, ROW_SCALE))
                steps.append((packed_id, sources[packed_id], shape))
            elif obj.id in selected:
                specs.append(TensorSpec(obj.id, tuple(obj.shape), FP8_ROW, ROW_SCALE))
                steps.append((obj.id, (obj.id,), selected[obj.id]))
            else:
                specs.append(TensorSpec(obj.id, tuple(obj.shape), obj.format, obj.layout))
                steps.append((obj.id, (obj.id,), None))
        bindings = dict(directory.bindings)
        for packed_id, parent in packed.items():
            elements = parent.shape[0] * parent.shape[1] // len(parent.parameters)
            for index, name in enumerate(parent.parameters):
                bindings[name] = {
                    "parts": [
                        {"object": packed_id, "range": [index * elements, (index + 1) * elements]}
                    ]
                }
        provenance = {
            "source": directory.provenance.get("source"),
            "recipe": recipe_id,
            "derived_from_recipe": directory.provenance.get("recipe"),
            "method": "fp8_row_maxabs",
        }
        tensors: dict[str, dict] = {}
        output.parent.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool, ArtifactWriter(
            output,
            specs,
            components=directory.components,
            bindings=bindings,
            uses=directory.uses,
            metadata=directory.metadata,
            provenance=provenance,
        ) as writer:
            for index, (object_id, source_ids, shape) in enumerate(steps, start=1):
                if shape is not None:
                    payload, tensors[object_id] = _fp8_payload(artifact, source_ids, shape, pool)
                    writer.write_object(object_id, payload)
                else:
                    writer.write_object(object_id, artifact.iter_object(object_id))
                print(f"[{index}/{len(steps)}] {object_id}", flush=True)
        source_id = artifact.artifact_id.hex()
        source_payload = artifact.payload_bytes
    with Artifact(output) as written:
        output_payload = written.payload_bytes
        output_file_bytes = written.file_bytes
    worst = max(tensors.items(), key=lambda item: item[1]["relative_rms_error"], default=None)
    report = {
        "recipe_id": recipe_id,
        "source": {"path": str(Path(source_path).resolve()), "artifact_id": source_id},
        "output": str(output.resolve()),
        "reencoded_tensors": len(tensors),
        "payload_bytes": {"source": source_payload, "output": output_payload},
        "file_bytes": output_file_bytes,
        "worst_relative_rms_error": None
        if worst is None
        else {"object": worst[0], **worst[1]},
        "tensors": tensors,
        "elapsed_seconds": time.perf_counter() - started,
    }
    Path(str(output) + ".conversion.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="BF16-projection artifact")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.name != OUTPUT_BASENAME:
        raise SystemExit(f"output basename must be {OUTPUT_BASENAME!r}")
    report = reencode(
        args.source,
        args.out,
        dense_fp8_parameters(),
        recipe_id=RECIPE_ID,
        packed=dense_fp8_packed_parents(),
    )
    print(
        f"complete: {report['reencoded_tensors']} tensors, "
        f"{report['payload_bytes']['source']} -> {report['payload_bytes']['output']} payload bytes",
        flush=True,
    )


if __name__ == "__main__":
    main()
