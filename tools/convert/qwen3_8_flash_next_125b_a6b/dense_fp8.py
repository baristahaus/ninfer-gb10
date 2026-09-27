"""Re-encode the Qwen3.8 Flash-Next 125B-A6B dense projections as row-scaled FP8.

The input is the BF16-projection artifact written by `convert` (or its v3 upgrade); its BF16 words
are the checkpoint's words, so no source checkpoint is needed. The projections whose Ops accept
`fp8_e4m3fn_row_bf16` are quantized from those exact words with `fp8_row_maxabs` rounding: one BF16
row multiplier (the round-to-nearest-even BF16 value of the row's max |w| / 448) and E4M3FN codes
rounded to nearest even. They are the GDN query/key/value, output-gate and output projections, the
QSA packed query/gate and output projections, the text HyperConnection down projections and the
output head. Their A16Only Uses, like every binding and component, are unchanged. Every other
object, including the MTP layer, the routed experts and the PLE table, is copied byte for byte.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence

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
OUTPUT_BASENAME = "qwen3_8_flash_next_125b_a6b_nvfp4_fp8.ninfer"
RECIPE_ID = "qwen3_8_flash_next_125b_a6b_nvfp4_fp8_dense-v3"
# Row chunks of about this many elements are quantized concurrently; the NumPy rounding releases
# the GIL, so the pool scales with the host's cores.
QUANTIZE_CHUNK_ELEMENTS = 1 << 21
_E4M3FN_VALUES = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).double().numpy()


def dense_fp8_parameters() -> dict[str, tuple[int, int]]:
    """Logical text projections re-encoded by this recipe, with their exact shapes."""

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
            parameters[prefix + connection + "input_mix_weight_down.weight"] = (320, 10240)
    parameters[text + "hyper_connection_mixer.input_mix_weight_down.weight"] = (320, 10240)
    parameters["lm_head.weight"] = (248320, 2560)
    return parameters


def _binding_objects(binding: object, objects: Mapping[str, object], label: str) -> set[str]:
    return {object_id for object_id, _, _ in binding_parts(binding, objects, label)}


def _select_objects(
    artifact: Artifact, parameters: Mapping[str, tuple[int, int]]
) -> dict[str, tuple[int, int]]:
    """Resolve each parameter to one whole BF16 object that no other consumer references."""

    directory = artifact.directory
    selected: dict[str, tuple[int, int]] = {}
    for name, shape in parameters.items():
        binding = directory.bindings.get(name)
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
        if object_id in selected:
            raise ValueError(f"{name}: object {object_id} is shared by two selected parameters")
        selected[object_id] = shape
    for name, binding in directory.bindings.items():
        if name not in parameters and _binding_objects(binding, artifact.by_id, name) & set(selected):
            raise ValueError(f"{name} shares a re-encoded object with a BF16 consumer")
    for use in directory.uses:
        for role, binding in use.get("auxiliaries", {}).items():
            label = f"{use['parameter']}@{use['input']}/{role}"
            if _binding_objects(binding, artifact.by_id, label) & set(selected):
                raise ValueError(f"{label} references a re-encoded object")
    return selected


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
    artifact: Artifact, obj: TensorObject, shape: tuple[int, int], pool: Executor
) -> tuple[bytes, dict]:
    """Quantize one BF16 matrix in concurrent row chunks; return its payload and error."""

    n, k = shape
    raw = bytearray(artifact.read_object(obj.id))
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
) -> dict:
    """Write a copy of `source_path` whose selected BF16 matrices are row-scaled FP8."""

    started = time.perf_counter()
    output = Path(out_path)
    with Artifact(source_path) as artifact:
        directory = artifact.directory
        selected = _select_objects(artifact, parameters)
        specs: list[TensorSpec | ResourceSpec] = []
        for obj in directory.objects:
            if isinstance(obj, ResourceObject):
                specs.append(ResourceSpec(obj.id, obj.bytes, obj.encoding))
            elif obj.id in selected:
                specs.append(TensorSpec(obj.id, tuple(obj.shape), FP8_ROW, ROW_SCALE))
            else:
                specs.append(TensorSpec(obj.id, tuple(obj.shape), obj.format, obj.layout))
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
            bindings=directory.bindings,
            uses=directory.uses,
            metadata=directory.metadata,
            provenance=provenance,
        ) as writer:
            for index, obj in enumerate(directory.objects, start=1):
                if obj.id in selected:
                    payload, tensors[obj.id] = _fp8_payload(
                        artifact, obj, selected[obj.id], pool
                    )
                    writer.write_object(obj.id, payload)
                else:
                    writer.write_object(obj.id, artifact.iter_object(obj.id))
                print(f"[{index}/{len(directory.objects)}] {obj.id}", flush=True)
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
    report = reencode(args.source, args.out, dense_fp8_parameters(), recipe_id=RECIPE_ID)
    print(
        f"complete: {report['reencoded_tensors']} tensors, "
        f"{report['payload_bytes']['source']} -> {report['payload_bytes']['output']} payload bytes",
        flush=True,
    )


if __name__ == "__main__":
    main()
