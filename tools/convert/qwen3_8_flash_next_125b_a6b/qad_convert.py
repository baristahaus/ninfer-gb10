"""Build the QAD-native Qwen3.8 Flash-Next 125B-A6B `.ninfer` artifact.

Native ingestion of the QAD checkpoint family
(``edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD``): MXFP8 dense
projections, NVFP4 W4A16 MTP expert banks, and the NVFP4 group-16 PLE table are
stored in the source-declared words without dequantizing or requantizing them
(user decision, 2026-10-15). BF16 leaves, the vision tower, and the shortlist
proposal head follow the vendor recipe unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable, Iterator, Sequence

import torch

from tools.artifact.writer import ArtifactWriter
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, plan_objects
from tools.artifact.codecs.direct import encode_direct
from tools.artifact.codecs.nvfp4 import (
    encode_block_scale_k16,
    swizzle_nvfp4_scales,
)
from tools.artifact.codecs.mxfp8 import encode_mxfp8
from tools.artifact.codecs.row_split import encode_row_split
from tools.convert.sources.safetensors import SafetensorsSource, TensorInfo
from tools.convert.quantization.groupwise import pick_device, quantize_matrix
from . import convert as vendor
from . import draft_head, descriptor, qad_inventory as inventory, vision


def read_tensor(reader, name):
    return reader.read_flat(name).reshape(reader.describe(name).shape)


def encode_tensor_payload(value, spec, device):
    if spec.layout == inventory.CONTIGUOUS:
        return encode_direct(value.float() if spec.format == inventory.FP32 else value, spec.format)
    quantized = quantize_matrix(value, spec.format, device=device)
    return encode_row_split(quantized.codes, quantized.scales, spec.format, spec.shape)


def check_members(label, values, expected):
    for key, value in expected.items():
        if values.get(key) != value:
            raise ValueError(f"{label}.{key}: expected {value!r}, got {values.get(key)!r}")


OUTPUT_BASENAME = inventory.OUTPUT_BASENAME
RECIPE_ID = inventory.RECIPE_ID
SOURCE_REPOSITORY = inventory.SOURCE_REPOSITORY

_PLE_PREFIX = inventory._PLE_PREFIX
_QAD_PLE_SHARDS = tuple(range(128))
_QAD_PLE_ROWS = 2_500_012
_QAD_PLE_WIDTH = 160
_QAD_PLE_METADATA = inventory._PLE_METADATA
_DRAFT_HEAD = "ninfer.optimized_proposal_head.weight"
_DRAFT_HEAD_IDS = "ninfer.optimized_proposal_head.token_ids"
_VISION_BY_NAME = vision.SOURCES
_CONVOLUTION_NAMES = frozenset(
    {
        *(f"model.language_model.layers.{layer}.linear_attn.conv1d.weight"
          for layer in inventory.GDN_LAYERS),
        "model.language_model.layers.1.ple.conv1d.weight",
    }
)

_MTP_EXPERTS = 512
_MTP_MLP = "mtp.layers.0.mlp."
_MTP_PROJECTIONS = {
    "gate_up": (("gate_proj", "up_proj"), 1280, 2560),
    "down": (("down_proj",), 2560, 640),
}

# Norm weights are FP32 in the QAD source (BF16 in the vendor recipe). The
# artifact keeps the vendor BF16 runtime contract; the converter rounds
# round-to-nearest-even and the closed set pins the source dtype to F32.
_F32_SOURCE_SUFFIXES = (
    "hc_norm.weight",
    "ple.norm_conv.weight",
    "ple.norm_key.weight",
    "ple.norm_query.weight",
)


def _source_dtype(spec_id: str) -> str:
    if spec_id.endswith(("ple.norm_conv.weight", "ple.norm_key.weight", "ple.norm_query.weight")):
        return "F32"
    if spec_id.endswith("hc_norm.weight") and not spec_id.startswith("mtp."):
        return "F32"
    return "BF16"


def _mtp_expert_source(role: str, expert: int, projection: str, field: str) -> str:
    return f"{_MTP_MLP}experts.{expert}.{projection}.{field}"


def _mtp_scale_source(spec_id: str) -> str:
    """Source scale name of a MXFP8 site: ``...X.weight`` -> ``...X.weight_scale``."""

    if not spec_id.endswith(".weight"):
        raise ValueError(f"{spec_id}: MXFP8 sites are weight projections")
    return spec_id[: -len(".weight")] + ".weight_scale"


def _expected_source_signatures() -> dict[str, tuple[tuple[int, ...], str]]:
    expected: dict[str, tuple[tuple[int, ...], str]] = {}
    for spec in inventory.TENSOR_SPECS:
        if spec.id in (_DRAFT_HEAD, _DRAFT_HEAD_IDS):
            continue
        if spec.id in _VISION_BY_NAME:
            continue
        if spec.id.endswith(("_input_divisors", "_weight_divisors")):
            continue
        if spec.layout == inventory.MXFP8_LAYOUT:
            expected[spec.id] = (spec.shape, "F8_E4M3")
            n, k = spec.shape
            expected[_mtp_scale_source(spec.id)] = ((n, k // 32), "U8")
        elif spec.layout == inventory.PLE_TABLE_LAYOUT:
            continue
        elif spec.layout == inventory.EXPERT_NVFP4:
            continue
        else:
            expected[spec.id] = (spec.shape, _source_dtype(spec.id))
    expected.update(vision.signatures())
    for name in _CONVOLUTION_NAMES:
        expected[name] = ((10240, 1, 4), "BF16")
    for shard in _QAD_PLE_SHARDS:
        expected[f"{_PLE_PREFIX}shard_{shard}.weight"] = ((_QAD_PLE_ROWS, _QAD_PLE_WIDTH // 2), "U8")
        expected[f"{_PLE_PREFIX}shard_{shard}.weight_scale"] = ((_QAD_PLE_ROWS, 10), "F8_E4M3")
    expected[f"{_PLE_PREFIX}weight_scale_2"] = ((1,), "F32")
    _MTP_SOURCE_PROJECTIONS = {"gate_up": ("gate_proj", "up_proj"), "down": ("down_proj",)}
    for role, projections in _MTP_SOURCE_PROJECTIONS.items():
        for expert in range(_MTP_EXPERTS):
            for projection in projections:
                n, k = (640, 2560) if projection != "down_proj" else (2560, 640)
                expected[_mtp_expert_source(role, expert, projection, "weight")] = (
                    (n, k // 2), "U8",
                )
                expected[_mtp_expert_source(role, expert, projection, "weight_scale")] = (
                    (n, k // 16), "F8_E4M3",
                )
                expected[_mtp_expert_source(role, expert, projection, "weight_scale_2")] = (
                    (), "F32",
                )
    for layer in inventory.LAYERS:
        for expert in range(inventory.EXPERTS):
            for projection, n, k in (
                ("gate_proj", 640, 2560),
                ("up_proj", 640, 2560),
                ("down_proj", 2560, 640),
            ):
                expected[vendor._expert_source(layer, expert, projection, "weight")] = (
                    (n, k // 2), "U8",
                )
                expected[vendor._expert_source(layer, expert, projection, "weight_scale")] = (
                    (n, k // 16), "F8_E4M3",
                )
                expected[vendor._expert_source(layer, expert, projection, "weight_scale_2")] = (
                    (), "F32",
                )
                expected[vendor._expert_source(layer, expert, projection, "input_scale")] = (
                    (), "F32",
                )
    return expected


def _validate_config(model_dir: Path) -> dict[str, object]:
    config = json.loads((model_dir / "config.json").read_text())
    text = config.get("text_config")
    vision_config = config.get("vision_config")
    quant = config.get("quantization_config")
    if not isinstance(text, dict) or not isinstance(vision_config, dict) or not isinstance(quant, dict):
        raise ValueError("checkpoint config is missing text, vision, or quantization config")
    check_members(
        "config",
        config,
        {
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "tie_word_embeddings": False,
        },
    )
    check_members(
        "text_config",
        text,
        {
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "full_attention_interval": 4,
            "hc_count": 4,
            "hc_lowrank": 320,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "max_position_embeddings": 262144,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "split_ngram_parts": 128,
            "mtp_num_hidden_layers": 1,
        },
    )
    check_members(
        "vision_config",
        vision_config,
        {
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
            "out_hidden_size": 2560,
        },
    )
    check_members(
        "quantization_config",
        quant,
        {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": None},
    )
    return {
        "layers": 48,
        "hidden_size": 2560,
        "experts": 512,
        "active_experts": 10,
        "max_context": 262144,
        "ple_rows": 320001536,
        "weights": inventory.WEIGHTS_ID,
    }


def _validate_source(
    reader: SafetensorsSource,
) -> tuple[dict[str, TensorInfo], dict[str, int]]:
    expected = _expected_source_signatures()
    actual_names = frozenset(reader.weight_map)
    expected_names = frozenset(expected)
    unexpected = actual_names - expected_names - _QAD_PLE_METADATA
    missing = expected_names - actual_names
    if unexpected or missing:
        detail = sorted(unexpected)[0] if unexpected else sorted(missing)[0]
        kind = "unexpected" if unexpected else "missing"
        raise ValueError(f"checkpoint tensor allocation is not closed: {kind} {detail}")
    metadata = {name: reader.describe(name) for name in expected_names}
    counts: Counter[str] = Counter()
    for name, signature in expected.items():
        item = metadata[name]
        if (item.shape, item.dtype) != signature:
            raise ValueError(
                f"{name}: source signature {(item.shape, item.dtype)} != {signature}"
            )
        counts[item.dtype] += 1
    return metadata, dict(sorted(counts.items()))


def _positive_reciprocal(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.dtype != torch.float32 or value.numel() != 1:
        raise ValueError(f"{name}: expected one FP32 scale")
    value = value.detach().reshape(()).cpu()
    if not bool(torch.isfinite(value)) or not bool(value > 0):
        raise ValueError(f"{name}: scale must be finite and positive")
    result = value.reciprocal()
    if not bool(torch.isfinite(result)) or not bool(result > 0):
        raise ValueError(f"{name}: reciprocal scale is not finite and positive")
    return result


def _mtp_weight_divisors(reader: SafetensorsSource, role: str) -> Iterator[bytes]:
    if role == "gate_up":
        divisors = torch.empty(_MTP_EXPERTS, dtype=torch.float32)
        for expert in range(_MTP_EXPERTS):
            name = _mtp_expert_source(role, expert, "gate_proj", "weight_scale_2")
            divisors[expert] = _positive_reciprocal(read_tensor(reader, name), name)
    else:
        divisors = torch.empty(_MTP_EXPERTS, dtype=torch.float32)
        for expert in range(_MTP_EXPERTS):
            name = _mtp_expert_source(role, expert, "down_proj", "weight_scale_2")
            divisors[expert] = _positive_reciprocal(read_tensor(reader, name), name)
    yield encode_direct(divisors, inventory.FP32)


def _mtp_bank_payload(reader: SafetensorsSource, role: str) -> Iterator[bytes]:
    if role not in _MTP_PROJECTIONS:
        raise ValueError(f"invalid MTP bank role: {role}")
    projections, n, k = _MTP_PROJECTIONS[role]

    for expert in range(_MTP_EXPERTS):
        pieces = [
            read_tensor(reader, _mtp_expert_source(role, expert, projection, "weight"))
            for projection in projections
        ]
        if any(piece.dtype != torch.uint8 for piece in pieces):
            raise ValueError(f"MTP expert {expert}: packed weight is not U8")
        packed = pieces[0].contiguous() if len(pieces) == 1 else torch.cat(pieces, dim=0)
        if tuple(packed.shape) != (n, k // 2):
            raise ValueError(f"MTP expert {expert}: packed weight shape mismatch")
        yield packed.numpy().tobytes()

    for expert in range(_MTP_EXPERTS):
        pieces = [
            read_tensor(reader, _mtp_expert_source(role, expert, projection, "weight_scale"))
            for projection in projections
        ]
        if any(piece.dtype != torch.float8_e4m3fn for piece in pieces):
            raise ValueError(f"MTP expert {expert}: weight scale is not E4M3FN")
        scales = pieces[0].contiguous() if len(pieces) == 1 else torch.cat(pieces, dim=0)
        if tuple(scales.shape) != (n, k // 16):
            raise ValueError(f"MTP expert {expert}: weight scale shape mismatch")
        yield swizzle_nvfp4_scales(scales.view(torch.uint8), (n, k)).numpy().tobytes()

    yield from _mtp_weight_divisors(reader, role)


def _mxfp8_payload(reader: SafetensorsSource, spec) -> bytes:
    codes = read_tensor(reader, spec.id)
    scales = read_tensor(reader, _mtp_scale_source(spec.id))
    if codes.dtype != torch.float8_e4m3fn:
        raise ValueError(f"{spec.id}: MXFP8 weight is not E4M3FN")
    if scales.dtype != torch.uint8:
        raise ValueError(f"{spec.id}: MXFP8 scale words are not U8")
    n, k = spec.shape
    if tuple(scales.shape) != (n, k // 32):
        raise ValueError(f"{spec.id}: MXFP8 scale shape mismatch")
    return encode_mxfp8(codes.view(torch.uint8), scales, (n, k))


def _qad_ple_payload(reader: SafetensorsSource) -> Iterator[bytes]:
    scale2 = read_tensor(reader, f"{_PLE_PREFIX}weight_scale_2")
    if scale2.dtype != torch.float32 or scale2.numel() != 1:
        raise ValueError("PLE weight_scale_2 must be one FP32 word")
    scale2 = scale2.detach().reshape(()).cpu()
    if not bool(torch.isfinite(scale2)) or not bool(scale2 > 0):
        raise ValueError("PLE weight_scale_2 must be finite and positive")

    for shard in _QAD_PLE_SHARDS:
        name = f"{_PLE_PREFIX}shard_{shard}.weight"
        tensor = read_tensor(reader, name)
        if tensor.dtype != torch.uint8 or tuple(tensor.shape) != (_QAD_PLE_ROWS, _QAD_PLE_WIDTH // 2):
            raise ValueError(f"{name}: PLE shard signature mismatch")
        yield tensor.contiguous().numpy().tobytes()

    for shard in _QAD_PLE_SHARDS:
        name = f"{_PLE_PREFIX}shard_{shard}.weight_scale"
        tensor = read_tensor(reader, name)
        if tensor.dtype != torch.float8_e4m3fn or tuple(tensor.shape) != (_QAD_PLE_ROWS, 10):
            raise ValueError(f"{name}: PLE shard scale signature mismatch")
        yield tensor.view(torch.uint8).contiguous().numpy().tobytes()

    yield encode_direct(scale2, inventory.FP32)


def _payload(
    spec: inventory.TensorSpec,
    reader: SafetensorsSource,
    device: torch.device,
    draft: draft_head.DraftHeadContext,
) -> bytes | Iterable[bytes]:
    if spec.id == _DRAFT_HEAD_IDS:
        return encode_direct(draft_head.materialize_draft_head_token_ids(draft), inventory.I32)
    if spec.id == _DRAFT_HEAD:
        full_head = read_tensor(reader, "lm_head.weight")
        selected = draft_head.materialize_draft_head(full_head, draft)
        return encode_tensor_payload(selected, spec, device)
    if spec.layout == inventory.PLE_TABLE_LAYOUT:
        return _qad_ple_payload(reader)
    if spec.layout == inventory.MXFP8_LAYOUT:
        return _mxfp8_payload(reader, spec)
    for layer in inventory.LAYERS:
        prefix = vendor._bank_name(layer, "")
        if spec.id == prefix + "gate_up":
            return vendor._expert_bank_payload(reader, layer, "gate_up")
        if spec.id == prefix + "down":
            return vendor._expert_bank_payload(reader, layer, "down")
        if spec.id == prefix + "gate_up_input_divisors":
            return vendor._input_divisors(reader, layer, "gate_up")
        if spec.id == prefix + "down_input_divisors":
            return vendor._input_divisors(reader, layer, "down")
    if spec.id == _MTP_MLP + "experts.gate_up":
        return _mtp_bank_payload(reader, "gate_up")
    if spec.id == _MTP_MLP + "experts.down":
        return _mtp_bank_payload(reader, "down")
    if spec.id == _MTP_MLP + "experts.gate_up_weight_divisors":
        return bytes(next(iter(_mtp_weight_divisors(reader, "gate_up"))))
    if spec.id == _MTP_MLP + "experts.down_weight_divisors":
        return bytes(next(iter(_mtp_weight_divisors(reader, "down"))))
    if spec.id in _VISION_BY_NAME:
        value = read_tensor(reader, _VISION_BY_NAME[spec.id]).reshape(spec.shape)
        return encode_tensor_payload(value, spec, device)
    tensor = read_tensor(reader, spec.id)
    expected_shape = (10240, 1, 4) if spec.id in _CONVOLUTION_NAMES else spec.shape
    if tuple(tensor.shape) != expected_shape or tensor.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"{spec.id}: direct source signature mismatch")
    if tensor.dtype == torch.float32:
        tensor = tensor.to(torch.bfloat16)
    if spec.id in _CONVOLUTION_NAMES:
        tensor = tensor[:, 0, :].transpose(0, 1).contiguous()
    if spec.format == inventory.FP32:
        return encode_direct(tensor.float(), inventory.FP32)
    return encode_direct(tensor, inventory.BF16)


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    source = Path(model_dir)
    output = Path(out_path)
    if output.name != OUTPUT_BASENAME:
        raise ValueError(f"output basename must be {OUTPUT_BASENAME!r}")
    started = time.perf_counter()
    resolved_device = pick_device(device)
    config_summary = _validate_config(source)
    resource_map = {name: (source / name.removeprefix("frontend/")).read_bytes()
                    for name in inventory.RESOURCE_SPECS}
    specs = [ResourceSpec(name, len(data)) for name, data in resource_map.items()] + list(inventory.TENSOR_SPECS)
    objects = plan_objects(specs)
    description = descriptor.describe([o.to_json() for o in objects])

    with SafetensorsSource(source) as reader:
        _, dtype_counts = _validate_source(reader)
        draft = draft_head.compute_shortlist(
            Path(__file__).resolve().parents[3] / draft_head.DEFAULT_RANKING,
            source,
            read_tensor(reader, "lm_head.weight"),
            n=147_456,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(
            output,
            specs, **description, metadata={"name": inventory.MODEL_ID},
            provenance={"source": SOURCE_REPOSITORY, "recipe": RECIPE_ID},
        ) as writer:
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                payload = (
                    resource_map[spec]
                    if isinstance(spec, str)
                    else _payload(spec, reader, resolved_device, draft)
                )
                writer.write_object(spec if isinstance(spec, str) else spec.id, payload)
                print(f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec if isinstance(spec, str) else spec.id}", flush=True)
        _verify_output(output, resource_map, reader, draft, resolved_device)

    elapsed = time.perf_counter() - started
    report = {
        "identity": {"model_id": inventory.MODEL_ID, "weights_id": inventory.WEIGHTS_ID},
        "target_key": inventory.TARGET_KEY,
        "recipe_id": RECIPE_ID,
        "source": {"repository": SOURCE_REPOSITORY, "path": str(source.resolve())},
        "output": str(output.resolve()),
        "config_summary": config_summary,
        "source_dtype_counts": dtype_counts,
        "objects": {"count": len(objects), "payload_bytes": sum(o.bytes for o in objects)},
        "file_bytes": _file_bytes(output),
        "verification": "post-write sha256 of every re-derived payload",
        "elapsed_seconds": elapsed,
        "conversion_device": str(resolved_device),
        "ple_materialization": "file-backed-read-only",
        "proposal_shortlist": "frequency-rank-plus-lm-head-norm-rank",
    }
    report_path = Path(str(output) + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"complete: {report['file_bytes']} bytes in {elapsed:.1f}s", flush=True)
    return report_path


def _file_bytes(output: Path) -> int:
    with Artifact(output) as artifact:
        return artifact.file_bytes


def _digest(chunks: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _chunks(payload: bytes | Iterable[bytes]) -> Iterable[bytes]:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return (bytes(payload),)
    return payload


def _verify_output(
    output: Path,
    resource_map: dict[str, bytes],
    reader: SafetensorsSource,
    draft: draft_head.DraftHeadContext,
    device: torch.device,
) -> None:
    """Post-write reopen: re-derive every payload from the source and compare
    SHA-256 digests against the written bytes (D2). Deterministic encoders make
    the re-derivation exact; a mismatch fails the conversion.
    """

    with Artifact(output) as artifact:
        for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
            if isinstance(spec, str):
                expected = _digest((resource_map[spec],))
            else:
                expected = _digest(_chunks(_payload(spec, reader, device, draft)))
            actual = _digest(artifact.iter_object(spec if isinstance(spec, str) else spec.id))
            if expected != actual:
                name = spec if isinstance(spec, str) else spec.id
                raise ValueError(f"{name}: post-write payload verification failed")
            if index % 500 == 0:
                print(f"[verify {index}/{len(inventory.OBJECT_SPECS)}]", flush=True)
    print("[verify] all payloads match", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device)


if __name__ == "__main__":
    main()
