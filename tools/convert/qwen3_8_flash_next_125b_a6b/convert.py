"""Build the Qwen3.8 Flash-Next 125B-A6B NVFP4 `.ninfer` artifact.

The checkpoint's numerical words are retained. BF16 projections and PLE FP8
tensors are copied directly, channel-wise convolution kernels are transposed
to NInfer's channel-fast runtime layout, and expert-major ModelOpt NVFP4
tensors are rearranged into the closed NInfer bank layout without dequantizing
or requantizing them.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Iterable, Iterator, Sequence

import torch

from tools.artifact.writer import ArtifactWriter
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, plan_objects
from tools.artifact.codecs.direct import encode_direct
from tools.artifact.codecs.nvfp4 import swizzle_nvfp4_scales
from tools.artifact.codecs.row_split import encode_row_split
from tools.convert.sources.safetensors import SafetensorsSource, TensorInfo
from tools.convert.quantization.groupwise import pick_device, quantize_matrix
from . import draft_head, inventory, descriptor, vision


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


SOURCE_PROFILES = {
    "radixark": ("RadixArk/Qwen3.8-Flash-Next-NVFP4",
                  "qwen3_8_flash_next_125b_a6b_nvfp4.ninfer", inventory.FP8),
    "swift": ("ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4",
              "swift_1_5_qwen3_8_flash_next_nvfp4.ninfer", inventory.BF16),
}
RECIPE_ID = "qwen3_8_flash_next_125b_a6b_nvfp4-v3"

_PLE_PREFIX = (
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
)
_PLE_TABLE = _PLE_PREFIX + "weight"
_PLE_SHARDS = tuple(_PLE_PREFIX + f"shard_{part}.weight" for part in range(128))
_PLE_METADATA = frozenset(
    {
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
    }
)
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


def _expert_source(layer: int, expert: int, projection: str, field: str) -> str:
    return (
        f"model.language_model.layers.{layer}.mlp.experts.{expert}."
        f"{projection}.{field}"
    )


def _bank_name(layer: int, role: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts.{role}"


def _direct_source_specs(ple_format: str = inventory.FP8) -> tuple[inventory.TensorSpec, ...]:
    special = {
        _bank_name(layer, role)
        for layer in inventory.LAYERS
        for role in (
            "gate_up",
            "gate_up_input_divisors",
            "down",
            "down_input_divisors",
        )
    }
    return tuple(
        spec
        for spec in inventory.object_specs(ple_format)
        if isinstance(spec, inventory.TensorSpec)
        if spec.id not in special
        and spec.id != _PLE_TABLE
        and spec.id not in (_DRAFT_HEAD, _DRAFT_HEAD_IDS)
        and spec.id not in _VISION_BY_NAME
    )


def _expected_source_signatures(
    ple_format: str = inventory.FP8,
) -> dict[str, tuple[tuple[int, ...], str]]:
    expected = {
        spec.id: (spec.shape, "BF16") for spec in _direct_source_specs(ple_format)
    }
    expected.update(
        vision.signatures()
    )
    for name in _CONVOLUTION_NAMES:
        expected[name] = ((10240, 1, 4), "BF16")
    for name in _PLE_SHARDS:
        expected[name] = ((2_500_012, 160),
                          "BF16" if ple_format == inventory.BF16 else "F8_E4M3")
    for layer in inventory.LAYERS:
        for expert in range(inventory.EXPERTS):
            for projection, n, k in (
                ("gate_proj", 640, 2560),
                ("up_proj", 640, 2560),
                ("down_proj", 2560, 640),
            ):
                expected[_expert_source(layer, expert, projection, "weight")] = (
                    (n, k // 2),
                    "U8",
                )
                expected[_expert_source(layer, expert, projection, "weight_scale")] = (
                    (n, k // 16),
                    "F8_E4M3",
                )
                expected[_expert_source(layer, expert, projection, "weight_scale_2")] = (
                    (),
                    "F32",
                )
                expected[_expert_source(layer, expert, projection, "input_scale")] = (
                    (),
                    "F32",
                )
    return expected


def _validate_config(model_dir: Path, ple_format: str = inventory.FP8) -> dict[str, object]:
    config = json.loads((model_dir / "config.json").read_text())
    text = config.get("text_config")
    vision = config.get("vision_config")
    quant = config.get("quantization_config")
    if not isinstance(text, dict) or not isinstance(vision, dict) or not isinstance(quant, dict):
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
    if ple_format == inventory.FP8:
        check_members("text_config", text, {"ple_embedding_dtype": "float8_e4m3fn"})
    elif ple_format == inventory.BF16:
        if text.get("ple_embedding_dtype") not in (None, "bfloat16"):
            raise ValueError("Swift PLE must be BF16")
    else:
        raise ValueError(f"unsupported PLE format: {ple_format}")
    check_members(
        "vision_config",
        vision,
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
        {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    )
    return {
        "layers": 48,
        "hidden_size": 2560,
        "experts": 512,
        "active_experts": 10,
        "max_context": 262144,
        "ple_rows": 320001536,
    }


def _validate_source(
    reader: SafetensorsSource,
    ple_format: str = inventory.FP8,
) -> tuple[dict[str, TensorInfo], dict[str, int]]:
    expected = _expected_source_signatures(ple_format)
    actual_names = frozenset(reader.weight_map)
    expected_names = frozenset(expected)
    unexpected = actual_names - expected_names - _PLE_METADATA
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


def _matching_pair(
    reader: SafetensorsSource,
    layer: int,
    expert: int,
    field: str,
) -> torch.Tensor:
    gate_name = _expert_source(layer, expert, "gate_proj", field)
    up_name = _expert_source(layer, expert, "up_proj", field)
    gate = read_tensor(reader, gate_name).detach().contiguous().cpu()
    up = read_tensor(reader, up_name).detach().contiguous().cpu()
    if gate.dtype != up.dtype or gate.shape != up.shape or not torch.equal(gate, up):
        raise ValueError(f"layer {layer} expert {expert}: gate/up {field} words differ")
    return gate


def _expert_bank_payload(
    reader: SafetensorsSource,
    layer: int,
    role: str,
) -> Iterator[bytes]:
    if role not in ("gate_up", "down"):
        raise ValueError(f"invalid expert bank role: {role}")
    projections = ("gate_proj", "up_proj") if role == "gate_up" else ("down_proj",)
    n, k = (1280, 2560) if role == "gate_up" else (2560, 640)

    for expert in range(inventory.EXPERTS):
        pieces = [
            read_tensor(reader, _expert_source(layer, expert, projection, "weight"))
            for projection in projections
        ]
        if any(piece.dtype != torch.uint8 for piece in pieces):
            raise ValueError(f"layer {layer} expert {expert}: packed weight is not U8")
        packed = pieces[0].contiguous() if len(pieces) == 1 else torch.cat(pieces, dim=0)
        if tuple(packed.shape) != (n, k // 2):
            raise ValueError(f"layer {layer} expert {expert}: packed weight shape mismatch")
        yield packed.numpy().tobytes()

    for expert in range(inventory.EXPERTS):
        pieces = [
            read_tensor(reader, _expert_source(layer, expert, projection, "weight_scale"))
            for projection in projections
        ]
        if any(piece.dtype != torch.float8_e4m3fn for piece in pieces):
            raise ValueError(f"layer {layer} expert {expert}: weight scale is not E4M3FN")
        scales = pieces[0].contiguous() if len(pieces) == 1 else torch.cat(pieces, dim=0)
        if tuple(scales.shape) != (n, k // 16):
            raise ValueError(f"layer {layer} expert {expert}: weight scale shape mismatch")
        yield swizzle_nvfp4_scales(scales.view(torch.uint8), (n, k)).numpy().tobytes()

    divisors = torch.empty(inventory.EXPERTS, dtype=torch.float32)
    for expert in range(inventory.EXPERTS):
        if role == "gate_up":
            scale = _matching_pair(reader, layer, expert, "weight_scale_2")
            name = _expert_source(layer, expert, "gate_proj", "weight_scale_2")
        else:
            name = _expert_source(layer, expert, "down_proj", "weight_scale_2")
            scale = read_tensor(reader, name)
        divisors[expert] = _positive_reciprocal(scale, name)
    yield encode_direct(divisors, inventory.FP32)


def _input_divisors(reader: SafetensorsSource, layer: int, role: str) -> bytes:
    projection = "gate_proj" if role == "gate_up" else "down_proj"
    values = torch.empty(inventory.EXPERTS, dtype=torch.float32)
    for expert in range(inventory.EXPERTS):
        if role == "gate_up":
            scale = _matching_pair(reader, layer, expert, "input_scale")
        else:
            scale = read_tensor(reader, _expert_source(layer, expert, projection, "input_scale"))
        name = _expert_source(layer, expert, projection, "input_scale")
        values[expert] = _positive_reciprocal(scale, name)
    return encode_direct(values, inventory.FP32)


def _ple_payload(reader: SafetensorsSource, ple_format: str) -> Iterator[bytes]:
    for name in _PLE_SHARDS:
        tensor = read_tensor(reader, name)
        dtype = torch.bfloat16 if ple_format == inventory.BF16 else torch.float8_e4m3fn
        if tensor.dtype != dtype or tuple(tensor.shape) != (2_500_012, 160):
            raise ValueError(f"{name}: PLE shard signature mismatch")
        yield encode_direct(tensor, ple_format)


def _payload(
    spec: inventory.TensorSpec,
    reader: SafetensorsSource,
    device: torch.device,
    draft: draft_head.DraftHeadContext,
    ple_format: str = inventory.FP8,
) -> bytes | Iterable[bytes]:
    if spec.id == _DRAFT_HEAD_IDS:
        return encode_direct(draft_head.materialize_draft_head_token_ids(draft), inventory.I32)
    if spec.id == _DRAFT_HEAD:
        full_head = read_tensor(reader, "lm_head.weight")
        selected = draft_head.materialize_draft_head(full_head, draft)
        return encode_tensor_payload(selected, spec, device)
    if spec.id == _PLE_TABLE:
        return _ple_payload(reader, ple_format)
    for layer in inventory.LAYERS:
        prefix = _bank_name(layer, "")
        if spec.id == prefix + "gate_up":
            return _expert_bank_payload(reader, layer, "gate_up")
        if spec.id == prefix + "down":
            return _expert_bank_payload(reader, layer, "down")
        if spec.id == prefix + "gate_up_input_divisors":
            return _input_divisors(reader, layer, "gate_up")
        if spec.id == prefix + "down_input_divisors":
            return _input_divisors(reader, layer, "down")
    if spec.id in _VISION_BY_NAME:
        value = read_tensor(reader, _VISION_BY_NAME[spec.id]).reshape(spec.shape)
        return encode_tensor_payload(value, spec, device)
    tensor = read_tensor(reader, spec.id)
    expected_shape = (10240, 1, 4) if spec.id in _CONVOLUTION_NAMES else spec.shape
    if tuple(tensor.shape) != expected_shape or tensor.dtype != torch.bfloat16:
        raise ValueError(f"{spec.id}: direct source signature mismatch")
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
    source_profile: str = "radixark",
) -> Path:
    source = Path(model_dir)
    output = Path(out_path)
    source_repository, output_basename, ple_format = SOURCE_PROFILES[source_profile]
    if output.name != output_basename:
        raise ValueError(f"output basename must be {output_basename!r}")
    started = time.perf_counter()
    resolved_device = pick_device(device)
    config_summary = _validate_config(source, ple_format)
    resource_map = {name: (source / name.removeprefix("frontend/")).read_bytes()
                    for name in inventory.RESOURCE_SPECS}
    object_specs = inventory.object_specs(ple_format)
    specs = [ResourceSpec(name, len(data)) for name, data in resource_map.items()] + [
        spec for spec in object_specs if isinstance(spec, inventory.TensorSpec)]
    objects = plan_objects(specs)
    description = descriptor.describe([o.to_json() for o in objects])

    with SafetensorsSource(source) as reader:
        _, dtype_counts = _validate_source(reader, ple_format)
        draft = draft_head.compute_shortlist(
            Path(__file__).resolve().parents[3] / draft_head.DEFAULT_RANKING,
            source,
            read_tensor(reader, "lm_head.weight"),
            n=147_456,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(
            output,
            specs, **description,
            metadata={"name": source_repository if source_profile == "swift" else inventory.MODEL_ID},
            provenance={"source": source_repository, "recipe": RECIPE_ID + "-" + source_profile},
            random_access_objects=(_PLE_TABLE,),
        ) as writer:
            for index, spec in enumerate(object_specs, start=1):
                payload = (
                    resource_map[spec]
                    if isinstance(spec, str)
                    else _payload(spec, reader, resolved_device, draft, ple_format)
                )
                writer.write_object(spec if isinstance(spec, str) else spec.id, payload)
                print(f"[{index}/{len(object_specs)}] {spec if isinstance(spec, str) else spec.id}", flush=True)

    with Artifact(output) as artifact:
        file_bytes = artifact.file_bytes
    elapsed = time.perf_counter() - started
    report = {
        "identity": {"model_id": inventory.MODEL_ID, "weights_id": inventory.WEIGHTS_ID},
        "target_key": inventory.TARGET_KEY,
        "recipe_id": RECIPE_ID,
        "source": {"repository": source_repository, "path": str(source.resolve())},
        "output": str(output.resolve()),
        "config_summary": config_summary,
        "source_dtype_counts": dtype_counts,
        "objects": {"count": len(objects), "payload_bytes": sum(o.bytes for o in objects)},
        "file_bytes": file_bytes,
        "elapsed_seconds": elapsed,
        "conversion_device": str(resolved_device),
        "ple_materialization": "file-backed-read-only",
        "ple_format": ple_format,
        "proposal_shortlist": "frequency-rank-plus-lm-head-norm-rank",
    }
    report_path = Path(str(output) + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"complete: {file_bytes} bytes in {elapsed:.1f}s", flush=True)
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-profile", choices=SOURCE_PROFILES, default="radixark")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device, source_profile=args.source_profile)


if __name__ == "__main__":
    main()
