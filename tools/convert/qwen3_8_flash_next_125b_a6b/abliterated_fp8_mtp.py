"""Build the Qwen3.8 Flash-Next 125B-A6B FP8+MTP `.ninfer` artifact from a plain BF16 checkpoint.

The registered FP8+MTP profile (`dense_fp8`) is derived from the checkpoint-word artifact, whose
routed-expert words are the trained ModelOpt NVFP4 codes of the closed RadixArk allocation. A
BF16-only checkpoint (for example an abliterated fine-tune distributed as plain HF safetensors)
has no such codes, so this route quantizes from its BF16 words with the same methods the FP8
profile's own derivation uses:

- Every routed expert bank (the 48 main layers and the MTP layer) quantizes per expert with
  `nvfp4_maxabs` (`tools.convert.quantization.nvfp4`) into the same
  `expert_block_scale_k16_m128x4_v1` banks, one FP32 weight divisor per expert. The banks'
  Uses are AllowA16 with a `weight_divisor` auxiliary, the loader's supported W4A16 route. No
  activation calibration exists for a BF16-only checkpoint, so the trained artifact's AllowA4
  activation path is not claimed.
- The dense projections of the FP8 profile (the GDN query/key/value, output-gate and output
  projections, the QSA packed query/gate and output projections, the HyperConnection down and
  up projections, the shared-expert projections, the text and MTP final mixers and the output
  head) quantize with `fp8_row_maxabs`; each shared expert's gate and up rows stack into one
  `[1280,2560]` parent the fused SwiGLU consumes whole.
- The PLE table quantizes to one FP8 E4M3FN tensor (the checkpoint's BF16 shards under one
  global BF16 `weight_scale`), the Vision inventory keeps its registered Q4/Q5/Q6/W8 groupwise
  profile, the optimized proposal head re-derives from the BF16 output head and the frequency
  ranking, and every other tensor copies its checkpoint words (GDN controls as exact FP32
  expansions, convolutions transposed to the channel-fast layout).

The output basename is fixed. Conversion rejects a missing, unexpected, wrongly shaped or
wrongly typed source tensor and writes a conversion report beside the artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from tools.artifact.codecs.direct import encode_direct
from tools.artifact.codecs.fp8_row import encode_fp8_row_scaled
from tools.artifact.codecs.nvfp4 import swizzle_nvfp4_scales
from tools.artifact.codecs.row_split import encode_row_split
from tools.artifact.layouts import align_up, encoded_size
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorSpec, plan_objects
from tools.artifact.writer import ArtifactWriter
from tools.convert.quantization.fp8_row import quantize_bf16_rows
from tools.convert.quantization.groupwise import pick_device, quantize_matrix
from tools.convert.quantization.nvfp4 import represented_nvfp4, quantize_bf16_nvfp4
from tools.convert.sources.safetensors import SafetensorsSource

from . import draft_head, inventory, vision
from .dense_fp8 import (
    EXPERT_NVFP4,
    NVFP4,
    PackedParent,
    QUANTIZE_CHUNK_ELEMENTS,
    ROW_SCALE,
    FP8_ROW,
    _E4M3FN_VALUES,
    _float64,
    dense_fp8_parameters,
)

OUTPUT_BASENAME = "flash_next_125b_a6b_abliterated_fp8_mtp.ninfer"
RECIPE_ID = "qwen3_8_flash_next_125b_a6b_abliterated_fp8_mtp-v3"
SOURCE_KIND = "bf16_hf"

_PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
_PLE_TABLE = _PLE_PREFIX + "weight"
_PLE_SHARDS = tuple(_PLE_PREFIX + f"shard_{part}.weight" for part in range(128))
_PLE_SCALE = _PLE_PREFIX + "weight_scale"
_PLE_METADATA = frozenset(
    "model.language_model.layers.1.ple.ple_embedding." + name
    for name in ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")
)
_CONVOLUTION_NAMES = frozenset(
    {
        *(f"model.language_model.layers.{layer}.linear_attn.conv1d.weight"
          for layer in inventory.GDN_LAYERS),
        "model.language_model.layers.1.ple.conv1d.weight",
    }
)
_DRAFT_HEAD = "ninfer.optimized_proposal_head.weight"
_DRAFT_HEAD_IDS = "ninfer.optimized_proposal_head.token_ids"

# The dense projections re-encoded in place, verbatim from the FP8 profile's derivation.
_DENSE_FP8_IDS = frozenset(dense_fp8_parameters())

# Shared-expert gate/up parents: object id -> (source parameters, shape).
_PACKED_PARENTS = {
    prefix + "gate_up_proj.weight": PackedParent(
        (prefix + "gate_proj.weight", prefix + "up_proj.weight"), (1280, 2560)
    )
    for prefix in [
        *(f"model.language_model.layers.{layer}.mlp.shared_expert."
          for layer in inventory.LAYERS),
        "mtp.layers.0.mlp.shared_expert.",
    ]
}


def _bank_parameter(layer: int, role: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts.{role}"


def _bank_source(layer: int, role: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts.{role}_proj"


def _mtp_bank_parameter(role: str) -> str:
    return f"mtp.layers.0.mlp.experts.{role}"


def _mtp_bank_source(role: str) -> str:
    return f"mtp.layers.0.mlp.experts.{role}_proj"

_BANK_SHAPES = {"gate_up": (512, 1280, 2560), "down": (512, 2560, 640)}


def _is_bank_parameter(spec_id: str) -> bool:
    """True for a main-layer or MTP routed NVFP4 bank object id."""

    if spec_id.startswith("mtp.layers.0.mlp.experts."):
        return spec_id in (_mtp_bank_parameter("gate_up"), _mtp_bank_parameter("down"))
    prefix, separator, role = spec_id.rpartition(".")
    if not separator or role not in _BANK_SHAPES or not prefix.endswith(".mlp.experts"):
        return False
    layer = prefix.removeprefix("model.language_model.layers.").removesuffix(".mlp.experts")
    return layer.isdigit()


def _divisor_parameter(parameter: str) -> str:
    return parameter + "_weight_divisors"


def _source_name(spec_id: str) -> str | None:
    """The checkpoint tensor a spec consumes, or None when it has no single source."""

    if spec_id == _PLE_TABLE or spec_id == _PLE_SCALE:
        return None
    if spec_id in (_DRAFT_HEAD, _DRAFT_HEAD_IDS):
        return None
    if _is_bank_parameter(spec_id) or spec_id.endswith("_weight_divisors"):
        return None
    if spec_id.endswith("_input_divisors"):
        return None  # activation divisors need trained calibration; this route drops them
    if spec_id in _PACKED_PARENTS:
        return spec_id  # the parent's rows stack two separately named sources; validated below
    if spec_id in _DENSE_FP8_IDS:
        return spec_id
    if spec_id.startswith("vision/"):
        return vision.SOURCES[spec_id]
    return spec_id


def _source_shape(source: str, spec_shape: tuple[int, ...]) -> tuple[int, ...]:
    if source in _CONVOLUTION_NAMES:
        return (10240, 1, 4)
    if source == "model.visual.patch_embed.proj.weight":
        return (1152, 3, 2, 16, 16)
    return spec_shape


def _expected_sources() -> dict[str, tuple[tuple[int, ...], str]]:
    """Every checkpoint tensor this route consumes with its exact signature."""

    expected: dict[str, tuple[tuple[int, ...], str]] = {}
    for spec in inventory.TENSOR_SPECS:
        source = _source_name(spec.id)
        if source is None:
            continue
        if spec.id in _PACKED_PARENTS:
            for name in _PACKED_PARENTS[spec.id].parameters:
                expected[name] = (_PACKED_PARENTS[spec.id].shape[1]
                                  if False else (640, 2560), "BF16")
            continue
        expected[source] = (_source_shape(source, tuple(spec.shape)), "BF16")
    for role, shape in _BANK_SHAPES.items():
        for layer in inventory.LAYERS:
            expected[_bank_source(layer, role)] = (shape, "BF16")
        expected[_mtp_bank_source(role)] = (shape, "BF16")
    for name in _PLE_SHARDS:
        expected[name] = ((2_500_012, 160), "BF16")
    expected["model.language_model.layers.1.ple.ple_embedding.layer_multipliers"] = ((3,), "I64")
    expected["model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets"] = ((16,), "I64")
    expected["model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes"] = ((16,), "I64")
    return expected


def _validate_source(reader: SafetensorsSource) -> dict[str, int]:
    expected = _expected_sources()
    actual = {name: (tuple(reader.describe(name).shape), reader.describe(name).dtype)
              for name in reader.weight_map}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        detail = missing[0] if missing else unexpected[0]
        raise ValueError(
            f"BF16 checkpoint allocation mismatch ({len(missing)} missing, "
            f"{len(unexpected)} unexpected): first {detail}"
        )
    for name, (shape, dtype) in expected.items():
        actual_shape, actual_dtype = actual[name]
        if actual_shape != tuple(shape) or actual_dtype != dtype:
            raise ValueError(
                f"{name}: source signature {(actual_shape, actual_dtype)} != {(tuple(shape), dtype)}"
            )
    counts: dict[str, int] = {}
    for _, dtype in actual.values():
        counts[dtype] = counts.get(dtype, 0) + 1
    return dict(sorted(counts.items()))


def _validate_config(source: Path) -> dict[str, object]:
    config = json.loads((source / "config.json").read_text())
    text = config.get("text_config")
    vision_config = config.get("vision_config")
    if not isinstance(text, dict) or not isinstance(vision_config, dict):
        raise ValueError("checkpoint config is missing the text or vision config")
    if config.get("quantization_config"):
        raise ValueError(
            "this route quantizes a plain BF16 checkpoint; a quantization_config is unexpected"
        )

    def check(label: str, values: dict, expected: dict) -> None:
        for key, value in expected.items():
            if values.get(key) != value:
                raise ValueError(f"{label}.{key}: expected {value!r}, got {values.get(key)!r}")

    check("config", config, {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "tie_word_embeddings": False,
    })
    check("text_config", text, {
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
    })
    check("vision_config", vision_config, {
        "depth": 27,
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_heads": 16,
        "out_hidden_size": 2560,
    })
    return {"layers": 48, "hidden_size": 2560, "experts": 512, "active_experts": 10,
            "max_context": 262144, "ple_rows": 320001536}


def _specs() -> tuple[TensorSpec, ...]:
    """The closed FP8+MTP inventory for a BF16-derived artifact."""

    replaced = set(_DENSE_FP8_IDS) | set(_PACKED_PARENTS)
    replaced |= {_mtp_bank_source(role) for role in _BANK_SHAPES}
    replaced |= {spec.id for spec in inventory.TENSOR_SPECS
                 if spec.id.endswith("_input_divisors")}
    for layer in inventory.LAYERS:
        for role in _BANK_SHAPES:
            replaced.add(_bank_parameter(layer, role))
    result: list[TensorSpec] = []
    for spec in inventory.TENSOR_SPECS:
        if spec.id in replaced:
            continue
        result.append(spec)
    for spec_id in _DENSE_FP8_IDS:
        shape = dense_fp8_parameters()[spec_id]
        result.append(TensorSpec(spec_id, shape, FP8_ROW, ROW_SCALE))
    for packed_id, parent in _PACKED_PARENTS.items():
        result.append(TensorSpec(packed_id, parent.shape, FP8_ROW, ROW_SCALE))
    for layer in inventory.LAYERS:
        for role, shape in _BANK_SHAPES.items():
            parameter = _bank_parameter(layer, role)
            result.append(TensorSpec(parameter, shape, NVFP4, EXPERT_NVFP4))
            result.append(TensorSpec(_divisor_parameter(parameter), (512,), "fp32",
                                     inventory.CONTIGUOUS))
    for role, shape in _BANK_SHAPES.items():
        parameter = _mtp_bank_parameter(role)
        result.append(TensorSpec(parameter, shape, NVFP4, EXPERT_NVFP4))
        result.append(TensorSpec(_divisor_parameter(parameter), (512,), "fp32",
                                 inventory.CONTIGUOUS))
    ids = [spec.id for spec in result]
    if len(ids) != len(set(ids)):
        duplicates = sorted({name for name in ids if ids.count(name) > 1})
        raise ValueError(f"duplicate object ids in the derived inventory: {duplicates[:4]}")
    return tuple(result)


def _fp8_payload(values: torch.Tensor, shape: tuple[int, int],
                 pool: ThreadPoolExecutor) -> tuple[bytes, dict]:
    n, k = shape
    rows = max(1, QUANTIZE_CHUNK_ELEMENTS // k)
    chunks = list(pool.map(quantize_bf16_rows, (values[r:r + rows] for r in range(0, n, rows))))
    codes = torch.cat([words.codes for words in chunks])
    scales = torch.cat([words.scales for words in chunks])
    exact = _float64(values.reshape(n, k))
    represented = _E4M3FN_VALUES[codes.numpy()] * _float64(scales)[:, None]
    error = represented - exact
    squared_value = float(np.square(exact).sum())
    return encode_fp8_row_scaled(codes, scales, shape), {
        "shape": list(shape),
        "relative_rms_error": math.sqrt(float(np.square(error).sum()) / squared_value)
        if squared_value else 0.0,
        "max_abs_error": float(np.abs(error).max()),
    }


def _quantize_expert(values: torch.Tensor) -> tuple:
    words = quantize_bf16_nvfp4(values)
    exact = _float64(values)
    error = represented_nvfp4(words) - exact
    return (
        words,
        float(np.square(error).sum()),
        float(np.square(exact).sum()),
        float(np.abs(error).max()),
    )


def _bank_payload(values: torch.Tensor, shape: tuple[int, int, int],
                  pool: ThreadPoolExecutor) -> tuple[bytes, torch.Tensor, dict]:
    experts, n, k = shape
    results = list(pool.map(_quantize_expert, (values[e] for e in range(experts))))
    codes = b"".join(words.codes.numpy().tobytes() for words, *_ in results)
    padding = bytes(align_up(len(codes), 256) - len(codes))
    scales = b"".join(
        swizzle_nvfp4_scales(words.scales, (n, k)).numpy().tobytes() for words, *_ in results
    )
    divisors = torch.stack([words.divisor for words, *_ in results])
    payload = codes + padding + scales + encode_direct(divisors, "fp32")
    if len(payload) != encoded_size(EXPERT_NVFP4, NVFP4, shape):
        raise ValueError("NVFP4 bank payload has the wrong size")
    squared_error = sum(result[1] for result in results)
    squared_value = sum(result[2] for result in results)
    return payload, divisors, {
        "shape": list(shape),
        "relative_rms_error": math.sqrt(squared_error / squared_value) if squared_value else 0.0,
        "max_abs_error": max(result[3] for result in results),
    }


def _read(reader: SafetensorsSource, name: str) -> torch.Tensor:
    """Read one source tensor in sub-2 GiB preadv chunks (NFS and read() bounds)."""

    info = reader.describe(name)
    elements = math.prod(info.shape)
    chunk = (1 << 31) // 4  # 512 MiB of BF16 words per read, under the 2 GiB preadv cap
    pieces = [reader.read_flat(name, begin, min(begin + chunk, elements))
              for begin in range(0, elements, chunk)] if elements > chunk \
        else [reader.read_flat(name)]
    return torch.cat(pieces).reshape(info.shape) if len(pieces) > 1 \
        else pieces[0].reshape(info.shape)


def _ple_payload(reader: SafetensorsSource) -> tuple[bytes, float, dict]:
    """Two streaming passes over the BF16 shards: the global BF16 scale, then the codes."""

    maximum = 0.0
    for name in _PLE_SHARDS:
        maximum = max(maximum, float(_read(reader, name).abs().max()))
    if maximum == 0.0:
        raise ValueError("the PLE table is all zeros")
    scale = torch.tensor(maximum / 448.0).to(torch.bfloat16).float()
    codes: list[torch.Tensor] = []
    exact: list[torch.Tensor] = []
    for name in _PLE_SHARDS:
        shard = _read(reader, name)
        codes.append((shard.float() / float(scale)).to(torch.float8_e4m3fn).view(torch.uint8))
        exact.append(shard.float().double())
    table = torch.cat(codes)
    if tuple(table.shape) != (320001536, 160):
        raise ValueError("PLE table payload has the wrong shape")
    exact_table = torch.cat(exact)
    error = (table.float().double() * float(scale) - exact_table).abs()
    report = {
        "shape": [320001536, 160],
        "weight_scale": float(scale),
        "relative_rms_error": float(error.square().sum().sqrt()
                                    / exact_table.square().sum().sqrt()),
        "max_abs_error": float(error.max()),
    }
    return encode_direct(table, inventory.FP8), float(scale), report


def encode_tensor_payload(value, spec, device):
    if spec.layout == inventory.CONTIGUOUS:
        return encode_direct(value.float() if spec.format == inventory.FP32 else value,
                             spec.format)
    quantized = quantize_matrix(value, spec.format, device=device)
    return encode_row_split(quantized.codes, quantized.scales, spec.format, spec.shape)


def _describe(objects: list[dict]) -> dict:
    """Components, bindings and Uses for the BF16-derived FP8+MTP inventory."""

    components = {
        "text": {"config": {
            "architectures": ["Qwen3_8FlashNextForCausalLM"],
            "model_type": "qwen3_8_flash_next_text", "hidden_size": 2560,
            "num_hidden_layers": 48, "vocab_size": 248320, "num_experts": 512,
        }, "resources": {}, "proposal": {"domain": "indexed", "rows": 147456}},
        "mtp": {"config": {"architectures": ["Qwen3_8FlashNextMTP"]}, "target": "text"},
        "vision": {"config": {"model_type": "qwen3_8_flash_next_vision"}, "target": "text",
                   "resources": {}},
    }
    bindings, uses = {}, []
    names = {o["id"] for o in objects}
    for obj in objects:
        name = obj["id"]
        if obj["kind"] == "resource":
            role = name.removeprefix("frontend/")
            component = ("vision" if role in ("preprocessor_config.json",
                                              "video_preprocessor_config.json") else "text")
            components[component]["resources"][role] = name
            continue
        bindings[name] = {"object": name}
        if obj["layout"] == EXPERT_NVFP4:
            auxiliary = _divisor_parameter(name)
            if auxiliary not in names:
                raise ValueError(f"{name}: missing weight divisors")
            uses.append({"parameter": name, "input": name + "/input",
                         "activation_policy": "AllowA16",
                         "auxiliaries": {"weight_divisor": {"object": auxiliary}}})
        elif len(obj["shape"]) >= 2 and obj["format"] != "fp8_e4m3fn":
            uses.append({"parameter": name, "input": name + "/input",
                         "activation_policy": "A16Only"})
    return {"components": components, "bindings": bindings, "uses": uses}


def convert(model_dir: str | Path, out_path: str | Path, *, device: str = "cpu") -> Path:
    source = Path(model_dir)
    output = Path(out_path)
    if output.name != OUTPUT_BASENAME:
        raise ValueError(f"output basename must be {OUTPUT_BASENAME!r}")
    started = time.perf_counter()
    resolved = pick_device(device)
    config_summary = _validate_config(source)
    resource_map = {name: (source / name.removeprefix("frontend/")).read_bytes()
                    for name in inventory.RESOURCE_SPECS}
    resource_specs = [ResourceSpec(name, len(data)) for name, data in resource_map.items()]
    tensor_specs = _specs()
    objects = plan_objects(list(resource_specs) + list(tensor_specs))
    description = _describe([o.to_json() for o in objects])
    ranking = Path(__file__).resolve().parents[3] / draft_head.DEFAULT_RANKING

    errors: dict[str, dict] = {}
    ple_payloads: dict[str, bytes] = {}
    with SafetensorsSource(source) as reader:
        dtype_counts = _validate_source(reader)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            # Quantize every routed bank once; the bank payload and its weight divisors share it.
            bank_payloads: dict[str, bytes] = {}
            bank_divisors: dict[str, torch.Tensor] = {}
            bank_jobs = [(_bank_parameter(layer, role), _bank_source(layer, role))
                         for layer in inventory.LAYERS for role in _BANK_SHAPES]
            bank_jobs += [(_mtp_bank_parameter(role), _mtp_bank_source(role))
                          for role in _BANK_SHAPES]
            for index, (parameter, source_name) in enumerate(bank_jobs, start=1):
                values = _read(reader, source_name)
                role = "gate_up" if parameter.endswith("gate_up") else "down"
                payload, divisors, report = _bank_payload(values, _BANK_SHAPES[role], pool)
                bank_payloads[parameter] = payload
                bank_divisors[parameter] = divisors
                errors[parameter] = report
                print(f"[bank {index}/{len(bank_jobs)}] {parameter}", flush=True)

            lm_head = _read(reader, "lm_head.weight")
            draft = draft_head.compute_shortlist(ranking, source, lm_head, n=147_456)

            def payload_for(spec):
                if isinstance(spec, ResourceSpec):
                    return resource_map[spec.id]
                spec_id = spec.id
                if spec_id in bank_payloads:
                    return bank_payloads[spec_id]
                if spec_id.endswith("_weight_divisors"):
                    parameter = spec_id.removesuffix("_weight_divisors")
                    return encode_direct(bank_divisors[parameter], "fp32")
                if spec_id == _PLE_TABLE:
                    return ple_payloads.pop(_PLE_TABLE)
                if spec_id == _PLE_SCALE:
                    payload, scale, report = _ple_payload(reader)
                    ple_payloads[_PLE_TABLE] = payload
                    errors["ple_table"] = report
                    return encode_direct(torch.tensor(scale).to(torch.bfloat16), "bf16")
                if spec_id in _PACKED_PARENTS:
                    parent = _PACKED_PARENTS[spec_id]
                    rows = parent.shape[0] // len(parent.parameters)
                    pieces = []
                    for name in parent.parameters:
                        piece = _read(reader, name)
                        if tuple(piece.shape) != (rows, parent.shape[1]):
                            raise ValueError(f"{name}: shared-expert source shape mismatch")
                        pieces.append(piece)
                    payload, report = _fp8_payload(torch.cat(pieces, dim=0), parent.shape, pool)
                    errors[spec_id] = report
                    return payload
                if spec_id in _DENSE_FP8_IDS:
                    payload, report = _fp8_payload(_read(reader, spec_id), tuple(spec.shape), pool)
                    errors[spec_id] = report
                    return payload
                if spec_id == _DRAFT_HEAD_IDS:
                    return encode_direct(
                        draft_head.materialize_draft_head_token_ids(draft), inventory.I32
                    )
                if spec_id == _DRAFT_HEAD:
                    selected = draft_head.materialize_draft_head(lm_head, draft)
                    return encode_tensor_payload(selected, spec, resolved)
                if spec_id.startswith("vision/"):
                    value = _read(reader, vision.SOURCES[spec_id]).reshape(tuple(spec.shape))
                    return encode_tensor_payload(value, spec, resolved)
                if spec_id in _CONVOLUTION_NAMES:
                    tensor = _read(reader, spec_id)
                    if tuple(tensor.shape) != (10240, 1, 4):
                        raise ValueError(f"{spec_id}: convolution source shape mismatch")
                    return encode_direct(tensor[:, 0, :].transpose(0, 1).contiguous(), "bf16")
                tensor = _read(reader, spec_id)
                if spec_id.endswith(("/A_log", "/dt_bias")):
                    return encode_direct(tensor.float(), "fp32")
                if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != tuple(spec.shape):
                    raise ValueError(f"{spec_id}: direct source signature mismatch")
                return encode_direct(tensor, "bf16")

            bindings = dict(description["bindings"])
            for packed_id, parent in _PACKED_PARENTS.items():
                del bindings[packed_id]
                elements = parent.shape[0] * parent.shape[1] // len(parent.parameters)
                for index, name in enumerate(parent.parameters):
                    bindings[name] = {"parts": [
                        {"object": packed_id, "range": [index * elements, (index + 1) * elements]}
                    ]}
            description["bindings"] = bindings

            output.parent.mkdir(parents=True, exist_ok=True)
            total = len(objects)
            with ArtifactWriter(
                output,
                list(resource_specs) + list(tensor_specs),
                **description,
                metadata={"name": inventory.MODEL_ID},
                provenance={"source": str(source.resolve()), "recipe": RECIPE_ID,
                            "source_kind": SOURCE_KIND},
            ) as writer:
                for index, spec in enumerate(list(resource_specs) + list(tensor_specs), start=1):
                    label = spec if isinstance(spec, str) else spec.id
                    writer.write_object(label, payload_for(spec))
                    print(f"[{index}/{total}] {label}", flush=True)

    with Artifact(output) as artifact:
        file_bytes = artifact.file_bytes
    elapsed = time.perf_counter() - started
    worst = max(errors.items(), key=lambda item: item[1].get("relative_rms_error", 0.0))
    report = {
        "recipe_id": RECIPE_ID,
        "source": {"path": str(source.resolve()), "kind": SOURCE_KIND},
        "output": str(output.resolve()),
        "config_summary": config_summary,
        "source_dtype_counts": dtype_counts,
        "objects": len(objects),
        "file_bytes": file_bytes,
        "worst_relative_rms_error": {"object": worst[0], **worst[1]},
        "errors": errors,
        "elapsed_seconds": elapsed,
        "conversion_device": str(resolved),
        "divisor_policy": "AllowA16 weight_divisor (nvfp4_maxabs); no activation calibration",
    }
    report_path = Path(str(output) + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"complete: {file_bytes} bytes in {elapsed:.1f}s", flush=True)
    return report_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--validate-only", action="store_true",
                        help="check the source allocation and exit without writing")
    args = parser.parse_args(argv)
    if args.validate_only:
        with SafetensorsSource(args.model) as reader:
            counts = _validate_source(reader)
        print("source allocation ok:", counts)
        print("inventory objects:", len(_specs()))
        return
    if args.out.name != OUTPUT_BASENAME:
        raise SystemExit(f"output basename must be {OUTPUT_BASENAME!r}")
    convert(args.model, args.out, device=args.device)


if __name__ == "__main__":
    main()
