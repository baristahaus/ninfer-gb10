"""Add the DFlash2 draft component to an existing Flash-Next 125B artifact.

Artifact-to-artifact: every object of the source is copied byte for byte; the
DFlash2 draft's tensors are appended from the prepared companion
(`dflash2_prepare.py` output) as contiguous BF16, with the component record
(the companion's config), the A16 uses for the input-carrying parameters, and
the recipe/provenance bumped. The runtime's DFlash2 execution is a separate
work item; until it lands the component binds and materializes inertly.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from math import prod
from pathlib import Path

import torch
from safetensors.torch import load_file
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceObject, ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.artifact.layouts import row_split_geometry
from tools.convert.quantization.groupwise import quantize_matrix

OUTPUT_BASENAME = "qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp_dflash2.ninfer"
RECIPE_ID = "qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp_dflash2-v1"
DRAFT_LAYERS = 5
CONTIGUOUS = "contiguous_le_v1"
Q8 = "q8_g32_fp16"

# (parameter name, companion tensor, shape, inputs); the conv base kernels and the
# codebooks carry no mathematical use and get no Use record.
DRAFT_PARAMETERS: list[tuple[str, str, tuple[int, ...], tuple[str, ...]]] = [
    ("dflash2/feature_projection", "fc.weight", (2560, 12800), ("dflash2/feature_projection/input",)),
    ("dflash2/context_norm", "hidden_norm.weight", (2560,), ()),
    ("dflash2/final_norm", "norm.weight", (2560,), ()),
]
for _layer in range(DRAFT_LAYERS):
    _p = f"dflash2/layers/{_layer}/"
    _s = f"layers.{_layer}."
    DRAFT_PARAMETERS += [
        (_p + "input_norm", _s + "input_layernorm.weight", (2560,), ()),
        (_p + "post_attention_norm", _s + "post_attention_layernorm.weight", (2560,), ()),
        (_p + "attention/query_key_value", _s + "self_attn.query_key_value.weight",
         (6144, 2560), (_p + "attention/query_key_value/input",)),
        (_p + "attention/context_key", _s + "self_attn.query_key_value.weight",
         (1024, 2560), (_p + "attention/context_key/input",)),
        (_p + "attention/context_value", _s + "self_attn.query_key_value.weight",
         (1024, 2560), (_p + "attention/context_value/input",)),
        (_p + "attention/query_norm", _s + "self_attn.q_norm.weight", (128,), ()),
        (_p + "attention/key_norm", _s + "self_attn.k_norm.weight", (128,), ()),
        (_p + "attention/output", _s + "self_attn.o_proj.weight", (2560, 4096),
         (_p + "attention/output/input",)),
        (_p + "attention_conv/base_kernel", _s + "attention_conv.base_kernel",
         (2, 2, 2560), ()),
        (_p + "attention_conv/kernel_projection",
         _s + "attention_conv.kernel_projection.weight", (640, 2560),
         (_p + "attention_conv/kernel_projection/input",)),
        (_p + "mlp_conv/base_kernel", _s + "mlp_conv.base_kernel", (2, 2, 2560), ()),
        (_p + "mlp_conv/kernel_projection", _s + "mlp_conv.kernel_projection.weight",
         (640, 2560), (_p + "mlp_conv/kernel_projection/input",)),
        (_p + "mlp/gate_up", _s + "mlp.gate_up_proj.weight", (15360, 2560),
         (_p + "mlp/gate_up/input",)),
        (_p + "mlp/down", _s + "mlp.down_proj.weight", (2560, 7680),
         (_p + "mlp/down/input",)),
    ]
DRAFT_PARAMETERS += [
    ("dflash2/candidate_selector/hidden_projection",
     "candidate_selector.hidden_projection.weight", (256, 2560),
     ("dflash2/candidate_selector/hidden_projection/input",)),
    ("dflash2/candidate_selector/predecessor_codebook",
     "candidate_selector.predecessor_codebook", (248320, 256), ()),
    ("dflash2/candidate_selector/successor_codebook",
     "candidate_selector.successor_codebook", (248320, 256), ()),
]


def add(source_path: str | Path, companion_path: Path, out_path: str | Path) -> dict:
    started = time.perf_counter()
    companion = load_file(companion_path / "model.safetensors")
    config = json.loads((companion_path / "config.json").read_text())
    draft_payloads: dict[str, bytes] = {}
    quantized: dict[str, str] = {}
    for name, tensor_name, shape, _inputs in DRAFT_PARAMETERS:
        value = companion[tensor_name]
        if name.endswith("attention/context_key"):
            value = value[4096:5120]
        elif name.endswith("attention/context_value"):
            value = value[5120:6144]
        if name.endswith(("/attention/output", "/mlp/down")):
            # The dynamic-conv add op consumes these projections as Q8_G32_FP16; the
            # physical payload is the row-split plane layout.
            encoded = quantize_matrix(value.contiguous(), Q8, device="cpu")
            geometry = row_split_geometry(Q8, shape)
            codes = encoded.codes.cpu().view(torch.uint8).numpy().tobytes()
            scales = encoded.scales.cpu().view(torch.float16).numpy().tobytes()
            if len(codes) != geometry.base_bytes or len(scales) != geometry.scale_bytes:
                raise SystemExit(f"{name}: Q8 planes disagree with the row-split geometry")
            payload = codes + bytes(geometry.scale_offset - len(codes)) + scales
            if len(payload) != geometry.payload_bytes:
                raise SystemExit(f"{name}: Q8 payload disagrees with the row-split geometry")
            draft_payloads[name] = payload
            quantized[name] = Q8
            continue
        payload = value.contiguous().view(torch.uint8).numpy().tobytes()
        if len(payload) != prod(shape) * 2:
            raise SystemExit(
                f"{name}: payload {len(payload)} bytes disagrees with the declared shape {shape}"
            )
        draft_payloads[name] = payload
    output = Path(out_path)
    with Artifact(source_path) as artifact:
        directory = artifact.directory
        components = dict(directory.components)
        components["dflash2"] = {"config": config, "target": "text"}
        bindings = dict(directory.bindings)
        existing_uses = [dict(use) for use in directory.uses]
        steps: list[tuple[str, bool]] = []
        specs: list[TensorSpec | ResourceSpec] = []
        for obj in directory.objects:
            if isinstance(obj, ResourceObject):
                specs.append(ResourceSpec(obj.id, obj.bytes, obj.encoding))
            else:
                specs.append(TensorSpec(obj.id, tuple(obj.shape), obj.format, obj.layout))
            steps.append((obj.id, False))
        for name, tensor_name, shape, _inputs in DRAFT_PARAMETERS:
            if name in quantized:
                specs.append(TensorSpec(name, shape, quantized[name], "row_split_k128_v1"))
            else:
                specs.append(TensorSpec(name, shape, "bf16", CONTIGUOUS))
            steps.append((name, True))
        bindings = {**bindings, **{
            name: {"object": name} for name, _, _, _ in DRAFT_PARAMETERS
        }}
        uses = existing_uses + [
            {"parameter": name, "input": input_name, "activation_policy": "A16Only"}
            for name, _, _, inputs in DRAFT_PARAMETERS
            for input_name in inputs
        ]
        provenance = dict(directory.provenance)
        provenance["dflash2_companion"] = str(Path(companion_path).resolve())
        output.parent.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool, ArtifactWriter(
            output, specs, components=components, bindings=bindings, uses=uses,
            metadata=directory.metadata, provenance=provenance,
        ) as writer:
            for index, (object_id, is_draft) in enumerate(steps, start=1):
                if not is_draft:
                    writer.write_object(object_id, artifact.iter_object(object_id))
                    continue
                writer.write_object(object_id, draft_payloads[object_id])
                print(f"[{index}/{len(steps)}] {object_id}", flush=True)
        source_id = artifact.artifact_id.hex()
        source_payload = artifact.payload_bytes
    with Artifact(output) as written:
        written_payload = written.payload_bytes
        written_file_bytes = written.file_bytes
    report = {
        "recipe_id": RECIPE_ID,
        "source": {"path": str(Path(source_path).resolve()), "artifact_id": source_id,
                   "payload_bytes": source_payload},
        "companion": {"path": str(Path(companion_path).resolve()),
                      "config": str((companion_path / "config.json").resolve())},
        "output": str(output.resolve()),
        "payload_bytes": written_payload,
        "file_bytes": written_file_bytes,
        "elapsed_seconds": time.perf_counter() - started,
    }
    Path(str(output) + ".conversion.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="the existing FP8+MTP artifact")
    parser.add_argument("companion", type=Path, help="the prepared DFlash2 companion")
    parser.add_argument("out", type=Path, help="the output artifact path")
    args = parser.parse_args(argv)
    if args.out.name != OUTPUT_BASENAME:
        raise SystemExit(f"output basename must be {OUTPUT_BASENAME!r}")
    report = add(args.source, args.companion, args.out)
    print(
        f"dflash2 artifact written: payload {report['payload_bytes']} bytes in "
        f"{report['elapsed_seconds']:.1f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
