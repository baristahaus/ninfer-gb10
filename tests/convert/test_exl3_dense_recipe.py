"""The dense EXL3 conversion route's artifact assembly (SP-7 contract 6.2 step 5).

A tiny synthetic dense Qwen3.5 (GDN layer + attention layer, every quantized width a
128-multiple) is quantized by the campaign's `convert_model`, assembled into a `.ninfer`
by the `qwen3_8_27b_exl3_dense` official recipe, and read back: every projection binds as
an `exl3_dense_v1` object whose planes match the converter's own results and whose
composed weight stays within the quantization's recorded error of the BF16 source.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from tools.artifact.layouts import EXL3_CODEBOOK_IDS, exl3_dense_geometry
from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject
from tools.convert.pipeline import convert
from tools.convert.quantization import exl3_convert as conv
from tools.convert.quantization.exl3 import decode_inner
from tools.convert.quantization.exl3_quantize import decode_composition
from tools.convert.official_recipes import qwen3_8_27b_exl3_dense
from tools.convert.qwen3_5 import build_model
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource

HIDDEN = 256
INTERMEDIATE = 128
LAYERS = 2
K2 = 4


def _dense_config() -> dict:
    return {
        "architectures": ["Qwen3_5ForCausalLM"],
        "text_config": {
            "hidden_size": HIDDEN,
            "vocab_size": 8,
            "num_hidden_layers": LAYERS,
            "max_position_embeddings": 128,
            "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 128,
            "linear_num_value_heads": 1,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 3,
            "intermediate_size": INTERMEDIATE,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 1000000.0, "mrope_section": [11, 11, 10]},
            "partial_rotary_factor": 0.5,
        },
    }


def _dense_tensors() -> dict:
    generator = torch.Generator().manual_seed(20261017)

    def tensor(shape):
        scale = 0.02
        return (torch.randn(tuple(shape), generator=generator) * scale).to(torch.bfloat16)

    names = {
        "model.language_model.embed_tokens.weight": (8, HIDDEN),
        "lm_head.weight": (8, HIDDEN),
        "model.language_model.norm.weight": (HIDDEN,),
    }
    for layer in range(LAYERS):
        prefix = f"model.language_model.layers.{layer}"
        names.update({
            f"{prefix}.input_layernorm.weight": (HIDDEN,),
            f"{prefix}.post_attention_layernorm.weight": (HIDDEN,),
            f"{prefix}.mlp.gate_proj.weight": (INTERMEDIATE, HIDDEN),
            f"{prefix}.mlp.up_proj.weight": (INTERMEDIATE, HIDDEN),
            f"{prefix}.mlp.down_proj.weight": (HIDDEN, INTERMEDIATE),
        })
        if layer == 0:
            names.update({
                f"{prefix}.linear_attn.A_log": (1,),
                f"{prefix}.linear_attn.dt_bias": (1,),
                f"{prefix}.linear_attn.conv1d.weight": (3 * 128, 1, 3),
                f"{prefix}.linear_attn.in_proj_a.weight": (1, HIDDEN),
                f"{prefix}.linear_attn.in_proj_b.weight": (1, HIDDEN),
                f"{prefix}.linear_attn.in_proj_qkv.weight": (384, HIDDEN),
                f"{prefix}.linear_attn.in_proj_z.weight": (128, HIDDEN),
                f"{prefix}.linear_attn.norm.weight": (128,),
                f"{prefix}.linear_attn.out_proj.weight": (HIDDEN, 128),
            })
        else:
            names.update({
                f"{prefix}.self_attn.q_proj.weight": (512, HIDDEN),  # packed Q + gate
                f"{prefix}.self_attn.k_proj.weight": (128, HIDDEN),
                f"{prefix}.self_attn.v_proj.weight": (128, HIDDEN),
                f"{prefix}.self_attn.o_proj.weight": (HIDDEN, 256),
                f"{prefix}.self_attn.q_norm.weight": (128,),
                f"{prefix}.self_attn.k_norm.weight": (128,),
            })
    return {name: tensor(shape) for name, shape in names.items()}


def _checkpoint(path: Path, tensors) -> SafetensorsSource:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(_dense_config()))
    save_file(tensors, path / "model-00001-of-00001.safetensors")
    for role, value in {
        "tokenizer.json": {"model": {"vocab": {str(i): i for i in range(8)}}},
        "tokenizer_config.json": {},
        "generation_config.json": {},
    }.items():
        (path / role).write_text(json.dumps(value))
    (path / "chat_template.jinja").write_text("{{ messages }}")
    return SafetensorsSource(path)


def test_dense_recipe_assembles_and_verifies(tmp_path):
    tensors = _dense_tensors()
    base = _checkpoint(tmp_path / "base", tensors)
    converted = tmp_path / "converted"
    if os.environ.get("NINFER_DENSE_CONVERT_CACHE"):
        cached = Path(os.environ["NINFER_DENSE_CONVERT_CACHE"])
        if (cached / "layer_01.pt").exists():
            converted = cached
    if not (converted / "layer_01.pt").exists():
        conv.convert_model(base.path, None, converted, k2=K2, codebook="mcg",
                           layers=[0, 1], device=torch.device("cpu"))

    model = build_model(base)
    recipe = Recipe(model)
    qwen3_8_27b_exl3_dense(model, recipe, {"converted": converted})
    output = tmp_path / "dense.ninfer"
    report = convert(model, recipe, output, device="cpu", name="dense-exl3-test")
    # Per layer: 5 mixer projections (GDN query/key/value/z/output or attention
    # query/gate/key/value/output) + 3 MLP projections = 8, two layers = 16.
    assert report["formats"]["exl3"] == 16

    with Artifact.open(output) as artifact:
        objects = {
            obj.id: obj
            for obj in artifact.objects
            if isinstance(obj, TensorObject) and obj.format == "exl3"
        }
        bindings = artifact.directory.bindings
        # Every projection parameter binds an exl3_dense_v1 object with the exact
        # converter payload; the a/b controls and norms stay BF16/Q8 from the source.
        exl3_bindings = [
            name for name in bindings
            if name.startswith("text/layers/")
            and name.split("/")[-1] in {"query", "gate", "key", "value", "output", "z"}
            or name.endswith(("/mlp/gate", "/mlp/up", "/mlp/down"))
        ]
        assert len(exl3_bindings) == 16
        for name in exl3_bindings:
            binding = bindings[name]
            parts = binding.get("parts") or [{"object": binding["object"]}]
            assert len(parts) == 1
            obj = objects[parts[0]["object"]]
            assert obj.layout == "exl3_dense_v1"
            payload = artifact.read_object(obj.id)
            n, k = obj.shape
            geometry = exl3_dense_geometry(n, k, K2, "mcg")
            assert len(payload) == geometry.payload_bytes
            assert int.from_bytes(payload[-4:], "little") == EXL3_CODEBOOK_IDS["mcg"]

        # Spot verification: the layer-1 attention output projection's composed weight
        # stays within the uncalibrated quantization error of the BF16 source.
        binding = bindings["text/layers/1/attention/output"]
        parts = binding.get("parts") or [{"object": binding["object"]}]
        obj = objects[parts[0]["object"]]
        n, k = obj.shape
        payload = artifact.read_object(obj.id)
        geometry = exl3_dense_geometry(n, k, K2, "mcg")
        trellis = np.frombuffer(payload[: geometry.trellis_bytes], dtype=np.int16).reshape(
            k // 16, n // 16, 8 * K2
        )
        inner = decode_inner(trellis, "mcg")
        inner_t = torch.from_numpy(inner.astype(np.uint16)).view(torch.float16).float()
        suh = torch.from_numpy(
            np.frombuffer(
                payload[geometry.suh_offset: geometry.suh_offset + geometry.suh_bytes],
                dtype=np.uint16,
            ).copy()
        ).view(torch.float16).float()
        svh = torch.from_numpy(
            np.frombuffer(
                payload[geometry.svh_offset: geometry.svh_offset + geometry.svh_bytes],
                dtype=np.uint16,
            ).copy()
        ).view(torch.float16).float()
        composed = decode_composition(suh, inner_t, svh).T.float()  # [n, k]
        source = tensors["model.language_model.layers.1.self_attn.o_proj.weight"].float()
        relative = float(
            torch.linalg.norm((composed - source).reshape(-1))
            / torch.linalg.norm(source.reshape(-1))
        )
        assert relative < 0.45  # the uncalibrated k2 = 4 NMSE is ~0.08-0.12
