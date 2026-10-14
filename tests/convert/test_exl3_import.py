"""The EXL3 import route (SP-7 contract section 6.1): reader, geometry, banks, artifact.

The synthetic checkpoint uses small 128-multiple dimensions and per-bank codebooks
covering 3inst, mcg and mul1 with mixed integer and half-integer bitrates, plus
legacy packed `su`/`sv` sign fields. The emitted banks are verified against the
source tensors with the qualified exact codec oracle.
"""

from __future__ import annotations

import json
from math import prod
from pathlib import Path
import struct

import numpy as np
import pytest
from safetensors.torch import save_file
import torch

from tools.artifact.layouts import (
    EXL3_CODEBOOK_IDS,
    exl3_bank_geometry,
    validate_exl3_rate,
)
from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject, plan_objects
from tools.convert.methods import import_exl3_bank
from tools.convert.official_recipes import qwen3_6_35b_a3b_exl3
from tools.convert.pipeline import convert
from tools.convert.qwen3_5 import build_model
from tools.convert.recipe import Recipe
from tools.convert.sources.exllamav3 import (
    MCG_MARKER,
    MUL1_MARKER,
    PINNED_EXL3_VERSION,
    check_quantization_config,
    exl3_expert_source,
    marker_as_int32,
    read_exl3_linear,
)
from tools.convert.sources.safetensors import SafetensorsSource

HIDDEN = 256
INTERMEDIATE = 128
EXPERTS = 3
LAYERS = 2
# Per (layer, role): the experts' K values and codebook, covering every codebook
# and mixed integer/half-integer widths inside single banks.
BANKS = {
    (0, "gate"): ((2, 3, 3), "mcg"),
    (0, "up"): ((2, 2, 2), "3inst"),
    (0, "down"): ((4, 4, 3.5), "mul1"),
    (1, "gate"): ((5, 2, 2), "mcg"),
    (1, "up"): ((1.5, 3.5, 1), "mul1"),
    (1, "down"): ((6, 8, 2), "3inst"),
}
PACKED_SIGN_EXPERTS = {(0, "gate", 1), (1, "down", 0)}


def _base_tensors():
    """Every non-routed tensor the synthetic MoE binding reads, in BF16."""

    generator = torch.Generator().manual_seed(20261015)

    def tensor(shape):
        return torch.randn(tuple(shape), generator=generator).to(torch.bfloat16)

    names = {
        "model.language_model.embed_tokens.weight": (8, 256),
        "lm_head.weight": (8, 256),
        "model.language_model.norm.weight": (256,),
    }
    for layer in range(LAYERS):
        prefix = f"model.language_model.layers.{layer}"
        names.update({
            f"{prefix}.input_layernorm.weight": (256,),
            f"{prefix}.post_attention_layernorm.weight": (256,),
            f"{prefix}.mlp.gate.weight": (EXPERTS, 256),
            f"{prefix}.mlp.shared_expert_gate.weight": (1, 256),
            f"{prefix}.mlp.experts.gate_up_proj": (EXPERTS, 2 * INTERMEDIATE, 256),
            f"{prefix}.mlp.experts.down_proj": (EXPERTS, 256, INTERMEDIATE),
            f"{prefix}.mlp.shared_expert.gate_proj.weight": (64, 256),
            f"{prefix}.mlp.shared_expert.up_proj.weight": (64, 256),
            f"{prefix}.mlp.shared_expert.down_proj.weight": (256, 64),
        })
        if layer == 0:
            names.update({
                f"{prefix}.linear_attn.A_log": (2,),
                f"{prefix}.linear_attn.dt_bias": (2,),
                f"{prefix}.linear_attn.conv1d.weight": (256, 1, 3),
                f"{prefix}.linear_attn.in_proj_a.weight": (2, 256),
                f"{prefix}.linear_attn.in_proj_b.weight": (2, 256),
                f"{prefix}.linear_attn.in_proj_qkv.weight": (256, 256),
                f"{prefix}.linear_attn.in_proj_z.weight": (128, 256),
                f"{prefix}.linear_attn.norm.weight": (64,),
                f"{prefix}.linear_attn.out_proj.weight": (256, 128),
            })
        else:
            names.update({
                f"{prefix}.self_attn.q_proj.weight": (512, 256),
                f"{prefix}.self_attn.k_proj.weight": (128, 256),
                f"{prefix}.self_attn.v_proj.weight": (128, 256),
                f"{prefix}.self_attn.o_proj.weight": (256, 256),
                f"{prefix}.self_attn.q_norm.weight": (64,),
                f"{prefix}.self_attn.k_norm.weight": (64,),
            })
    return {name: tensor(shape) for name, shape in names.items()}


def _config(*, quantized=False):
    text = {
        "hidden_size": HIDDEN,
        "vocab_size": 8,
        "num_hidden_layers": LAYERS,
        "max_position_embeddings": 128,
        "layer_types": ["linear_attention", "full_attention"],
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "rope_parameters": {"partial_rotary_factor": 0.5, "mrope_section": [6, 5, 5]},
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 64,
        "linear_num_value_heads": 2,
        "linear_value_head_dim": 64,
        "linear_conv_kernel_dim": 3,
        "num_experts": EXPERTS,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": INTERMEDIATE,
        "shared_expert_intermediate_size": 64,
    }
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": text,
    }
    if quantized:
        config["quantization_config"] = {
            "quant_method": "exl3",
            "version": PINNED_EXL3_VERSION,
        }
    return config


def _role_dims(role):
    if role in ("gate", "up"):
        return HIDDEN, INTERMEDIATE  # k, n
    return INTERMEDIATE, HIDDEN


def _exl3_tensors():
    """The synthetic EXL3 tensors: trellis, suh/svh (some legacy packed), markers."""

    generator = torch.Generator().manual_seed(20261014)
    tensors = {}
    for (layer, role), (rates, codebook) in BANKS.items():
        k, n = _role_dims(role)
        for expert, rate in enumerate(rates):
            name = f"model.layers.{layer}.mlp.experts.{expert}.{role}_proj"
            words = 8 * round(2 * rate)
            trellis = torch.randint(
                -(1 << 15), 1 << 15, (k // 16, n // 16, words),
                generator=generator, dtype=torch.int16,
            )
            tensors[f"{name}.trellis"] = trellis
            if (layer, role, expert) in PACKED_SIGN_EXPERTS:
                tensors[f"{name}.su"] = torch.randint(
                    -(1 << 15), 1 << 15, ((k + 15) // 16,), generator=generator,
                    dtype=torch.int16,
                )
                tensors[f"{name}.sv"] = torch.randint(
                    -(1 << 15), 1 << 15, ((n + 15) // 16,), generator=generator,
                    dtype=torch.int16,
                )
            else:
                tensors[f"{name}.suh"] = torch.randn(
                    k, generator=generator
                ).to(torch.float16)
                tensors[f"{name}.svh"] = torch.randn(
                    n, generator=generator
                ).to(torch.float16)
            if codebook == "mcg":
                tensors[f"{name}.mcg"] = torch.tensor(
                    marker_as_int32(MCG_MARKER), dtype=torch.int32
                )
            elif codebook == "mul1":
                tensors[f"{name}.mul1"] = torch.tensor(
                    marker_as_int32(MUL1_MARKER), dtype=torch.int32
                )
    return tensors


def _checkpoint(path: Path, config, tensors):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config))
    save_file(tensors, path / "model.safetensors")
    for role, value in {
        "tokenizer.json": {"model": {"vocab": {str(i): i for i in range(6)}}},
        "tokenizer_config.json": {},
        "generation_config.json": {},
    }.items():
        (path / role).write_text(json.dumps(value))
    (path / "chat_template.jinja").write_text("{{ messages }}")
    return SafetensorsSource(path)


def test_geometry_packs_extents_and_planes_per_contract():
    rates = (2, 3, 7)
    geometry = exl3_bank_geometry(128, 256, rates, "mul1")
    assert geometry.experts == 3
    extents = [256 * ((128 * 256 * rate // 16 + 255) // 256) for rate in rates]
    assert list(geometry.trellis_extents) == extents
    assert geometry.trellis_offset == 256
    assert geometry.trellis_bytes == sum(extents)
    assert geometry.suh_bytes == 3 * 256 * 2
    assert geometry.svh_bytes == 3 * 128 * 2
    assert geometry.codebook_offset == geometry.svh_offset + geometry.svh_bytes
    assert geometry.payload_bytes == geometry.codebook_offset + 4
    entries = geometry.directory_entries()
    assert [entry[0] for entry in entries] == [
        geometry.trellis_offset + sum(extents[:i]) for i in range(3)
    ]
    assert [entry[2] for entry in entries] == list(rates)
    with pytest.raises(ValueError, match="divisible by 128"):
        exl3_bank_geometry(120, 256, rates, "mul1")
    with pytest.raises(ValueError, match="mul1"):
        validate_exl3_rate(3, "mcg")
    with pytest.raises(ValueError, match="2..16"):
        validate_exl3_rate(18, "mul1")


def test_reader_validates_version_markers_shapes_and_packed_signs(tmp_path):
    tensors = _exl3_tensors()
    with _checkpoint(tmp_path / "exl3", _config(quantized=True), tensors) as store:
        linear = read_exl3_linear(
            store, "model.layers.0.mlp.experts.1.gate_proj", expected=(HIDDEN, INTERMEDIATE)
        )
        assert linear.k2 == 6 and linear.codebook == "mcg"
        assert linear.k == HIDDEN and linear.n == INTERMEDIATE
        assert torch.equal(
            linear.trellis, tensors["model.layers.0.mlp.experts.1.gate_proj.trellis"]
        )
        # The legacy packed `su` expert expands to +-1 FP16 signs.
        packed = tensors["model.layers.0.mlp.experts.1.gate_proj.su"]
        expected = torch.where(
            (packed.to(torch.int32) & 0xFFFF) == 0, 1.0, -1.0
        ).repeat_interleave(16)[:HIDDEN]
        assert torch.equal(linear.suh.float(), expected)

        bad_version = _config(quantized=True)
        bad_version["quantization_config"]["version"] = "1.5.0"
        with pytest.raises(ValueError, match="pinned"):
            check_quantization_config(bad_version)
        with pytest.raises(ValueError, match="quantization_config"):
            check_quantization_config(_config())
        marker = dict(tensors)
        marker["model.layers.0.mlp.experts.1.gate_proj.mcg"] = torch.tensor(
            1234, dtype=torch.int32
        )
        with _checkpoint(tmp_path / "bad-marker", _config(quantized=True), marker) as bad:
            with pytest.raises(ValueError, match="multiplier"):
                read_exl3_linear(bad, "model.layers.0.mlp.experts.1.gate_proj")
        half = dict(tensors)
        del half["model.layers.0.mlp.experts.2.gate_proj.mcg"]
        half["model.layers.0.mlp.experts.2.gate_proj.mul1"] = torch.tensor(
            marker_as_int32(MCG_MARKER), dtype=torch.int32
        )
        with _checkpoint(tmp_path / "wrong-marker", _config(quantized=True), half) as bad:
            with pytest.raises(ValueError, match="must equal its multiplier"):
                read_exl3_linear(bad, "model.layers.0.mlp.experts.2.gate_proj")


def _prepare_recipe(tmp_path):
    """build_model + the official EXL3 recipe, prepared with complete synthetic weights."""

    tensors = _exl3_tensors()
    exl3 = _checkpoint(tmp_path / "exl3", _config(quantized=True), tensors)
    base = _checkpoint(tmp_path / "base", _config(), _base_tensors())
    model = build_model(base)
    recipe = Recipe(model)
    qwen3_6_35b_a3b_exl3(model, recipe, {"exl3": exl3})
    return model, recipe.prepare(device="cpu"), exl3, tensors


def test_official_recipe_prepares_one_bank_per_layer_and_projection(tmp_path):
    model, prepared, exl3, tensors = _prepare_recipe(tmp_path)
    banks = {}
    for job in prepared.weights:
        if job.spec.format == "exl3":
            banks.setdefault(job.spec.layout, []).append(job)
    assert list(banks) == ["exl3_expert_bank"]
    exl3_jobs = banks["exl3_expert_bank"]
    assert len(exl3_jobs) == LAYERS * 3
    by_layer_role = {}
    for job in exl3_jobs:
        assert len(job.parameters) == EXPERTS
        assert job.method_name == "import_exl3_bank"
        layer = int(job.parameters[0].split("/")[2])
        role = job.parameters[0].rsplit("/", 1)[1]
        by_layer_role[(layer, role)] = job
        k, n = _role_dims(role)
        assert job.spec.shape == (EXPERTS * n, k)
        rates = BANKS[(layer, role)][0]
        geometry = exl3_bank_geometry(
            n, k, [round(2 * rate) for rate in rates], BANKS[(layer, role)][1]
        )
        assert job.spec.bytes == geometry.payload_bytes
    # Non-routed projections keep the groupwise recipe's Q8 assignment.
    formats = {}
    for job in prepared.weights:
        for parameter in job.parameters:
            formats[parameter] = job.spec.format
    assert formats["text/layers/1/attention/query"] == "q8_g32_fp16"
    assert formats["text/layers/0/gdn/query"] == "q8_g32_fp16"
    assert formats["text/layers/1/moe/shared/gate"] == "q8_g32_fp16"
    assert "text/token_embedding" in formats
    # Every routed expert's binding fragment addresses its own bank rows.
    for name, binding in prepared.bindings.items():
        if "/moe/experts/" not in name:
            continue
        expert = int(name.split("/")[5])
        role = name.rsplit("/", 1)[1]
        layer = int(name.split("/")[2])
        job = by_layer_role[(layer, role)]
        k, n = _role_dims(role)
        assert binding == {
            "parts": [{
                "object": job.spec.id,
                "range": [expert * n * k, (expert + 1) * n * k],
            }]
        }
    # The banks' specs plan into objects at their declared bytes.
    specs = [job.spec for job in exl3_jobs]
    objects = {obj.id: obj for obj in plan_objects(specs)}
    for job in exl3_jobs:
        assert objects[job.spec.id].bytes == job.spec.bytes


def test_official_recipe_imports_emits_and_reads_back_verified_banks(tmp_path):
    tensors = _exl3_tensors()
    exl3 = _checkpoint(tmp_path / "exl3", _config(quantized=True), tensors)
    base = _checkpoint(tmp_path / "base", _config(), _base_tensors())
    with exl3, base:
        model = build_model(base)
        recipe = Recipe(model)
        qwen3_6_35b_a3b_exl3(model, recipe, {"exl3": exl3})
        output = tmp_path / "imported.ninfer"
        report = convert(
            model, recipe, output, device="cpu", name="exl3-import-test"
        )
        assert report["formats"]["exl3"] == LAYERS * 3
        with Artifact.open(output) as artifact:
            objects = {
                obj.id: obj for obj in artifact.objects
                if isinstance(obj, TensorObject) and obj.format == "exl3"
            }
            assert len(objects) == LAYERS * 3
            bindings = artifact.directory.bindings
            from tools.convert.quantization.exl3 import decode_tiles

            for (layer, role), (rates, codebook) in BANKS.items():
                k, n = _role_dims(role)
                bank = objects[bindings[
                    f"text/layers/{layer}/moe/experts/0/{role}"
                ]["parts"][0]["object"]]
                assert bank.shape == (EXPERTS * n, k)
                payload = artifact.read_object(bank.id)
                geometry = exl3_bank_geometry(
                    n, k, [round(2 * rate) for rate in rates], codebook
                )
                assert len(payload) == geometry.payload_bytes
                assert int.from_bytes(payload[-4:], "little") == EXL3_CODEBOOK_IDS[codebook]
                for expert, rate in enumerate(rates):
                    offset, extent, k2 = geometry.directory_entries()[expert]
                    entry = struct.unpack_from("<QIBBBB", payload, expert * 16)
                    assert entry[:3] == (offset, extent, k2)
                    source = tensors[
                        f"model.layers.{layer}.mlp.experts.{expert}.{role}_proj.trellis"
                    ]
                    emitted = torch.frombuffer(
                        bytearray(payload[offset : offset + extent]),
                        dtype=torch.int16,
                    ).reshape(-1)[: source.numel()].reshape(source.shape)
                    assert torch.equal(emitted, source)
                    assert np.array_equal(
                        decode_tiles(
                            emitted.reshape(k // 16, n // 16, 8 * k2), k2, codebook
                        ),
                        decode_tiles(source, k2, codebook),
                    )
                    suh = torch.frombuffer(
                        bytearray(
                            payload[
                                geometry.suh_offset + expert * k * 2 :
                                geometry.suh_offset + (expert + 1) * k * 2
                            ]
                        ),
                        dtype=torch.float16,
                    )
                    svh = torch.frombuffer(
                        bytearray(
                            payload[
                                geometry.svh_offset + expert * n * 2 :
                                geometry.svh_offset + (expert + 1) * n * 2
                            ]
                        ),
                        dtype=torch.float16,
                    )
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{role}_proj"
                    if (layer, role, expert) in PACKED_SIGN_EXPERTS:
                        packed = tensors[f"{name}.su"]
                        expected = torch.where(
                            (packed.to(torch.int32) & 0xFFFF) == 0, 1.0, -1.0
                        ).repeat_interleave(16)[:k]
                        assert torch.equal(suh.float(), expected)
                        packed_v = tensors[f"{name}.sv"]
                        expected_v = torch.where(
                            (packed_v.to(torch.int32) & 0xFFFF) == 0, 1.0, -1.0
                        ).repeat_interleave(16)[:n]
                        assert torch.equal(svh.float(), expected_v)
                    else:
                        assert torch.equal(suh, tensors[f"{name}.suh"])
                        assert torch.equal(svh, tensors[f"{name}.svh"])



def test_cli_imports_exl3_banks_and_records_pinned_provenance(tmp_path):
    import os
    import subprocess
    import sys

    exl3 = _checkpoint(tmp_path / "exl3", _config(quantized=True), _exl3_tensors())
    base = _checkpoint(tmp_path / "base", _config(), _base_tensors())
    exl3.close()
    base.close()
    output = tmp_path / "cli.ninfer"
    command = [
        sys.executable, "-B", "-m", "tools.convert",
        "--model", str(base.path),
        "--recipe", "qwen3_6_35b_a3b_exl3",
        "--source", f"exl3={exl3.path}",
        "--out", str(output),
        "--device", "cpu",
        "--name", "exl3-cli-test",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    result = subprocess.run(
        command, capture_output=True, text=True, env=environment, timeout=600
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(Path(str(output) + ".conversion.json").read_text())
    assert report["formats"]["exl3"] == LAYERS * 3
    assert report["provenance"]["sources"]["exl3"]["exl3_version"] == PINNED_EXL3_VERSION
    with Artifact.open(output) as artifact:
        banks = [
            obj for obj in artifact.objects
            if isinstance(obj, TensorObject) and obj.format == "exl3"
        ]
        assert len(banks) == LAYERS * 3
        for bank in banks:
            payload = artifact.read_object(bank.id)
            assert int.from_bytes(payload[-4:], "little") in (0, 1, 2)


def test_non_uniform_codebook_and_wrong_shapes_are_rejected(tmp_path):
    tensors = _exl3_tensors()
    # Layer 0 gate: expert 2 switches to mul1, breaking the bank's uniform codebook.
    # The codebook is a bank-level property, so the recipe's prepare rejects it.
    mixed = dict(tensors)
    name = "model.layers.0.mlp.experts.2.gate_proj"
    del mixed[f"{name}.mcg"]
    mixed[f"{name}.mul1"] = torch.tensor(
        marker_as_int32(MUL1_MARKER), dtype=torch.int32
    )
    exl3 = _checkpoint(tmp_path / "mixed", _config(quantized=True), mixed)
    base = _checkpoint(tmp_path / "mixed-base", _config(), _base_tensors())
    with exl3, base:
        model = build_model(base)
        recipe = Recipe(model)
        with pytest.raises(ValueError, match="uniform"):
            qwen3_6_35b_a3b_exl3(model, recipe, {"exl3": exl3})
            recipe.prepare(device="cpu")
    # A trellis whose width disagrees with the architecture's expected shape.
    narrow = dict(tensors)
    narrow["model.layers.0.mlp.experts.0.gate_proj.trellis"] = torch.randint(
        -(1 << 15), 1 << 15, (HIDDEN // 16, INTERMEDIATE // 32, 32), dtype=torch.int16
    )
    with _checkpoint(tmp_path / "narrow", _config(quantized=True), narrow) as exl3:
        with pytest.raises(ValueError, match="128"):
            read_exl3_linear(
                exl3, "model.layers.0.mlp.experts.0.gate_proj",
                expected=(HIDDEN, INTERMEDIATE),
            )
    # Section 6.1 step 5: the EXL3 checkpoint's architecture must match the source's.
    mismatch = _config(quantized=True)
    mismatch["text_config"]["hidden_size"] = HIDDEN * 2
    with _checkpoint(tmp_path / "mismatch", mismatch, tensors) as exl3:
        base = _checkpoint(tmp_path / "mismatch-base", _config(), _base_tensors())
        with exl3, base:
            model = build_model(base)
            recipe = Recipe(model)
            with pytest.raises(ValueError, match="hidden_size"):
                qwen3_6_35b_a3b_exl3(model, recipe, {"exl3": exl3})
