"""Artifact-to-artifact FP8 re-encoding of the Flash-Next dense projections."""

import json

import pytest
import torch

from tools.artifact.codecs.direct import encode_direct
from tools.artifact.codecs.fp8_row import decode_fp8_row_scaled_words
from tools.artifact.codecs.nvfp4 import decode_nvfp4_words, encode_nvfp4, swizzle_nvfp4_scales
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.qwen3_8_flash_next_125b_a6b import dense_fp8
from tools.convert.quantization.fp8_row import quantize_bf16_rows
from tools.convert.quantization.nvfp4 import quantize_bf16_nvfp4


def _source(path, *, alias=False):
    dense = torch.linspace(-3.0, 5.0, 4 * 64, dtype=torch.float32).reshape(4, 64)
    dense[1] = 0.0
    dense[2] *= 1.0e-3
    dense = dense.to(torch.bfloat16)
    other = torch.arange(2 * 32, dtype=torch.float32).reshape(2, 32).to(torch.bfloat16)
    divisor = torch.tensor([0.5, 2.0, 4.0], dtype=torch.float32)
    specs = [
        ResourceSpec("frontend/tokenizer.json", 2),
        TensorSpec("w.dense", (4, 64), "bf16", "contiguous_le_v1"),
        TensorSpec("w.other", (2, 32), "bf16", "contiguous_le_v1"),
        TensorSpec("w.divisor", (3,), "fp32", "contiguous_le_v1"),
    ]
    bindings = {
        "text/dense": {"object": "w.dense"},
        "text/other": {"object": "w.other"},
        "text/divisor": {"object": "w.divisor"},
    }
    if alias:
        bindings["text/alias"] = {"object": "w.dense"}
    components = {
        "text": {
            "config": {"architectures": ["Synthetic"]},
            "resources": {"tokenizer.json": "frontend/tokenizer.json"},
        }
    }
    uses = [
        {"parameter": "text/dense", "input": "text/dense/input", "activation_policy": "A16Only"},
        {"parameter": "text/other", "input": "text/other/input", "activation_policy": "A16Only"},
    ]
    with ArtifactWriter(
        path,
        specs,
        components=components,
        bindings=bindings,
        uses=uses,
        metadata={"name": "synthetic"},
        provenance={"source": "unit-test", "recipe": "synthetic-bf16"},
    ) as writer:
        writer.write_object("frontend/tokenizer.json", b"{}")
        writer.write_object("w.dense", encode_direct(dense, "bf16"))
        writer.write_object("w.other", encode_direct(other, "bf16"))
        writer.write_object("w.divisor", encode_direct(divisor, "fp32"))
    return dense


def test_selected_matrix_is_quantized_and_everything_else_is_copied(tmp_path, monkeypatch):
    source = tmp_path / "source.ninfer"
    output = tmp_path / "out" / "fp8.ninfer"
    dense = _source(source)
    # Two rows per chunk: the concurrent chunks must reassemble in row order.
    monkeypatch.setattr(dense_fp8, "QUANTIZE_CHUNK_ELEMENTS", 128)
    report = dense_fp8.reencode(source, output, {"text/dense": (4, 64)}, recipe_id="test-fp8")

    expected = quantize_bf16_rows(dense)
    with Artifact(source) as before, Artifact(output) as after:
        converted = after.object("w.dense")
        assert (converted.format, converted.layout) == ("fp8_e4m3fn_row_bf16", "row_scale_v1")
        codes, scales = decode_fp8_row_scaled_words(after.read_object("w.dense"), (4, 64))
        assert torch.equal(codes, expected.codes)
        assert torch.equal(scales.view(torch.int16), expected.scales.view(torch.int16))
        for object_id in ("frontend/tokenizer.json", "w.other", "w.divisor"):
            assert after.read_object(object_id) == before.read_object(object_id)
        for key in ("components", "bindings", "uses", "metadata"):
            assert getattr(after.directory, key) == getattr(before.directory, key)
        assert after.directory.provenance["recipe"] == "test-fp8"
        assert after.directory.provenance["derived_from_recipe"] == "synthetic-bf16"

    written = json.loads((tmp_path / "out" / "fp8.ninfer.conversion.json").read_text())
    assert written["reencoded_tensors"] == report["reencoded_tensors"] == 1
    assert 0.0 < written["tensors"]["w.dense"]["relative_rms_error"] < 0.1


def test_shared_representation_is_refused(tmp_path):
    source = tmp_path / "source.ninfer"
    _source(source, alias=True)
    with pytest.raises(ValueError, match="shares a re-encoded object"):
        dense_fp8.reencode(
            source, tmp_path / "fp8.ninfer", {"text/dense": (4, 64)}, recipe_id="test-fp8"
        )


def _packed_source(path):
    gate = torch.linspace(-2.0, 2.0, 2 * 32, dtype=torch.float32).reshape(2, 32)
    up = torch.linspace(0.5, 7.0, 2 * 32, dtype=torch.float32).reshape(2, 32)
    gate, up = gate.to(torch.bfloat16), up.to(torch.bfloat16)
    other = torch.ones(3, dtype=torch.float32)
    specs = [
        ResourceSpec("frontend/tokenizer.json", 2),
        TensorSpec("w.gate", (2, 32), "bf16", "contiguous_le_v1"),
        TensorSpec("w.other", (3,), "fp32", "contiguous_le_v1"),
        TensorSpec("w.up", (2, 32), "bf16", "contiguous_le_v1"),
    ]
    bindings = {
        "text/gate": {"object": "w.gate"},
        "text/up": {"object": "w.up"},
        "text/other": {"object": "w.other"},
    }
    uses = [
        {"parameter": "text/gate", "input": "text/input", "activation_policy": "A16Only"},
        {"parameter": "text/up", "input": "text/input", "activation_policy": "A16Only"},
    ]
    components = {
        "text": {
            "config": {"architectures": ["Synthetic"]},
            "resources": {"tokenizer.json": "frontend/tokenizer.json"},
        }
    }
    with ArtifactWriter(
        path,
        specs,
        components=components,
        bindings=bindings,
        uses=uses,
        metadata={"name": "synthetic"},
        provenance={"source": "unit-test", "recipe": "synthetic-bf16"},
    ) as writer:
        writer.write_object("frontend/tokenizer.json", b"{}")
        writer.write_object("w.gate", encode_direct(gate, "bf16"))
        writer.write_object("w.other", encode_direct(other, "fp32"))
        writer.write_object("w.up", encode_direct(up, "bf16"))
    return gate, up


def test_packed_parent_stacks_rows_and_rebinds_its_parameters(tmp_path):
    source = tmp_path / "source.ninfer"
    output = tmp_path / "fp8.ninfer"
    gate, up = _packed_source(source)
    packed = {"w.gate_up": dense_fp8.PackedParent(("text/gate", "text/up"), (4, 32))}
    report = dense_fp8.reencode(source, output, {}, recipe_id="test-fp8", packed=packed)

    expected = quantize_bf16_rows(torch.cat([gate, up]))
    with Artifact(source) as before, Artifact(output) as after:
        # The parent takes the first source's place; the second source is gone.
        assert [obj.id for obj in after.directory.objects] == [
            "frontend/tokenizer.json",
            "w.gate_up",
            "w.other",
        ]
        converted = after.object("w.gate_up")
        assert (converted.format, converted.layout) == ("fp8_e4m3fn_row_bf16", "row_scale_v1")
        codes, scales = decode_fp8_row_scaled_words(after.read_object("w.gate_up"), (4, 32))
        assert torch.equal(codes, expected.codes)
        assert torch.equal(scales.view(torch.int16), expected.scales.view(torch.int16))
        assert after.directory.bindings["text/gate"] == {
            "parts": [{"object": "w.gate_up", "range": [0, 64]}]
        }
        assert after.directory.bindings["text/up"] == {
            "parts": [{"object": "w.gate_up", "range": [64, 128]}]
        }
        assert after.directory.bindings["text/other"] == before.directory.bindings["text/other"]
        assert after.read_object("w.other") == before.read_object("w.other")
        for key in ("components", "uses", "metadata"):
            assert getattr(after.directory, key) == getattr(before.directory, key)
    assert report["reencoded_tensors"] == 1


def test_recipe_covers_every_fp8_capable_projection_and_the_mtp_banks():
    parameters = dense_fp8.dense_fp8_parameters()
    # Text: 12 QSA layers x 2, 36 GDN layers x 3, 96 layer HyperConnections and the final mixer
    # x 2 (down and up), 48 shared-expert down projections, the head. MTP: its QSA q/o, its two
    # HyperConnections and final mixer x 2, its shared-expert down projection.
    assert len(parameters) == (24 + 108 + 2 * (96 + 1) + 48 + 1) + (2 + 2 * 3 + 1)
    mtp = {name for name in parameters if name.startswith("mtp.")}
    assert mtp == {
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
        "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
        "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
        "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
        "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    }
    assert parameters["lm_head.weight"] == (248320, 2560)
    packed = dense_fp8.dense_fp8_packed_parents()
    assert len(packed) == 48 + 1
    for packed_id, parent in packed.items():
        assert parent.shape == (1280, 2560)
        gate, up = parent.parameters
        assert gate.endswith("shared_expert.gate_proj.weight")
        assert up.endswith("shared_expert.up_proj.weight")
        assert packed_id.endswith("shared_expert.gate_up_proj.weight")
        assert gate not in parameters and up not in parameters
    banks = dense_fp8.mtp_nvfp4_banks()
    assert banks == {
        "mtp.layers.0.mlp.experts.gate_up": dense_fp8.ExpertBank(
            "mtp.layers.0.mlp.experts.gate_up_proj", (512, 1280, 2560), "gate_up"
        ),
        "mtp.layers.0.mlp.experts.down": dense_fp8.ExpertBank(
            "mtp.layers.0.mlp.experts.down_proj", (512, 2560, 640), "down"
        ),
    }


def _bank_source(path):
    """Two main layers' activation divisors and one two-expert BF16 MTP bank [2,128,64]."""

    generator = torch.Generator().manual_seed(11)
    bank = (torch.randn(2, 128, 64, generator=generator) * 0.02).to(torch.bfloat16)
    bank[1, :, :32] *= 50.0  # the second expert has its own range
    divisors = {
        0: torch.tensor([400.0, 300.0], dtype=torch.float32),
        1: torch.tensor([250.0, 500.0], dtype=torch.float32),
    }
    specs = [ResourceSpec("frontend/tokenizer.json", 2)]
    bindings = {}
    for layer, values in divisors.items():
        name = f"model.language_model.layers.{layer}.mlp.experts.gate_up_input_divisors"
        specs.append(TensorSpec(name, (2,), "fp32", "contiguous_le_v1"))
        bindings[name] = {"object": name}
    specs.append(TensorSpec("mtp.bank_proj", (2, 128, 64), "bf16", "contiguous_le_v1"))
    bindings["mtp.bank_proj"] = {"object": "mtp.bank_proj"}
    uses = [
        {"parameter": "mtp.bank_proj", "input": "mtp.bank_proj/input", "activation_policy": "A16Only"}
    ]
    components = {
        "text": {
            "config": {"architectures": ["Synthetic"]},
            "resources": {"tokenizer.json": "frontend/tokenizer.json"},
        }
    }
    with ArtifactWriter(
        path,
        specs,
        components=components,
        bindings=bindings,
        uses=uses,
        metadata={"name": "synthetic"},
        provenance={"source": "unit-test", "recipe": "synthetic-bf16"},
    ) as writer:
        writer.write_object("frontend/tokenizer.json", b"{}")
        for layer, values in divisors.items():
            name = f"model.language_model.layers.{layer}.mlp.experts.gate_up_input_divisors"
            writer.write_object(name, encode_direct(values, "fp32"))
        writer.write_object("mtp.bank_proj", encode_direct(bank, "bf16"))
    return bank


def test_mtp_bank_becomes_an_nvfp4_bank_with_main_layer_divisors(tmp_path, monkeypatch):
    source = tmp_path / "source.ninfer"
    output = tmp_path / "mtp.ninfer"
    bank = _bank_source(source)
    monkeypatch.setattr(dense_fp8.inventory, "LAYERS", (0, 1))
    monkeypatch.setattr(dense_fp8.inventory, "EXPERTS", 2)
    banks = {"mtp.bank": dense_fp8.ExpertBank("mtp.bank_proj", (2, 128, 64), "gate_up")}
    report = dense_fp8.reencode(source, output, {}, recipe_id="test-mtp", banks=banks)

    with Artifact(output) as after:
        directory = after.directory
        assert "mtp.bank_proj" not in directory.bindings
        assert directory.bindings["mtp.bank"] == {"object": "mtp.bank"}
        assert all(use["parameter"] != "mtp.bank_proj" for use in directory.uses)
        assert {
            "parameter": "mtp.bank",
            "input": "mtp.bank/input",
            "activation_policy": "AllowA4",
            "auxiliaries": {"activation_divisor": {"object": "mtp.bank_input_divisors"}},
        } in directory.uses
        converted = after.object("mtp.bank")
        assert (converted.format, converted.layout) == ("nvfp4", "expert_block_scale_k16_m128x4_v1")
        # The smallest main-layer divisor, for every expert.
        divisors = torch.frombuffer(bytearray(after.read_object("mtp.bank_input_divisors")),
                                    dtype=torch.float32)
        assert divisors.tolist() == [250.0, 250.0]

        payload = after.read_object("mtp.bank")
        code_bytes = 2 * 128 * 64 // 2
        scale_offset = (code_bytes + 255) // 256 * 256
        scale_bytes = 128 * 64 // 16
        for expert in range(2):
            expected = quantize_bf16_nvfp4(bank[expert])
            words = encode_nvfp4(expected.codes, expected.scales, expected.divisor, (128, 64))
            codes, scales, divisor = decode_nvfp4_words(words, (128, 64))
            begin = expert * code_bytes // 2
            assert payload[begin : begin + code_bytes // 2] == codes.numpy().tobytes()
            stored = payload[scale_offset + expert * scale_bytes :
                             scale_offset + (expert + 1) * scale_bytes]
            assert stored == swizzle_nvfp4_scales(scales, (128, 64)).numpy().tobytes()
            stored_divisor = payload[scale_offset + 2 * scale_bytes + 4 * expert :
                                     scale_offset + 2 * scale_bytes + 4 * (expert + 1)]
            assert stored_divisor == encode_direct(divisor.reshape(1), "fp32")
    assert 0.0 < report["tensors"]["mtp.bank"]["relative_rms_error"] < 0.12
    assert report["activation_divisors"]["mtp.bank_input_divisors"]["divisor"] == 250.0
