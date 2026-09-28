"""Artifact-to-artifact FP8 re-encoding of the Flash-Next dense projections."""

import json

import pytest
import torch

from tools.artifact.codecs.direct import encode_direct
from tools.artifact.codecs.fp8_row import decode_fp8_row_scaled_words
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.qwen3_8_flash_next_125b_a6b import dense_fp8
from tools.convert.quantization.fp8_row import quantize_bf16_rows


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


def test_recipe_covers_every_fp8_capable_text_projection():
    parameters = dense_fp8.dense_fp8_parameters()
    # 12 QSA layers x 2, 36 GDN layers x 3, 96 layer HyperConnections and the final mixer x 2
    # (down and up), 48 shared-expert down projections, the head.
    assert len(parameters) == 24 + 108 + 2 * (96 + 1) + 48 + 1
    assert not any(name.startswith("mtp.") for name in parameters)
    assert parameters["lm_head.weight"] == (248320, 2560)
    packed = dense_fp8.dense_fp8_packed_parents()
    assert len(packed) == 48
    for packed_id, parent in packed.items():
        assert parent.shape == (1280, 2560)
        gate, up = parent.parameters
        assert gate.endswith("shared_expert.gate_proj.weight")
        assert up.endswith("shared_expert.up_proj.weight")
        assert packed_id.endswith("shared_expert.gate_up_proj.weight")
        assert gate not in parameters and up not in parameters
