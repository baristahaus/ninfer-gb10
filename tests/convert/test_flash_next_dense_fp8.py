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


def test_recipe_covers_every_fp8_capable_text_projection():
    parameters = dense_fp8.dense_fp8_parameters()
    # 12 QSA layers x 2, 36 GDN layers x 3, 96 layer HyperConnections, the final mixer, the head.
    assert len(parameters) == 24 + 108 + 96 + 1 + 1
    assert not any(name.startswith("mtp.") for name in parameters)
    assert parameters["lm_head.weight"] == (248320, 2560)
