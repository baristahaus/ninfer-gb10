"""Exact source-word preservation across Flash-Next's v3 conversion boundary."""
import json
import struct
from pathlib import Path

import pytest
import torch

from tools.artifact.codecs.direct import encode_direct, decode_direct
from tools.artifact.layouts import encoded_size
from tools.artifact.schema import TensorSpec, plan_objects
from tools.convert.qwen3_8_flash_next_125b_a6b import convert, descriptor, inventory
from tools.convert.sources.safetensors import SafetensorsSource
from tools.upgrade_ninfer_v2_to_v3 import flash_next_directory


def test_raw_fp8_words():
    words = bytes(range(256))
    source = torch.arange(256).to(torch.uint8).view(torch.float8_e4m3fn)
    assert encode_direct(source, "fp8_e4m3fn") == words
    assert decode_direct(words, "fp8_e4m3fn", (256,)).view(torch.uint8).tolist() == list(range(256))


def test_swift_bf16_ple_inventory():
    specs = inventory.object_specs(inventory.BF16)
    names = {spec if isinstance(spec, str) else spec.id for spec in specs}
    table = next(spec for spec in specs if isinstance(spec, TensorSpec)
                 and spec.id == convert._PLE_TABLE)
    assert table.format == inventory.BF16
    directory = descriptor.describe([o.to_json() for o in plan_objects([
        spec for spec in specs if isinstance(spec, TensorSpec)])])
    assert all(use["parameter"] != table.id for use in directory["uses"])
    assert convert._PLE_PREFIX + "weight_scale" not in names
    signatures = convert._expected_source_signatures(inventory.BF16)
    assert signatures[convert._PLE_SHARDS[0]] == ((2_500_012, 160), "BF16")
    assert convert._PLE_PREFIX + "weight_scale" not in signatures


@pytest.mark.parametrize("role,n,k", [("gate_up", 1280, 2560), ("down", 2560, 640)])
def test_expert_bank_exact_source_words(tmp_path, monkeypatch, role, n, k):
    # Two experts at real projection dimensions expose bank offsets and gate/up concatenation.
    monkeypatch.setattr(inventory, "EXPERTS", 2)
    tensors = {}
    for expert in range(2):
        for projection in (("gate_proj", "up_proj") if role == "gate_up" else ("down_proj",)):
            rows = n // 2 if role == "gate_up" else n
            prefix = convert._expert_source(0, expert, projection, "")
            seed = expert * 17 + (3 if projection == "up_proj" else 0)
            tensors[prefix + "weight"] = ((torch.arange(rows * k // 2) + seed) % 256).to(torch.uint8).reshape(rows, k // 2)
            tensors[prefix + "weight_scale"] = ((torch.arange(rows * k // 16) + seed) % 127).to(torch.uint8).reshape(rows, k // 16).view(torch.float8_e4m3fn)
            tensors[prefix + "weight_scale_2"] = torch.tensor(0.5 * (expert + 1))
            tensors[prefix + "input_scale"] = torch.tensor(0.25 * (expert + 1))
    # Minimal genuine Safetensors file, independently framed without safetensors or converter codecs.
    header, raw = {}, bytearray()
    dtypes = {torch.uint8: "U8", torch.float8_e4m3fn: "F8_E4M3", torch.float32: "F32"}
    for name, tensor in tensors.items():
        data = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": dtypes[tensor.dtype], "shape": list(tensor.shape), "data_offsets": [len(raw), len(raw)+len(data)]}
        raw.extend(data)
    encoded = json.dumps(header).encode()
    path = tmp_path / "source.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw)
    with SafetensorsSource(path) as source:
        payload = b"".join(convert._expert_bank_payload(source, 0, role))
        divisors = convert._input_divisors(source, 0, role)
    assert len(payload) == encoded_size("expert_block_scale_k16_m128x4_v1", "nvfp4", (2, n, k))
    assert divisors == struct.pack("<ff", 4.0, 2.0)
    assert payload[-8:] == struct.pack("<ff", 2.0, 1.0)
    for expert in range(2):
        projections = ("gate_proj", "up_proj") if role == "gate_up" else ("down_proj",)
        packed = b"".join(tensors[convert._expert_source(0, expert, p, "weight")].numpy().tobytes() for p in projections)
        assert payload[expert*n*k//2:(expert+1)*n*k//2] == packed
        for row, group in ((0,0),(31,3),(32,4),(n//2,7),(n-1,k//16-1)):
            projection = projections[int(row >= n//2)] if role == "gate_up" else "down_proj"
            local_row = row % (n//2) if role == "gate_up" else row
            source_word = tensors[convert._expert_source(0, expert, projection, "weight_scale")].view(torch.uint8)[local_row,group].item()
            # Contract: tile(N/128,K/64), row%32, row-group-of-32, K-scale%4.
            within = ((((row//128)*(k//64)+group//4)*32+row%32)*4+(row%128)//32)*4+group%4
            assert payload[2*n*k//2 + expert*n*k//16 + within] == source_word


def test_source_and_upgrade_descriptors_agree():
    objects = [o.to_json() for o in plan_objects(inventory.TENSOR_SPECS)]
    old = []
    from tools.upgrade_ninfer_v2_to_v3 import FORMATS, LAYOUTS
    formats, layouts = {v:k for k,v in FORMATS.items()}, {v:k for k,v in LAYOUTS.items()}
    for obj in objects:
        old.append({"name": obj["id"], "kind":"tensor", "shape": obj["shape"],
                    "format": formats[obj["format"]], "layout": layouts[obj["layout"]],
                    "offset": obj["offset"], "bytes": obj["bytes"]})
    upgraded = flash_next_directory({"model_id":"qwen3.8-flash-next-125b-a6b", "weights_id":"nvfp4"}, old)
    expected = descriptor.describe(objects)
    for key in ("components", "bindings", "uses"):
        assert upgraded[key] == expected[key]
