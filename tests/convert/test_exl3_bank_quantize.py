"""The routed-bank quantize route (SP-7 contract 6.2): `convert_model_banks` plus the
`pack_exl3_bank_quantized` emit, assembled through the model's expert packing groups.

Two synthetic geometries run the same helpers:

- the 35B-shaped case (4 experts, hidden 256, intermediate 128), with the real
  uncalibrated `convert_model_banks` campaign, and
- a 512-expert case approximating the 125B Flash-Next MoE directory scale (the
  real geometry's hidden 2560 / intermediate 640 are out of budget here; the
  point is the 512-entry directory, the 512-row suh/svh planes and the 512-member
  fused group). Its campaign expands one genuine per-role `quantize_linear`
  result -- the same helper `convert_model_banks` calls -- to the full expert
  count by sign negation, because the real 512-expert campaign is compute-bound
  in the per-expert Viterbi (~hours on CPU) while the emit path under test is
  per-expert independent.
"""

import json
import struct

import torch
from safetensors.torch import load_file, save_file

from tools.artifact.reader import Artifact
from tools.convert.pipeline import convert
from tools.convert.sources.safetensors import SafetensorsSource
from tools.convert.qwen3_5 import build_model
from tools.convert.recipe import Recipe
from tools.convert.official_recipes import _optional, _assign, Q8, Q6

from pathlib import Path

HIDDEN, INTERMEDIATE, EXPERTS, LAYERS = 256, 128, 4, 2
K2 = 6
CODEBOOK = "mcg"


def _config(hidden, intermediate, experts, layers, top_k):
    return {"architectures": ["Qwen3_5MoeForCausalLM"], "text_config": {
        "hidden_size": hidden, "vocab_size": 8, "num_hidden_layers": layers,
        "max_position_embeddings": 128, "layer_types": ["linear_attention"] * layers,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 128,
        "linear_num_key_heads": 1, "linear_key_head_dim": 128,
        "linear_num_value_heads": 1, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 3, "num_experts": experts,
        "moe_intermediate_size": intermediate, "shared_expert_intermediate_size": intermediate,
        "num_experts_per_tok": top_k,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 1000000.0, "mrope_section": [11, 11, 10]},
        "partial_rotary_factor": 0.5}}


def _write_checkpoint(tmp, hidden, intermediate, experts, layers, top_k):
    generator = torch.Generator().manual_seed(20261019)
    def tensor(shape):
        return (torch.randn(tuple(shape), generator=generator) * 0.02).to(torch.bfloat16)
    tensors = {
        "model.language_model.embed_tokens.weight": tensor((8, hidden)),
        "lm_head.weight": tensor((8, hidden)),
        "model.language_model.norm.weight": tensor((hidden,)),
    }
    for layer in range(layers):
        p = f"model.language_model.layers.{layer}"
        tensors.update({
            f"{p}.input_layernorm.weight": tensor((hidden,)),
            f"{p}.post_attention_layernorm.weight": tensor((hidden,)),
            f"{p}.mlp.gate.weight": tensor((experts, hidden)),
            f"{p}.mlp.shared_expert.gate_proj.weight": tensor((intermediate, hidden)),
            f"{p}.mlp.shared_expert.up_proj.weight": tensor((intermediate, hidden)),
            f"{p}.mlp.shared_expert.down_proj.weight": tensor((hidden, intermediate)),
            f"{p}.mlp.shared_expert_gate.weight": tensor((1, hidden)),
            f"{p}.linear_attn.A_log": tensor((1,)),
            f"{p}.linear_attn.dt_bias": tensor((1,)),
            f"{p}.linear_attn.conv1d.weight": tensor((3 * 128, 1, 3)),
            f"{p}.linear_attn.in_proj_a.weight": tensor((1, hidden)),
            f"{p}.linear_attn.in_proj_b.weight": tensor((1, hidden)),
            f"{p}.linear_attn.in_proj_qkv.weight": tensor((384, hidden)),
            f"{p}.linear_attn.in_proj_z.weight": tensor((128, hidden)),
            f"{p}.linear_attn.norm.weight": tensor((128,)),
            f"{p}.linear_attn.out_proj.weight": tensor((hidden, 128)),
            f"{p}.mlp.experts.gate_up_proj": tensor((experts, 2 * intermediate, hidden)),
            f"{p}.mlp.experts.down_proj": tensor((experts, hidden, intermediate)),
        })
    (base := tmp / 'base').mkdir(exist_ok=True)
    save_file(tensors, base / "model-00001-of-00001.safetensors")
    (base / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(t.numel() * t.element_size()
                                        for t in tensors.values())},
         "weight_map": {name: "model-00001-of-00001.safetensors"
                        for name in tensors}}))
    (base / "config.json").write_text(json.dumps(
        _config(hidden, intermediate, experts, layers, top_k)))
    for role, value in {"tokenizer.json": {"model": {"vocab": {str(i): i for i in range(8)}}},
                        "tokenizer_config.json": {}, "generation_config.json": {}}.items():
        (base / role).write_text(json.dumps(value))
    (base / "chat_template.jinja").write_text("{{ messages }}")
    return base


def _campaign_real(base, converted, k2, experts=None):
    """The uncalibrated per-expert campaign, exactly as the converter runs it.

    The converter's own default device is CUDA; the CPU fallback keeps the test
    runnable on CPU-only hosts (at ~4x the campaign cost).
    """

    from tools.convert.quantization import exl3_convert as conv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conv.convert_model_banks(base, converted, k2=k2, codebook=CODEBOOK, device=device)


def _campaign_expanded(base, converted, k2, experts):
    """The 512-expert campaign at emit-test cost.

    Quantizes expert 0 of each role with the same `quantize_linear` call
    `convert_model_banks` makes (same weight slicing, same su/sv draw order and
    seed), then expands it to the full expert count by sign negation: negating
    suh rows / svh columns is exactly the linear of a row- / column-negated
    expert weight with the trellis unchanged, so every expert entry stays a
    self-consistent (suh, svh, trellis) triple while the planes differ per
    expert. The full per-expert Viterbi campaign is compute-bound (~hours) and
    exercises nothing the emit path depends on.
    """

    from tools.convert.quantization import exl3_quantize as q
    tensors = load_file(Path(base) / "model-00001-of-00001.safetensors")
    gate_up = tensors["model.language_model.layers.0.mlp.experts.gate_up_proj"].float()
    down = tensors["model.language_model.layers.0.mlp.experts.down_proj"].float()
    hidden = gate_up.shape[-1]
    intermediate = gate_up.shape[1] // 2
    generator = torch.Generator().manual_seed(20261016)
    result = {}
    for role, weight in (("routed_gate", gate_up[0][0:intermediate]),
                         ("routed_up", gate_up[0][intermediate:2 * intermediate]),
                         ("routed_down", down[0])):
        su = torch.sign(torch.randn(weight.shape[1], generator=generator))
        sv = torch.sign(torch.randn(weight.shape[0], generator=generator))
        out = q.quantize_linear(weight.T.contiguous(), k2, CODEBOOK, su, sv)
        out["weight_shape"] = tuple(weight.shape)
        suh0, svh0 = out["suh"], out["svh"]
        entries = []
        for _expert in range(experts):
            entry = dict(out)
            entry["suh"] = suh0 * torch.sign(
                torch.randn(suh0.shape[0], generator=generator)).to(torch.float16)
            entry["svh"] = svh0 * torch.sign(
                torch.randn(svh0.shape[0], generator=generator)).to(torch.float16)
            entries.append(entry)
        result[role] = entries
    converted.mkdir(parents=True, exist_ok=True)
    torch.save(result, converted / "layer_00.pt")


def main(tmp, *, hidden=HIDDEN, intermediate=INTERMEDIATE, experts=EXPERTS,
         layers=LAYERS, top_k=2, campaign=_campaign_real):
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    base = _write_checkpoint(tmp, hidden, intermediate, experts, layers, top_k)

    # The bank campaign (the real converter path by default).
    converted = tmp / "banks"
    campaign(base, converted, K2, experts)

    source = SafetensorsSource(base)
    model = build_model(source)
    recipe = Recipe(model)
    from tools.convert.official_recipes import _optional, _assign, Q8, Q6
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for layer in range(layers):
        for role, k, n in (("gate", hidden, intermediate), ("up", hidden, intermediate),
                           ("down", intermediate, hidden)):
            names = []
            for expert in range(experts):
                name = f"text/layers/{layer}/moe/experts/{expert}/{role}"
                recipe.assign(name, format="exl3", layout="exl3_expert_bank",
                              method="pack_exl3_bank_quantized",
                              parameters={"layer_file": str(converted / f"layer_{layer:02d}.pt"),
                                          "projection": f"routed_{role}", "k2": K2})
                names.append(name)
            recipe.group(names)
    output = tmp / "moe_banks.ninfer"
    report = convert(model, recipe, output, device="cpu", name="moe-banks-test")
    assert report["formats"]["exl3"] == 3 * layers, report["formats"]
    print("assembled:", output, report["formats"])

    from tools.artifact.layouts import exl3_bank_geometry, align_up
    from tools.artifact.schema import TensorObject
    with Artifact.open(output) as artifact:
        banks = [o for o in artifact.objects
                 if isinstance(o, TensorObject) and o.layout == "exl3_expert_bank"]
        assert len(banks) == 3 * layers, len(banks)
        bindings = artifact.directory.bindings
        e0 = bindings[f"text/layers/0/moe/experts/0/gate"]
        parts = e0.get("parts") or [{"object": e0["object"], "range": None}]
        obj = next(o for o in banks if o.id == parts[0]["object"])
        payload = artifact.read_object(obj.id)
        n, k = intermediate, hidden
        geometry = exl3_bank_geometry(n, k, [K2] * experts, CODEBOOK)
        assert tuple(obj.shape) == (n * experts, k), obj.shape
        assert len(payload) == geometry.payload_bytes, (len(payload), geometry.payload_bytes)
        assert int.from_bytes(payload[-4:], "little") == 1

        # The per-expert directory: one 16-byte <QIBBBB> entry per expert, offsets
        # increasing in expert order, every extent 256-byte aligned at the aligned
        # plane size, and the trellis plane closing exactly at the suh plane.
        entries = [struct.unpack_from("<QIBBBB", payload, expert * 16)
                   for expert in range(experts)]
        extent = align_up(n * k * K2 // 16, 256)
        assert all(offset == geometry.trellis_offset + expert * extent
                   and size == extent and rate == K2 and tuple(rest) == (0, 0, 0)
                   for expert, (offset, size, rate, *rest) in enumerate(entries)), \
            (entries[0], entries[-1])
        assert entries[-1][0] + extent == geometry.suh_offset

        # The suh/svh planes: fixed-stride fp16 [E, k] / [E, n] rows at their
        # 256-byte-aligned offsets, matching the campaign's per-expert rows.
        campaign_results = torch.load(converted / "layer_00.pt", weights_only=False)
        suh_plane = torch.frombuffer(
            bytearray(payload[geometry.suh_offset:geometry.suh_offset + geometry.suh_bytes]),
            dtype=torch.float16).reshape(experts, k)
        svh_plane = torch.frombuffer(
            bytearray(payload[geometry.svh_offset:geometry.svh_offset + geometry.svh_bytes]),
            dtype=torch.float16).reshape(experts, n)
        assert geometry.svh_offset == align_up(geometry.suh_offset + geometry.suh_bytes, 256)
        for expert in (0, experts // 2, experts - 1):
            result = campaign_results["routed_gate"][expert]
            assert torch.equal(suh_plane[expert], result["suh"].cpu()), expert
            assert torch.equal(svh_plane[expert], result["svh"].cpu()), expert

        # Fused-group members bind consecutive expert row ranges of the bank parent.
        rows = [b for b in bindings
                if b.startswith("text/layers/0/moe/experts/") and b.endswith("/gate")]
        assert len(rows) == experts, len(rows)
        spans = {}
        for name in rows:
            bb = bindings[name]
            pp = bb.get("parts") or [{"range": None}]
            spans[int(name.split("/")[-2])] = \
                tuple(pp[0]["range"]) if pp[0].get("range") else None
        assert all(span is not None for span in spans.values()), spans
        for expert in range(experts):
            # The parts' ranges are element ranges of the [E*n, k] bank parent.
            assert spans[expert] == (expert * n * k, (expert + 1) * n * k), \
                (expert, spans[expert])
        assert spans[experts - 1][1] == n * experts * k
    print("bank binding and payload checks passed")


def test_bank_quantize_route_assembles_and_verifies(tmp_path):
    main(str(tmp_path))


def test_bank_quantize_route_512_experts(tmp_path):
    main(str(tmp_path), experts=512, layers=1, top_k=10, campaign=_campaign_expanded)


def test_bank_geometry_rejects_non_128_multiple_dims(tmp_path):
    """The 128-multiple contract the synthetic Flash-Next geometry runs into: the
    converter's bank geometry admits N, K multiples of 128 only, so the real
    125B shapes (640 / 2560) pass while an intermediate of 64 cannot express a
    bank -- no 256-expert assumption exists anywhere on the route."""

    from tools.artifact.layouts import exl3_bank_geometry
    assert exl3_bank_geometry(640, 2560, [K2] * 512, CODEBOOK).experts == 512
    try:
        exl3_bank_geometry(64, 256, [K2] * 512, CODEBOOK)
    except ValueError as error:
        assert "divisible by 128" in str(error), error
    else:
        raise AssertionError("bank geometry accepted N not divisible by 128")

