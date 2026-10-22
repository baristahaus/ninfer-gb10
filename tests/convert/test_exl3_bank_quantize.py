"""The routed-bank quantize route (SP-7 contract 6.2): `convert_model_banks` plus the
`pack_exl3_bank_quantized` emit, assembled through the model's expert packing groups."""

import json

import torch
from safetensors.torch import save_file

from tools.convert.sources.safetensors import SafetensorsSource
from tools.convert.qwen3_5 import build_model
from tools.convert.recipe import Recipe
from tools.convert.official_recipes import _optional, _assign, Q8, Q6

from pathlib import Path

HIDDEN, INTERMEDIATE, EXPERTS, LAYERS = 256, 128, 4, 2
K2 = 6

def _config():
    return {"architectures": ["Qwen3_5MoeForCausalLM"], "text_config": {
        "hidden_size": HIDDEN, "vocab_size": 8, "num_hidden_layers": LAYERS,
        "max_position_embeddings": 128, "layer_types": ["linear_attention"] * LAYERS,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 128,
        "linear_num_key_heads": 1, "linear_key_head_dim": 128,
        "linear_num_value_heads": 1, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 3, "num_experts": EXPERTS,
        "moe_intermediate_size": INTERMEDIATE, "shared_expert_intermediate_size": INTERMEDIATE,
        "num_experts_per_tok": 2,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 1000000.0, "mrope_section": [11, 11, 10]},
        "partial_rotary_factor": 0.5}}

from tools.convert.pipeline import convert
from tools.artifact.reader import Artifact


def main(tmp):
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(20261019)
    def tensor(shape):
        return (torch.randn(tuple(shape), generator=generator) * 0.02).to(torch.bfloat16)
    tensors = {
        "model.language_model.embed_tokens.weight": tensor((8, HIDDEN)),
        "lm_head.weight": tensor((8, HIDDEN)),
        "model.language_model.norm.weight": tensor((HIDDEN,)),
    }
    for layer in range(LAYERS):
        p = f"model.language_model.layers.{layer}"
        tensors.update({
            f"{p}.input_layernorm.weight": tensor((HIDDEN,)),
            f"{p}.post_attention_layernorm.weight": tensor((HIDDEN,)),
            f"{p}.mlp.gate.weight": tensor((EXPERTS, HIDDEN)),
            f"{p}.mlp.shared_expert.gate_proj.weight": tensor((INTERMEDIATE, HIDDEN)),
            f"{p}.mlp.shared_expert.up_proj.weight": tensor((INTERMEDIATE, HIDDEN)),
            f"{p}.mlp.shared_expert.down_proj.weight": tensor((HIDDEN, INTERMEDIATE)),
            f"{p}.mlp.shared_expert_gate.weight": tensor((1, HIDDEN)),
            f"{p}.linear_attn.A_log": tensor((1,)),
            f"{p}.linear_attn.dt_bias": tensor((1,)),
            f"{p}.linear_attn.conv1d.weight": tensor((3 * 128, 1, 3)),
            f"{p}.linear_attn.in_proj_a.weight": tensor((1, HIDDEN)),
            f"{p}.linear_attn.in_proj_b.weight": tensor((1, HIDDEN)),
            f"{p}.linear_attn.in_proj_qkv.weight": tensor((384, HIDDEN)),
            f"{p}.linear_attn.in_proj_z.weight": tensor((128, HIDDEN)),
            f"{p}.linear_attn.norm.weight": tensor((128,)),
            f"{p}.linear_attn.out_proj.weight": tensor((HIDDEN, 128)),
            f"{p}.mlp.experts.gate_up_proj": tensor((EXPERTS, 2 * INTERMEDIATE, HIDDEN)),
            f"{p}.mlp.experts.down_proj": tensor((EXPERTS, HIDDEN, INTERMEDIATE)),
        })
    (base := tmp / 'base').mkdir(exist_ok=True)
    (base / "config.json").write_text(json.dumps(_config()))
    save_file(tensors, base / "model-00001-of-00001.safetensors")
    for role, value in {"tokenizer.json": {"model": {"vocab": {str(i): i for i in range(8)}}},
                        "tokenizer_config.json": {}, "generation_config.json": {}}.items():
        (base / role).write_text(json.dumps(value))
    (base / "chat_template.jinja").write_text("{{ messages }}")

    # The bank campaign: uncalibrated per-expert quantize (the real converter path).
    from tools.convert.quantization import exl3_convert as conv
    converted = tmp / "banks"
    conv.convert_model_banks(base, converted, k2=K2, codebook="mcg", device=torch.device("cpu"))

    source = SafetensorsSource(base)
    model = build_model(source)
    recipe = Recipe(model)
    from tools.convert.official_recipes import _optional, _assign, Q8, Q6
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    experts, layers = EXPERTS, LAYERS
    for layer in range(layers):
        for role, k, n in (("gate", HIDDEN, INTERMEDIATE), ("up", HIDDEN, INTERMEDIATE),
                           ("down", INTERMEDIATE, HIDDEN)):
            names = []
            for expert in range(experts):
                name = f"text/layers/{layer}/moe/experts/{expert}/{role}"
                recipe.assign(name, format="exl3", layout="exl3_expert_bank",
                              method="pack_exl3_bank_quantized",
                              parameters={"layer_file": str(converted / f"layer_{layer:02d}.pt"),
                                          "projection": f"routed_{role}", "k2": K2})
                names.append(name)
            recipe.group(names)
    from tools.convert.pipeline import convert as _c  # noqa
    output = tmp / "moe_banks.ninfer"
    report = convert(model, recipe, output, device="cpu", name="moe-banks-test")
    assert report["formats"]["exl3"] == 3 * LAYERS, report["formats"]
    print("assembled:", output, report["formats"])

    from tools.artifact.layouts import exl3_bank_geometry
    from tools.artifact.schema import TensorObject
    with Artifact.open(output) as artifact:
        banks = [o for o in artifact.objects
                 if isinstance(o, TensorObject) and o.layout == "exl3_expert_bank"]
        assert len(banks) == 3 * LAYERS, len(banks)
        bindings = artifact.directory.bindings
        e0 = bindings["text/layers/0/moe/experts/0/gate"]
        parts = e0.get("parts") or [{"object": e0["object"], "range": None}]
        obj = next(o for o in banks if o.id == parts[0]["object"])
        payload = artifact.read_object(obj.id)
        n, k = INTERMEDIATE, HIDDEN
        geometry = exl3_bank_geometry(n, k, [K2] * EXPERTS, "mcg")
        assert tuple(obj.shape) == (n * EXPERTS, k), obj.shape
        assert len(payload) == geometry.payload_bytes, (len(payload), geometry.payload_bytes)
        assert int.from_bytes(payload[-4:], "little") == 1
        # Fused-group members bind consecutive expert row ranges of the bank parent.
        rows = [b for b in bindings if b.startswith("text/layers/0/moe/experts/") and b.endswith("/gate")]
        spans = []
        for name in sorted(rows):
            bb = bindings[name]
            pp = bb.get("parts") or [{"range": None}]
            spans.append(tuple(pp[0]["range"]) if pp[0].get("range") else None)
        assert all(sp is not None for sp in spans), spans
        assert spans[0][0] == 0 and spans[1][0] == spans[0][1], spans
    print("bank binding and payload checks passed")

def test_bank_quantize_route_assembles_and_verifies(tmp_path):
    main(str(tmp_path))
