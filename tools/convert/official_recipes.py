"""Official representation recipes built from the same public conversion functions."""

from __future__ import annotations

from pathlib import Path

from .methods import (
    cast_direct,
    fp8_row_maxabs,
    grouped_absmax,
    import_encoded,
    import_exl3_bank,
    pack_exl3_dense,
)
from .sources.exllamav3 import (
    _check_architecture,
    check_quantization_config,
    exl3_expert_source,
)

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"
FP8 = "fp8_e4m3fn_row_bf16"


def _assign(recipe, name, format, *, source=None):
    method = grouped_absmax if format in (Q4, Q5, Q6, Q8) else cast_direct
    recipe.assign(name, format=format, method=method, source=source)


def _optional(model, recipe):
    for name, parameter in model.parameters.items():
        if not parameter.projection:
            continue
        if name.startswith("vision/"):
            if name == "vision/patch_embedding":
                format = Q6
            elif name.startswith("vision/merger/"):
                format = Q8
            elif name.endswith(
                ("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")
            ):
                format = Q4
            else:
                format = Q5
            _assign(recipe, name, format)
        elif name.startswith(("mtp/", "dflash/", "dflash2/")):
            if name.endswith(
                (
                    "/moe/router",
                    "/moe/shared_score",
                    "/attention_conv/kernel_projection",
                    "/mlp_conv/kernel_projection",
                    "/candidate_selector/hidden_projection",
                )
            ):
                continue
            _assign(recipe, name, Q8)
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        layers = model.components[backend]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def _dense_groupwise(model, recipe, vocabulary):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", vocabulary)
    _assign(recipe, "text/output_head", vocabulary)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        if name.endswith(
            (
                "/attention/query",
                "/attention/key",
                "/gdn/query",
                "/gdn/key",
                "/mlp/gate",
                "/mlp/up",
            )
        ):
            format = Q4
        else:
            format = Q5
        _assign(recipe, name, format)


def qwen3_6_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q6)


def qwen3_8_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8)


def qwen3_6_35b_a3b(model, recipe, sources):
    if "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen3.5 MoE mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name:
            layer = int(name.split("/")[2])
            format = (
                (Q6 if layer in (34, 38, 39) else Q5) if name.endswith("/down") else Q4
            )
        else:
            format = Q8
        _assign(recipe, name, format)


def qwen3_8_27b_exl3_dense(model, recipe, sources):
    """The 27B dense route with EXL3-encoded projections (SP-7 contract 6.2 step 5).

    `sources["converted"]` names the fused quantize campaign's output directory: its
    `layer_NN.pt` files carry every projection group's packed result. Each mixer/FFN
    group (attention query/key/gate/value, GDN query/key/value/z, MLP gate/up) is ONE
    `exl3_dense_v1` object in the binding row order, bound through the model's packing
    groups; the campaign quantizes each group as a single linear with the group's shared
    input-side transform. The output projections (single parameters) are their own dense
    objects, and the a/b control projections plus every non-projection tensor keep the
    groupwise recipe's formats from the BF16 source.

    Bitrate map (the campaign's rate choice, recorded here because the recipe consumes
    its output): uniform k2 = 4 (K = 2 bits/weight, mcg) for every projection. The fused
    groups' measured budgets: the 64 layers' projection payloads total ~5.7 GiB, plus
    the Q8 token embedding / Q6 output head and the small remaining tensors, landing the
    artifact near 8.5 GiB against the 21.5 GiB NVFP4 artifact and the 32 GiB fully-hot
    envelope. The rate is fixed per quantize campaign; raising it (k2 = 6 or 8) also
    fits the envelope and is a measured quality-per-byte decision after the first
    artifact's perplexity, not an assumption.
    """

    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    converted = Path(sources["converted"])
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)

    # Parameter role -> the fused campaign tensor key. Attention layers fuse
    # (query, key, gate, value) where query and gate are the interleaved q_proj rows the
    # model binds separately; GDN layers fuse (query, key, value, z); every layer fuses
    # (gate, up). The campaign stores each group as one linear with a shared suh.
    layers = model.config["num_hidden_layers"]
    group_roles = ("query", "key", "gate", "value")
    gdn_group_roles = ("query", "key", "value", "z")
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            continue
        parts = name.split("/")
        layer = int(parts[2])
        role = parts[-1]
        gdn = "/gdn/" in name
        attention = "/attention/" in name
        fused_tensor = (
            "attention_fused" if attention else "gdn_fused" if gdn else "gate_up"
        )
        roles = group_roles if attention else gdn_group_roles if gdn else ("gate", "up")
        if role in roles:
            recipe.assign(
                name,
                format="exl3",
                layout="exl3_dense_v1",
                method="pack_exl3_dense_group",
                parameters={
                    "layer_file": str(converted / f"layer_{layer:02d}.pt"),
                    "tensor": fused_tensor,
                },
            )
            continue
        converted_key = {
            ("gdn", "output"): "out",
            ("attention", "output"): "output",
            ("mlp", "down"): "down",
        }.get(("gdn" if gdn else "attention" if attention else "mlp", role))
        if converted_key is None:
            continue
        recipe.assign(
            name,
            format="exl3",
            layout="exl3_dense_v1",
            method="pack_exl3_dense",
            parameters={
                "layer_file": str(converted / f"layer_{layer:02d}.pt"),
                "tensor": converted_key,
            },
        )
    # Bind each fused group as one parent object through the model's declared groups;
    # a layer carries either the attention or the GDN mixer.
    for layer in range(layers):
        prefix = f"text/layers/{layer}/"
        for roles in (group_roles, gdn_group_roles, ("gate", "up")):
            stem = "attention" if roles is group_roles else "gdn" if roles is gdn_group_roles else "mlp"
            names = [prefix + stem + "/" + role for role in roles]
            if all(name in model.parameters for name in names):
                recipe.group(names)


def qwen3_6_35b_a3b_exl3(model, recipe, sources):
    """Import the routed experts as EXL3 banks; everything else per the groupwise recipe.

    The EXL3 checkpoint supplies only the routed banks' tensors (SP-7 contract
    6.1); every non-routed tensor keeps the existing recipe's format from the BF16
    source, so artifacts differ from `qwen3_6_35b_a3b` only in the routed banks.
    """

    if "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen3.5 MoE mathematics")
    exl3 = sources["exl3"]
    check_quantization_config(exl3.config)
    _check_architecture(exl3.config, model.config)
    experts = model.config["num_experts"]
    hidden = model.config["hidden_size"]
    intermediate = model.config["moe_intermediate_size"]
    layers = model.config["num_hidden_layers"]
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name:
            continue
        _assign(recipe, name, Q8)
    for layer in range(layers):
        for role, k, n in (
            ("gate", hidden, intermediate),
            ("up", hidden, intermediate),
            ("down", intermediate, hidden),
        ):
            names = []
            for expert in range(experts):
                name = f"text/layers/{layer}/moe/experts/{expert}/{role}"
                tensor = f"model.layers.{layer}.mlp.experts.{expert}.{role}_proj"
                recipe.assign(
                    name,
                    format="exl3",
                    layout="exl3_expert_bank",
                    method="import_exl3_bank",
                    source=exl3_expert_source(exl3, tensor, expected=(k, n)),
                )
                names.append(name)
            recipe.group(names)


def qwen3_6_35b_a3b_exl3_banks(model, recipe, sources):
    """Quantize the routed experts as EXL3 banks from the BF16 source (SP-7 contract 6.2).

    `sources["converted"]` names the bank campaign's output directory (its `layer_NN.pt`
    files carry `routed_gate`/`routed_up`/`routed_down`, one uncalibrated quantize per
    expert); every expert parameter binds its bank through the model's expert packing
    groups, and every non-routed tensor keeps the groupwise recipe's formats from the
    BF16 source, so artifacts differ from `qwen3_6_35b_a3b` only in the routed banks.
    The campaign fixes one rate for every bank (BANK_K2 below); a different rate is a
    new campaign, and the recipe passes it to the emit for the bank geometry.
    """
    bank_k2 = 6

    if "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen3.5 MoE mathematics")
    converted = Path(sources["converted"])
    experts = model.config["num_experts"]
    layers = model.config["num_hidden_layers"]
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name:
            continue
        _assign(recipe, name, Q8)
    for layer in range(layers):
        for role in ("gate", "up", "down"):
            names = []
            for expert in range(experts):
                name = f"text/layers/{layer}/moe/experts/{expert}/{role}"
                recipe.assign(
                    name,
                    format="exl3",
                    layout="exl3_expert_bank",
                    method="pack_exl3_bank_quantized",
                    parameters={
                        "layer_file": str(converted / f"layer_{layer:02d}.pt"),
                        "projection": f"routed_{role}",
                        "k2": bank_k2,
                    },
                )
                names.append(name)
            recipe.group(names)


def qwen3_6_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q8)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        layer = int(name.split("/")[2])
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        direct = (
            ("/attention/" in name and not name.endswith("/output") and layer < 24)
            or (name.endswith("/attention/output") and layer in (3, 7))
            or (name.endswith("/gdn/output") and layer == 4)
        )
        if direct:
            continue
        recipe.assign(
            name,
            format="nvfp4",
            method=import_encoded,
            source=model.source(name, quantized, "nvfp4"),
            activation_policy="AllowA4",
        )


def qwen3_8_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name == "text/token_embedding":
            continue
        source = model.source(name, quantized)
        if not parameter.projection or name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection")
        ):
            recipe.assign(name, source=source)
            continue
        layer = int(name.split("/")[2]) if name.startswith("text/layers/") else -1
        format = "nvfp4" if "/mlp/" in name and layer < 56 else FP8
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


RECIPES = {
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_6_27b_nvfp4": qwen3_6_27b_nvfp4,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_8_27b_nvfp4": qwen3_8_27b_nvfp4,
    "qwen3_6_35b_a3b": qwen3_6_35b_a3b,
    "qwen3_6_35b_a3b_exl3": qwen3_6_35b_a3b_exl3,
    "qwen3_6_35b_a3b_exl3_banks": qwen3_6_35b_a3b_exl3_banks,
    "qwen3_8_27b_exl3_dense": qwen3_8_27b_exl3_dense,
}
