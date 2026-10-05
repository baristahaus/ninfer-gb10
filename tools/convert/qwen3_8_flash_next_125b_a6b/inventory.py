"""Closed persistent-object contract for Qwen3.8 Flash-Next 125B-A6B NVFP4."""

from __future__ import annotations

from tools.artifact.schema import TensorSpec


MODEL_ID = "qwen3.8-flash-next-125b-a6b"
WEIGHTS_ID = "nvfp4"
TARGET_KEY = "qwen3_8_flash_next_125b_a6b"

BF16 = "bf16"
FP32 = "fp32"
FP8 = "fp8_e4m3fn"
NVFP4 = "nvfp4"
Q4 = "q4_g64_fp16"
I32 = "int32"
CONTIGUOUS = "contiguous_le_v1"
ROW_SPLIT = "row_split_k128_v1"
EXPERT_NVFP4 = "expert_block_scale_k16_m128x4_v1"
FP8_ROW = "fp8_e4m3fn_row_bf16"
ROW_SCALE = "row_scale_v1"

LAYERS = tuple(range(48))
FULL_ATTENTION_LAYERS = tuple(range(3, 48, 4))
GDN_LAYERS = tuple(layer for layer in LAYERS if layer not in FULL_ATTENTION_LAYERS)
EXPERTS = 512

RESOURCE_SPECS = tuple(
    name
    for name in (
        "frontend/tokenizer.json",
        "frontend/tokenizer_config.json",
        "frontend/chat_template.jinja",
        "frontend/generation_config.json",
        "frontend/preprocessor_config.json",
        "frontend/video_preprocessor_config.json",
    )
)


def direct(name: str, shape: tuple[int, ...], numeric_format: str = BF16) -> TensorSpec:
    return TensorSpec(name, shape, numeric_format, CONTIGUOUS)


def _hyperconnection(prefix: str, *, inject: bool) -> tuple[TensorSpec, ...]:
    specs = [
        direct(prefix + "hc_norm.weight", (10240,)),
        direct(prefix + "input_mix_weight_down.weight", (320, 10240)),
        direct(prefix + "input_mix_weight_up.weight", (10240, 320)),
    ]
    if inject:
        specs.insert(0, direct(prefix + "block_inject_weight.weight", (4, 10240)))
    return tuple(specs)


def _moe(prefix: str) -> tuple[TensorSpec, ...]:
    return (
        direct(prefix + "gate.weight", (512, 2560)),
        direct(prefix + "shared_expert.gate_proj.weight", (640, 2560)),
        direct(prefix + "shared_expert.up_proj.weight", (640, 2560)),
        direct(prefix + "shared_expert.down_proj.weight", (2560, 640)),
        direct(prefix + "shared_expert_gate.weight", (1, 2560)),
        TensorSpec(prefix + "experts.gate_up", (512, 1280, 2560), NVFP4, EXPERT_NVFP4),
        direct(prefix + "experts.gate_up_input_divisors", (512,), FP32),
        TensorSpec(prefix + "experts.down", (512, 2560, 640), NVFP4, EXPERT_NVFP4),
        direct(prefix + "experts.down_input_divisors", (512,), FP32),
    )


def _text_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        direct("model.language_model.embed_tokens.weight", (248320, 2560)),
    ]
    for layer in LAYERS:
        prefix = f"model.language_model.layers.{layer}."
        specs.extend(_hyperconnection(prefix + "attn_hyper_connection.", inject=True))
        if layer in FULL_ATTENTION_LAYERS:
            attention = prefix + "self_attn."
            specs.extend(
                (
                    direct(attention + "q_proj.weight", (12288, 2560)),
                    direct(attention + "k_proj.weight", (512, 2560)),
                    direct(attention + "v_proj.weight", (512, 2560)),
                    direct(attention + "o_proj.weight", (2560, 6144)),
                    direct(attention + "q_norm.weight", (256,)),
                    direct(attention + "k_norm.weight", (256,)),
                    direct(attention + "indexer.index_qk_proj.weight", (640, 2560)),
                    direct(attention + "indexer.q_layernorm.weight", (128,)),
                    direct(attention + "indexer.k_layernorm.weight", (128,)),
                )
            )
        else:
            attention = prefix + "linear_attn."
            specs.extend(
                (
                    # GDN control values are represented as exact FP32 expansions of the
                    # checkpoint BF16 words so the persistent runtime state transition uses the
                    # same FP32 contract as the shared Gated DeltaNet Op.
                    direct(attention + "A_log", (48,), FP32),
                    direct(attention + "dt_bias", (48,), FP32),
                    direct(attention + "conv1d.weight", (4, 10240)),
                    direct(attention + "in_proj_a.weight", (48, 2560)),
                    direct(attention + "in_proj_b.weight", (48, 2560)),
                    direct(attention + "in_proj_qkv.weight", (10240, 2560)),
                    direct(attention + "in_proj_z.weight", (6144, 2560)),
                    direct(attention + "norm.weight", (128,)),
                    direct(attention + "out_proj.weight", (2560, 6144)),
                )
            )
        if layer == 1:
            ple = prefix + "ple."
            specs.extend(
                (
                    direct(ple + "conv1d.weight", (4, 10240)),
                    direct(ple + "key_proj.weight", (10240, 2560)),
                    direct(ple + "norm_conv.weight", (10240,)),
                    direct(ple + "norm_key.weight", (10240,)),
                    direct(ple + "norm_query.weight", (10240,)),
                    direct(ple + "value_proj.weight", (2560, 2560)),
                    direct(ple + "ple_embedding.ngram_embedding.weight_scale", (1,)),
                    direct(
                        ple + "ple_embedding.ngram_embedding.weight",
                        (320001536, 160),
                        FP8,
                    ),
                )
            )
        specs.extend(_hyperconnection(prefix + "mlp_hyper_connection.", inject=True))
        specs.extend(_moe(prefix + "mlp."))
    specs.extend(_hyperconnection("model.language_model.hyper_connection_mixer.", inject=False))
    specs.append(direct("lm_head.weight", (248320, 2560)))
    specs.extend(
        (
            TensorSpec("ninfer.optimized_proposal_head.weight", (147456, 2560), Q4,
                       ROW_SPLIT),
            direct("ninfer.optimized_proposal_head.token_ids", (147456,), I32),
        )
    )
    return tuple(specs)


def _mtp_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        direct("mtp.pre_fc_norm_embedding.weight", (2560,)),
        direct("mtp.pre_fc_norm_hidden.weight", (10240,)),
        direct("mtp.fc_embedding.weight", (2560, 2560)),
        direct("mtp.fc_hidden.weight", (2560, 2560)),
    ]
    specs.extend(_hyperconnection("mtp.layers.0.attn_hyper_connection.", inject=True))
    attention = "mtp.layers.0.self_attn."
    specs.extend(
        (
            direct(attention + "q_proj.weight", (12288, 2560)),
            direct(attention + "k_proj.weight", (512, 2560)),
            direct(attention + "v_proj.weight", (512, 2560)),
            direct(attention + "o_proj.weight", (2560, 6144)),
            direct(attention + "q_norm.weight", (256,)),
            direct(attention + "k_norm.weight", (256,)),
            direct(attention + "indexer.index_qk_proj.weight", (640, 2560)),
            direct(attention + "indexer.q_layernorm.weight", (128,)),
            direct(attention + "indexer.k_layernorm.weight", (128,)),
        )
    )
    specs.extend(_hyperconnection("mtp.layers.0.mlp_hyper_connection.", inject=True))
    moe = "mtp.layers.0.mlp."
    specs.extend(
        (
            direct(moe + "gate.weight", (512, 2560)),
            direct(moe + "shared_expert.gate_proj.weight", (640, 2560)),
            direct(moe + "shared_expert.up_proj.weight", (640, 2560)),
            direct(moe + "shared_expert.down_proj.weight", (2560, 640)),
            direct(moe + "shared_expert_gate.weight", (1, 2560)),
            direct(moe + "experts.gate_up_proj", (512, 1280, 2560)),
            direct(moe + "experts.down_proj", (512, 2560, 640)),
        )
    )
    specs.extend(_hyperconnection("mtp.hyper_connection_mixer.", inject=False))
    return tuple(specs)


Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
W8 = "q8_g32_fp16"

def vision_tensor(name, shape, format):
    return TensorSpec(name, shape, format, CONTIGUOUS if format == BF16 else ROW_SPLIT)

def build_vision_specs(text_width: int) -> tuple[TensorSpec, ...]:
    """Build the Flash-Next Vision inventory."""

    specs: list[TensorSpec] = [
        vision_tensor("vision/patch_embedding", (1152, 1536), Q6),
        vision_tensor("vision/patch_embedding_bias", (1152,), BF16),
        vision_tensor("vision/position_embedding", (2304, 1152), BF16),
    ]

    for layer in range(27):
        prefix = f"vision/layers/{layer}/"
        specs.extend(
            (
                vision_tensor(prefix + "attention/qkv", (3456, 1152), Q4),
                vision_tensor(prefix + "attention/qkv_bias", (3456,), BF16),
                vision_tensor(prefix + "attention/output", (1152, 1152), Q5),
                vision_tensor(prefix + "attention/output_bias", (1152,), BF16),
                vision_tensor(prefix + "mlp/fc1", (4304, 1152), Q4),
                vision_tensor(prefix + "mlp/fc1_bias", (4304,), BF16),
                vision_tensor(prefix + "mlp/fc2", (1152, 4304), Q5),
                vision_tensor(prefix + "mlp/fc2_bias", (1152,), BF16),
                vision_tensor(prefix + "norm1/weight", (1152,), BF16),
                vision_tensor(prefix + "norm1/bias", (1152,), BF16),
                vision_tensor(prefix + "norm2/weight", (1152,), BF16),
                vision_tensor(prefix + "norm2/bias", (1152,), BF16),
            )
        )

    specs.extend(
        (
            vision_tensor("vision/merger/fc1", (4608, 4608), W8),
            vision_tensor("vision/merger/fc1_bias", (4608,), BF16),
            vision_tensor("vision/merger/fc2", (text_width, 4608), W8),
            vision_tensor("vision/merger/fc2_bias", (text_width,), BF16),
            vision_tensor("vision/merger/norm/weight", (1152,), BF16),
            vision_tensor("vision/merger/norm/bias", (1152,), BF16),
        )
    )
    return tuple(specs)


TEXT_TENSOR_SPECS = _text_specs()
MTP_TENSOR_SPECS = _mtp_specs()
VISION_TENSOR_SPECS = build_vision_specs(2560)
TENSOR_SPECS = TEXT_TENSOR_SPECS + MTP_TENSOR_SPECS + VISION_TENSOR_SPECS
OBJECT_SPECS: tuple[str | TensorSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS


# Main-model attention and GDN projections, router and shared-expert gate/up projections and the
# output head, which the 8-bit projection profile stores as row-scaled FP8 (weight-only). MTP,
# indexer, GDN control, HyperConnection, shared-expert down (K = 640) and routed expert weights keep
# their formats.
PROJECTION_NAMES = frozenset(
    [f"model.language_model.layers.{layer}.self_attn.{name}_proj.weight"
     for layer in FULL_ATTENTION_LAYERS for name in ("q", "k", "v", "o")]
    + [f"model.language_model.layers.{layer}.linear_attn.{name}.weight"
       for layer in GDN_LAYERS for name in ("in_proj_qkv", "in_proj_z", "out_proj")]
    + ["lm_head.weight"]
    + [f"model.language_model.layers.{layer}.mlp.{name}.weight"
       for layer in LAYERS
       for name in ("gate", "shared_expert.gate_proj", "shared_expert.up_proj")]
    + [f"model.language_model.layers.{layer}.{site}_hyper_connection.input_mix_weight_{name}.weight"
       for layer in LAYERS for site in ("attn", "mlp") for name in ("down", "up")]
)


def object_specs(ple_format: str = FP8, projection_format: str = BF16) -> tuple[str | TensorSpec, ...]:
    if projection_format not in (BF16, FP8_ROW):
        raise ValueError(f"unsupported projection format: {projection_format}")
    if ple_format == FP8:
        specs = OBJECT_SPECS
    elif ple_format == BF16:
        prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
        specs = tuple(
            direct(spec.id, spec.shape, BF16)
            if isinstance(spec, TensorSpec) and spec.id == prefix + "weight"
            else spec
            for spec in OBJECT_SPECS
            if not isinstance(spec, TensorSpec) or spec.id != prefix + "weight_scale"
        )
    else:
        raise ValueError(f"unsupported PLE format: {ple_format}")
    if projection_format == BF16:
        return specs
    return tuple(
        TensorSpec(spec.id, spec.shape, FP8_ROW, ROW_SCALE)
        if isinstance(spec, TensorSpec) and spec.id in PROJECTION_NAMES
        else spec
        for spec in specs
    )


__all__ = [
    "EXPERTS",
    "FULL_ATTENTION_LAYERS",
    "GDN_LAYERS",
    "LAYERS",
    "MODEL_ID",
    "MTP_TENSOR_SPECS",
    "OBJECT_SPECS",
    "PROJECTION_NAMES",
    "FP8_ROW",
    "object_specs",
    "RESOURCE_SPECS",
    "TARGET_KEY",
    "TENSOR_SPECS",
    "TEXT_TENSOR_SPECS",
    "VISION_TENSOR_SPECS",
    "WEIGHTS_ID",
]
