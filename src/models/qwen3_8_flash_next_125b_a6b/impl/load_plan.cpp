#include "models/qwen3_8_flash_next_125b_a6b/impl/load_plan.h"

#include "artifact/formats.h"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::models::qwen3_8_flash_next_125b_a6b {
namespace {

enum class Placement { Device, Disabled };
using Reference = artifact::ParameterReference;

Reference device(artifact::Binder& binder, std::string_view name, QType format,
                 std::initializer_list<std::uint64_t> shape,
                 Placement placement = Placement::Device) {
    if (placement == Placement::Disabled) { return {}; }
    if (shape.size() >= 2 && format != QType::NVFP4) {
        const auto& use = binder.use(name, std::string(name) + "/input");
        if (!use.activation_policy || !use.auxiliaries.empty()) {
            throw artifact::ArtifactError(std::string(name) + ": invalid A16 weight use");
        }
        // All Flash-Next non-expert projections consume A16. Broader permissions also allow A16.
    }
    return binder.parameter(name, {shape.begin(), shape.end()}, artifact::Residency::Device,
                            format);
}

// Attention, GDN, MoE router/shared and HyperConnection projections are BF16, or row-scaled FP8 in
// an 8-bit projection artifact. The stored format selects the binding; consumers dispatch on the
// bound weight type.
Reference projection(artifact::Binder& binder, std::string_view name,
                     std::initializer_list<std::uint64_t> shape,
                     Placement placement = Placement::Device) {
    if (placement == Placement::Disabled) { return {}; }
    const auto& binding = binder.reader().directory().bindings.at(std::string(name));
    const auto& stored  = binder.reader().directory().tensor(binding.parts.front().object);
    const QType format  = stored.format == artifact::format_name(QType::FP8_E4M3FN_ROW_BF16)
                              ? QType::FP8_E4M3FN_ROW_BF16
                              : QType::BF16;
    return device(binder, name, format, shape, placement);
}

Reference expert_bank(artifact::Binder& binder, std::string_view name,
                      std::initializer_list<std::uint64_t> shape,
                      Placement placement = Placement::Device) {
    if (placement == Placement::Disabled) { return {}; }
    const auto& use = binder.use(name, std::string(name) + "/input");
    if (use.activation_policy != artifact::ActivationPolicy::AllowA4 ||
        use.auxiliaries.size() != 1 || !use.auxiliaries.contains("activation_divisor")) {
        throw artifact::ArtifactError(
            std::string(name) +
            ": NVFP4 experts require AllowA4 and per-expert activation divisors");
    }
    auto reference = device(binder, name, QType::NVFP4, shape, placement);
    if (!reference.binding.whole_object || reference.binding.parts.size() != 1 ||
        binder.reader().directory().tensor(reference.binding.parts.front().object).layout !=
            "expert_block_scale_k16_m128x4_v1") {
        throw artifact::ArtifactError(std::string(name) +
                                      ": unsupported expert-bank representation");
    }
    return reference;
}

Reference expert_input_divisors(artifact::Binder& binder, const std::string& parameter) {
    const auto& use = binder.use(parameter, parameter + "/input");
    return binder.binding(parameter + "/activation_divisor",
                          use.auxiliaries.at("activation_divisor"), {512},
                          artifact::Residency::Device, QType::FP32);
}

HyperConnectionPlan bind_hc(artifact::Binder& binder, const std::string& prefix,
                            Placement placement = Placement::Device) {
    return {
        .block_inject = device(binder, prefix + "block_inject_weight.weight", QType::BF16,
                               {4, 10240}, placement),
        .norm         = device(binder, prefix + "hc_norm.weight", QType::BF16, {10240}, placement),
        .down = projection(binder, prefix + "input_mix_weight_down.weight", {320, 10240}, placement),
        .up   = projection(binder, prefix + "input_mix_weight_up.weight", {10240, 320}, placement),
    };
}

FinalHyperConnectionPlan bind_final_hc(artifact::Binder& binder, const std::string& prefix,
                                       Placement placement = Placement::Device) {
    return {
        .norm = device(binder, prefix + "hc_norm.weight", QType::BF16, {10240}, placement),
        .down = device(binder, prefix + "input_mix_weight_down.weight", QType::BF16, {320, 10240},
                       placement),
        .up   = device(binder, prefix + "input_mix_weight_up.weight", QType::BF16, {10240, 320},
                       placement),
    };
}

MoePlan bind_main_moe(artifact::Binder& binder, const std::string& prefix) {
    return {
        .router      = projection(binder, prefix + "gate.weight", {512, 2560}),
        .shared_gate = projection(binder, prefix + "shared_expert.gate_proj.weight", {640, 2560}),
        .shared_up   = projection(binder, prefix + "shared_expert.up_proj.weight", {640, 2560}),
        .shared_down =
            device(binder, prefix + "shared_expert.down_proj.weight", QType::BF16, {2560, 640}),
        .shared_scale =
            device(binder, prefix + "shared_expert_gate.weight", QType::BF16, {1, 2560}),
        .routed_gate_up = expert_bank(binder, prefix + "experts.gate_up", {512, 1280, 2560}),
        .routed_gate_up_input_divisors = expert_input_divisors(binder, prefix + "experts.gate_up"),
        .routed_down = expert_bank(binder, prefix + "experts.down", {512, 2560, 640}),
        .routed_down_input_divisors = expert_input_divisors(binder, prefix + "experts.down"),
    };
}

MoePlan bind_mtp_moe(artifact::Binder& binder, const std::string& prefix, Placement placement) {
    return {
        .router      = device(binder, prefix + "gate.weight", QType::BF16, {512, 2560}, placement),
        .shared_gate = device(binder, prefix + "shared_expert.gate_proj.weight", QType::BF16,
                              {640, 2560}, placement),
        .shared_up   = device(binder, prefix + "shared_expert.up_proj.weight", QType::BF16,
                              {640, 2560}, placement),
        .shared_down = device(binder, prefix + "shared_expert.down_proj.weight", QType::BF16,
                              {2560, 640}, placement),
        .shared_scale =
            device(binder, prefix + "shared_expert_gate.weight", QType::BF16, {1, 2560}, placement),
        .routed_gate_up = device(binder, prefix + "experts.gate_up_proj", QType::BF16,
                                 {512, 1280, 2560}, placement),
        .routed_down =
            device(binder, prefix + "experts.down_proj", QType::BF16, {512, 2560, 640}, placement),
    };
}

FullAttentionPlan bind_attention(artifact::Binder& binder, const std::string& prefix,
                                 Placement placement = Placement::Device) {
    return {
        .query_gate =
            projection(binder, prefix + "q_proj.weight", {12288, 2560}, placement),
        .key    = projection(binder, prefix + "k_proj.weight", {512, 2560}, placement),
        .value  = projection(binder, prefix + "v_proj.weight", {512, 2560}, placement),
        .output = projection(binder, prefix + "o_proj.weight", {2560, 6144}, placement),
        .query_norm      = device(binder, prefix + "q_norm.weight", QType::BF16, {256}, placement),
        .key_norm        = device(binder, prefix + "k_norm.weight", QType::BF16, {256}, placement),
        .index_query_key = device(binder, prefix + "indexer.index_qk_proj.weight", QType::BF16,
                                  {640, 2560}, placement),
        .index_query_norm =
            device(binder, prefix + "indexer.q_layernorm.weight", QType::BF16, {128}, placement),
        .index_key_norm =
            device(binder, prefix + "indexer.k_layernorm.weight", QType::BF16, {128}, placement),
    };
}

GdnPlan bind_gdn(artifact::Binder& binder, const std::string& prefix) {
    return {
        .a_log        = device(binder, prefix + "A_log", QType::FP32, {48}),
        .dt_bias      = device(binder, prefix + "dt_bias", QType::FP32, {48}),
        .convolution  = device(binder, prefix + "conv1d.weight", QType::BF16, {4, 10240}),
        .a_projection = device(binder, prefix + "in_proj_a.weight", QType::BF16, {48, 2560}),
        .b_projection = device(binder, prefix + "in_proj_b.weight", QType::BF16, {48, 2560}),
        .query_key_value =
            projection(binder, prefix + "in_proj_qkv.weight", {10240, 2560}),
        .output_gate = projection(binder, prefix + "in_proj_z.weight", {6144, 2560}),
        .norm        = device(binder, prefix + "norm.weight", QType::BF16, {128}),
        .output      = projection(binder, prefix + "out_proj.weight", {2560, 6144}),
    };
}

Reference bind_ple_table(artifact::Binder& binder, const std::string& name) {
    const auto& binding = binder.reader().directory().bindings.at(name);
    const artifact::Shape shape{320001536, 160};
    if (!binding.whole_object || binding.parts.size() != 1) {
        throw artifact::ArtifactError("PLE requires a complete raw table");
    }
    const auto& object = binder.reader().directory().tensor(binding.parts.front().object);
    binder.reader().validate_object(binding.parts.front().object);
    if (object.shape != shape ||
        (object.format != "fp8_e4m3fn" && object.format != "bf16") ||
        object.layout != "contiguous_le_v1") {
        throw artifact::ArtifactError("PLE table representation is invalid");
    }
    return {name, shape, binding, artifact::Residency::Host};
}

PlePlan bind_ple(artifact::Binder& binder, const std::string& prefix) {
    const auto& binding = binder.reader().directory().bindings.at(
        prefix + "ple_embedding.ngram_embedding.weight");
    const auto& table = binder.reader().directory().tensor(binding.parts.front().object);
    PlePlan out{
        .convolution      = device(binder, prefix + "conv1d.weight", QType::BF16, {4, 10240}),
        .key_projection   = device(binder, prefix + "key_proj.weight", QType::BF16, {10240, 2560}),
        .convolution_norm = device(binder, prefix + "norm_conv.weight", QType::BF16, {10240}),
        .key_norm         = device(binder, prefix + "norm_key.weight", QType::BF16, {10240}),
        .query_norm       = device(binder, prefix + "norm_query.weight", QType::BF16, {10240}),
        .value_projection = device(binder, prefix + "value_proj.weight", QType::BF16, {2560, 2560}),
        .embedding_scale = table.format == "fp8_e4m3fn"
            ? std::make_optional(device(binder, prefix +
                "ple_embedding.ngram_embedding.weight_scale", QType::BF16, {1}))
            : std::nullopt,
    };
    out.embedding_table = bind_ple_table(binder, prefix + "ple_embedding.ngram_embedding.weight");
    return out;
}

VisionPlan bind_vision(artifact::Binder& binder, Placement placement) {
    VisionPlan out;
    out.patch_embedding =
        device(binder, "vision/patch_embedding", QType::Q6_G64_FP16, {1152, 1536}, placement);
    out.patch_embedding_bias =
        device(binder, "vision/patch_embedding_bias", QType::BF16, {1152}, placement);
    out.position_embedding =
        device(binder, "vision/position_embedding", QType::BF16, {2304, 1152}, placement);
    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        const std::string p = "vision/layers/" + std::to_string(layer) + "/";
        auto& target        = out.layers[layer];
        target.qkv =
            device(binder, p + "attention/qkv", QType::Q4_G64_FP16, {3456, 1152}, placement);
        target.qkv_bias = device(binder, p + "attention/qkv_bias", QType::BF16, {3456}, placement);
        target.output =
            device(binder, p + "attention/output", QType::Q5_G64_FP16, {1152, 1152}, placement);
        target.output_bias =
            device(binder, p + "attention/output_bias", QType::BF16, {1152}, placement);
        target.fc1 = device(binder, p + "mlp/fc1", QType::Q4_G64_FP16, {4304, 1152}, placement);
        target.fc1_bias = device(binder, p + "mlp/fc1_bias", QType::BF16, {4304}, placement);
        target.fc2 = device(binder, p + "mlp/fc2", QType::Q5_G64_FP16, {1152, 4304}, placement);
        target.fc2_bias     = device(binder, p + "mlp/fc2_bias", QType::BF16, {1152}, placement);
        target.norm1_weight = device(binder, p + "norm1/weight", QType::BF16, {1152}, placement);
        target.norm1_bias   = device(binder, p + "norm1/bias", QType::BF16, {1152}, placement);
        target.norm2_weight = device(binder, p + "norm2/weight", QType::BF16, {1152}, placement);
        target.norm2_bias   = device(binder, p + "norm2/bias", QType::BF16, {1152}, placement);
    }
    out.merger_fc1 =
        device(binder, "vision/merger/fc1", QType::Q8_G32_FP16, {4608, 4608}, placement);
    out.merger_fc1_bias = device(binder, "vision/merger/fc1_bias", QType::BF16, {4608}, placement);
    out.merger_fc2 =
        device(binder, "vision/merger/fc2", QType::Q8_G32_FP16, {2560, 4608}, placement);
    out.merger_fc2_bias = device(binder, "vision/merger/fc2_bias", QType::BF16, {2560}, placement);
    out.merger_norm_weight =
        device(binder, "vision/merger/norm/weight", QType::BF16, {1152}, placement);
    out.merger_norm_bias =
        device(binder, "vision/merger/norm/bias", QType::BF16, {1152}, placement);
    return out;
}

} // namespace

ArtifactLoadPlan plan_artifact(artifact::Binder& binder,
                               qwen3_8_flash_next::StartupFeatures features) {
    static constexpr std::array<std::string_view, 6> resources = {
        "frontend/tokenizer.json",           "frontend/tokenizer_config.json",
        "frontend/chat_template.jinja",      "frontend/generation_config.json",
        "frontend/preprocessor_config.json", "frontend/video_preprocessor_config.json",
    };
    const auto& directory = binder.reader().directory();
    if (features.mtp()) {
        const auto& mtp = directory.component("mtp");
        if (mtp.target != "text" || mtp.config.value("architectures", artifact::Json{}) !=
                                        artifact::Json::array({"Qwen3_8FlashNextMTP"})) {
            throw artifact::ArtifactError("Flash-Next requires its matching MTP component");
        }
    }
    if (features.vision) {
        const auto& vision = directory.component("vision");
        if (vision.target != "text" ||
            vision.config.value("model_type", "") != "qwen3_8_flash_next_vision") {
            throw artifact::ArtifactError("Flash-Next requires its matching Vision component");
        }
    }
    if (features.optimized_proposal()) {
        const auto& proposal = directory.component("text").proposal;
        if (!proposal || !proposal->indexed || proposal->rows != 147456) {
            throw artifact::ArtifactError(
                "Flash-Next optimized proposal requires 147456 indexed rows");
        }
    }
    ArtifactLoadPlan out;
    out.bindings.features = features;
    for (std::size_t i = 0; i < resources.size(); ++i) {
        if (i < 4 || features.vision) {
            out.bindings.frontend[i] =
                binder.resource(i < 4 ? "text" : "vision", resources[i].substr(9));
        }
    }
    out.bindings.token_embedding =
        device(binder, "model.language_model.embed_tokens.weight", QType::BF16, {248320, 2560});
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        auto& target             = out.bindings.text_layers[layer];
        const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
        target.attention_hc      = bind_hc(binder, prefix + "attn_hyper_connection.");
        target.is_full_attention = layer >= 3 && (layer - 3) % 4 == 0;
        if (target.is_full_attention) {
            target.attention = bind_attention(binder, prefix + "self_attn.");
        } else {
            target.gdn = bind_gdn(binder, prefix + "linear_attn.");
        }
        target.has_ple = layer == 1;
        if (target.has_ple) {
            target.ple          = bind_ple(binder, prefix + "ple.");
            const auto& binding = target.ple.embedding_table.binding;
            const auto& object  = binder.reader().directory().tensor(binding.parts.front().object);
            out.bindings.ple_mapping = binder.reader().map_range(
                object.offset, object.bytes, artifact::MappingAccess::Random);
        }
        target.mlp_hc = bind_hc(binder, prefix + "mlp_hyper_connection.");
        target.moe    = bind_main_moe(binder, prefix + "mlp.");
    }
    out.bindings.final_hc = bind_final_hc(binder, "model.language_model.hyper_connection_mixer.");
    out.bindings.output_head = projection(binder, "lm_head.weight", {248320, 2560});
    const Placement proposal_placement =
        features.optimized_proposal() ? Placement::Device : Placement::Disabled;
    out.bindings.optimized_proposal_head =
        device(binder, "ninfer.optimized_proposal_head.weight", QType::Q4_G64_FP16, {147456, 2560},
               proposal_placement);
    out.bindings.optimized_proposal_token_ids =
        device(binder, "ninfer.optimized_proposal_head.token_ids", QType::INT32, {147456},
               proposal_placement);

    const Placement mtp_placement = features.mtp() ? Placement::Device : Placement::Disabled;
    auto& mtp                     = out.bindings.mtp;
    mtp.embedding_norm =
        device(binder, "mtp.pre_fc_norm_embedding.weight", QType::BF16, {2560}, mtp_placement);
    mtp.hidden_norm =
        device(binder, "mtp.pre_fc_norm_hidden.weight", QType::BF16, {10240}, mtp_placement);
    mtp.embedding_projection =
        device(binder, "mtp.fc_embedding.weight", QType::BF16, {2560, 2560}, mtp_placement);
    mtp.hidden_projection =
        device(binder, "mtp.fc_hidden.weight", QType::BF16, {2560, 2560}, mtp_placement);
    mtp.attention_hc = bind_hc(binder, "mtp.layers.0.attn_hyper_connection.", mtp_placement);
    mtp.attention    = bind_attention(binder, "mtp.layers.0.self_attn.", mtp_placement);
    mtp.mlp_hc       = bind_hc(binder, "mtp.layers.0.mlp_hyper_connection.", mtp_placement);
    mtp.moe          = bind_mtp_moe(binder, "mtp.layers.0.mlp.", mtp_placement);
    mtp.final_hc     = bind_final_hc(binder, "mtp.hyper_connection_mixer.", mtp_placement);
    const Placement vision_placement = features.vision ? Placement::Device : Placement::Disabled;
    out.bindings.vision              = bind_vision(binder, vision_placement);

    out.materialization = std::move(binder).finish();
    return out;
}

} // namespace ninfer::models::qwen3_8_flash_next_125b_a6b
