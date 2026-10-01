#include "models/qwen3_8_flash_next_125b_a6b/impl/model.h"

#include "artifact/reader.h"
#include "artifact/views.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::models::qwen3_8_flash_next_125b_a6b {
namespace {

Tensor tensor(const artifact::MaterializedArtifact& artifact,
              const artifact::ParameterReference& handle, std::initializer_list<std::int32_t> shape,
              QType format = QType::BF16) {
    return weight_tensor(artifact::bind_view(handle, artifact), shape);
}

Weight weight(const artifact::MaterializedArtifact& artifact,
              const artifact::ParameterReference& reference, QType, int, int) {
    return native_weight(artifact::bind_view(reference, artifact));
}

const std::byte* data(const artifact::MaterializedArtifact& artifact,
                      const artifact::ParameterReference& reference) {
    const auto view = artifact::bind_view(reference, artifact);
    if (!is_complete_weight(view)) {
        throw std::logic_error("expert bank requires complete storage");
    }
    return view.parts.front().parent->data;
}

Weight bf16(const artifact::MaterializedArtifact& artifact,
            const artifact::ParameterReference& handle, int rows, int columns) {
    return native_weight(artifact::bind_view(handle, artifact));
}

Weight row_view(const Weight& parent, int begin, int rows) {
    if (parent.qtype != QType::BF16 || parent.layout != QuantLayout::Contiguous || begin < 0 ||
        rows <= 0 || begin + rows > parent.n) {
        throw std::logic_error("invalid Flash-Next BF16 row view");
    }
    Weight out = parent;
    out.qdata  = static_cast<const std::byte*>(parent.qdata) +
                 static_cast<std::size_t>(begin) * parent.k * sizeof(std::uint16_t);
    out.n = out.shape[0] = out.padded_shape[0] = rows;
    return out;
}

ops::HyperConnectionWeights hc(const HyperConnectionPlan& plan,
                               const artifact::MaterializedArtifact& artifact) {
    return {
        .norm      = tensor(artifact, plan.norm, {10240}),
        .down      = bf16(artifact, plan.down, 320, 10240),
        .up        = bf16(artifact, plan.up, 10240, 320),
        .injection = bf16(artifact, plan.block_inject, 4, 10240),
    };
}

FinalHyperConnectionWeights final_hc(const FinalHyperConnectionPlan& plan,
                                     const artifact::MaterializedArtifact& artifact) {
    return {
        .norm = tensor(artifact, plan.norm, {10240}),
        .down = bf16(artifact, plan.down, 320, 10240),
        .up   = bf16(artifact, plan.up, 10240, 320),
    };
}

ops::FlashNextQsaWeights attention(const FullAttentionPlan& plan,
                                   const artifact::MaterializedArtifact& artifact) {
    const Weight query_gate = bf16(artifact, plan.query_gate, 12288, 2560);
    const Weight index      = bf16(artifact, plan.index_query_key, 640, 2560);
    return {
        .query_gate       = query_gate,
        .key              = bf16(artifact, plan.key, 512, 2560),
        .value            = bf16(artifact, plan.value, 512, 2560),
        .output           = bf16(artifact, plan.output, 2560, 6144),
        .query_norm       = tensor(artifact, plan.query_norm, {256}),
        .key_norm         = tensor(artifact, plan.key_norm, {256}),
        .index_query      = row_view(index, 0, 512),
        .index_key        = row_view(index, 512, 128),
        .index_query_norm = tensor(artifact, plan.index_query_norm, {128}),
        .index_key_norm   = tensor(artifact, plan.index_key_norm, {128}),
    };
}

GdnWeights gdn(const GdnPlan& plan, const artifact::MaterializedArtifact& artifact) {
    return {
        .a_log           = tensor(artifact, plan.a_log, {48}, QType::FP32),
        .dt_bias         = tensor(artifact, plan.dt_bias, {48}, QType::FP32),
        .convolution     = tensor(artifact, plan.convolution, {10240, 4}),
        .a_projection    = bf16(artifact, plan.a_projection, 48, 2560),
        .b_projection    = bf16(artifact, plan.b_projection, 48, 2560),
        .query_key_value = bf16(artifact, plan.query_key_value, 10240, 2560),
        .output_gate     = bf16(artifact, plan.output_gate, 6144, 2560),
        .norm            = tensor(artifact, plan.norm, {128}),
        .output          = bf16(artifact, plan.output, 2560, 6144),
    };
}

ops::FlashNextExpertBank nvfp4_bank(const artifact::MaterializedArtifact& artifact,
                                    const artifact::ParameterReference& handle,
                                    const artifact::ParameterReference& input_divisors, int rows,
                                    int columns) {
    const std::uint64_t shape[] = {512, static_cast<std::uint64_t>(rows),
                                   static_cast<std::uint64_t>(columns)};
    const auto geometry =
        weight_geometry(QType::NVFP4, QuantLayout::ExpertBlockScaleK16M128x4, shape);
    const auto* base = data(artifact, handle);
    return {
        .codes                 = base,
        .scales                = base + geometry.scale_offset,
        .weight_scale_divisors = reinterpret_cast<const float*>(base + geometry.divisor_offset),
        .input_scale_divisors  = reinterpret_cast<const float*>(data(artifact, input_divisors)),
        .qtype                 = QType::NVFP4,
        .experts               = 512,
        .rows                  = rows,
        .columns               = columns,
    };
}

ops::FlashNextExpertBank bf16_bank(const artifact::MaterializedArtifact& artifact,
                                   const artifact::ParameterReference& handle, int rows,
                                   int columns) {
    return {
        .codes   = data(artifact, handle),
        .qtype   = QType::BF16,
        .experts = 512,
        .rows    = rows,
        .columns = columns,
    };
}

ops::FlashNextMoeWeights moe(const MoePlan& plan, const artifact::MaterializedArtifact& artifact,
                             bool mtp) {
    return {
        .router         = bf16(artifact, plan.router, 512, 2560),
        .shared_gate    = bf16(artifact, plan.shared_gate, 640, 2560),
        .shared_up      = bf16(artifact, plan.shared_up, 640, 2560),
        .shared_down    = bf16(artifact, plan.shared_down, 2560, 640),
        .shared_scale   = bf16(artifact, plan.shared_scale, 1, 2560),
        .routed_gate_up = mtp ? bf16_bank(artifact, plan.routed_gate_up, 1280, 2560)
                              : nvfp4_bank(artifact, plan.routed_gate_up,
                                           plan.routed_gate_up_input_divisors, 1280, 2560),
        .routed_down = mtp ? bf16_bank(artifact, plan.routed_down, 2560, 640)
                           : nvfp4_bank(artifact, plan.routed_down, plan.routed_down_input_divisors,
                                        2560, 640),
    };
}

ops::FlashNextPleWeights ple(const PlePlan& plan, const artifact::MaterializedArtifact& artifact) {
    return {
        .key_projection   = bf16(artifact, plan.key_projection, 10240, 2560),
        .value_projection = bf16(artifact, plan.value_projection, 2560, 2560),
        .key_norm         = tensor(artifact, plan.key_norm, {10240}),
        .query_norm       = tensor(artifact, plan.query_norm, {10240}),
        .convolution_norm = tensor(artifact, plan.convolution_norm, {10240}),
        .convolution      = tensor(artifact, plan.convolution, {10240, 4}),
        .embedding_scale  = plan.embedding_scale
            ? tensor(artifact, *plan.embedding_scale, {1}) : Tensor{},
    };
}

qwen3_8_flash_next::VisionWeights vision(const VisionPlan& plan,
                                         const artifact::MaterializedArtifact& artifact) {

    qwen3_8_flash_next::VisionWeights out;
    auto& common           = out.common;
    common.patch_embedding = weight(artifact, plan.patch_embedding, QType::Q6_G64_FP16, 1152, 1536);
    common.patch_embedding_bias = tensor(artifact, plan.patch_embedding_bias, {1152});
    common.position_embedding   = tensor(artifact, plan.position_embedding, {1152, 2304});
    for (std::size_t layer = 0; layer < common.layers.size(); ++layer) {
        const auto& source  = plan.layers[layer];
        auto& target        = common.layers[layer];
        target.qkv          = weight(artifact, source.qkv, QType::Q4_G64_FP16, 3456, 1152);
        target.qkv_bias     = tensor(artifact, source.qkv_bias, {3456});
        target.output       = weight(artifact, source.output, QType::Q5_G64_FP16, 1152, 1152);
        target.output_bias  = tensor(artifact, source.output_bias, {1152});
        target.fc1          = weight(artifact, source.fc1, QType::Q4_G64_FP16, 4304, 1152);
        target.fc1_bias     = tensor(artifact, source.fc1_bias, {4304});
        target.fc2          = weight(artifact, source.fc2, QType::Q5_G64_FP16, 1152, 4304);
        target.fc2_bias     = tensor(artifact, source.fc2_bias, {1152});
        target.norm1_weight = tensor(artifact, source.norm1_weight, {1152});
        target.norm1_bias   = tensor(artifact, source.norm1_bias, {1152});
        target.norm2_weight = tensor(artifact, source.norm2_weight, {1152});
        target.norm2_bias   = tensor(artifact, source.norm2_bias, {1152});
    }
    common.merger_fc1         = weight(artifact, plan.merger_fc1, QType::Q8_G32_FP16, 4608, 4608);
    common.merger_fc1_bias    = tensor(artifact, plan.merger_fc1_bias, {4608});
    common.merger_norm_weight = tensor(artifact, plan.merger_norm_weight, {1152});
    common.merger_norm_bias   = tensor(artifact, plan.merger_norm_bias, {1152});
    out.merger_fc2            = weight(artifact, plan.merger_fc2, QType::Q8_G32_FP16, 2560, 4608);
    out.merger_fc2_bias       = tensor(artifact, plan.merger_fc2_bias, {2560});
    return out;
}

qwen3_8_flash_next::FrontendResources
take_frontend(artifact::MaterializedArtifact& artifact,
              const std::array<artifact::ObjectHandle, 6>& plan) {
    auto take = [&](std::size_t index) {
        if (plan[index].index == std::numeric_limits<std::size_t>::max()) { return std::string{}; }
        const auto bytes = artifact.host_bytes(plan[index]);
        return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    return {.tokenizer_json                 = take(0),
            .tokenizer_config_json          = take(1),
            .chat_template_jinja            = take(2),
            .generation_config_json         = take(3),
            .preprocessor_config_json       = take(4),
            .video_preprocessor_config_json = take(5)};
}

} // namespace

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)), ple_mapping(std::move(plan.ple_mapping)),
      frontend(take_frontend(backing, plan.frontend)) {
    runtime.weight_bytes    = backing.stats().device_capacity_bytes;
    runtime.token_embedding = bf16(backing, plan.token_embedding, 248320, 2560);
    std::size_t full_index  = 0;
    std::size_t gdn_index   = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (source.is_full_attention) {
            FullLayerWeights& out = runtime.full_layers.at(full_index++);
            out.attention_hc      = hc(source.attention_hc, backing);
            out.projection        = attention(source.attention, backing);
            out.query_norm        = out.projection.query_norm;
            out.key_norm          = out.projection.key_norm;
            out.output            = out.projection.output;
            out.has_ple           = source.has_ple;
            if (source.has_ple) { out.ple = ple(source.ple, backing); }
            out.mlp_hc     = hc(source.mlp_hc, backing);
            out.post_mixer = moe(source.moe, backing, false);
        } else {
            GdnLayerWeights& out = runtime.gdn_layers.at(gdn_index++);
            out.attention_hc     = hc(source.attention_hc, backing);
            out.projection       = gdn(source.gdn, backing);
            out.convolution      = out.projection.convolution;
            out.norm             = out.projection.norm;
            out.output           = out.projection.output;
            out.has_ple          = source.has_ple;
            if (source.has_ple) { out.ple = ple(source.ple, backing); }
            out.mlp_hc     = hc(source.mlp_hc, backing);
            out.post_mixer = moe(source.moe, backing, false);
        }
    }
    runtime.final_hc    = final_hc(plan.final_hc, backing);
    runtime.output_head = bf16(backing, plan.output_head, 248320, 2560);
    runtime.features    = plan.features;
    if (plan.features.optimized_proposal()) {
        auto& proposal = runtime.optimized_proposal.emplace();
        proposal.head =
            weight(backing, plan.optimized_proposal_head, QType::Q4_G64_FP16, 147456, 2560);
        proposal.token_ids = tensor(backing, plan.optimized_proposal_token_ids, {147456});
    }
    if (plan.features.mtp()) {
        const MtpPlan& source = plan.mtp;
        runtime.mtp           = MtpWeights{
            .embedding_norm       = tensor(backing, source.embedding_norm, {2560}),
            .hidden_norm          = tensor(backing, source.hidden_norm, {10240}),
            .embedding_projection = bf16(backing, source.embedding_projection, 2560, 2560),
            .hidden_projection    = bf16(backing, source.hidden_projection, 2560, 2560),
            .input_norm           = tensor(backing, source.attention_hc.norm, {10240}),
            .attention_hc         = hc(source.attention_hc, backing),
            .attention            = attention(source.attention, backing),
            .query_norm           = tensor(backing, source.attention.query_norm, {256}),
            .key_norm             = tensor(backing, source.attention.key_norm, {256}),
            .output               = bf16(backing, source.attention.output, 2560, 6144),
            .mlp_hc               = hc(source.mlp_hc, backing),
            .moe                  = moe(source.moe, backing, true),
            .post_mixer           = moe(source.moe, backing, true),
            .final_hc             = final_hc(source.final_hc, backing),
        };
    }
    if (plan.features.vision) { runtime.vision = vision(plan.vision, backing); }
    runtime.ple_table = &ple_mapping;
    runtime.ple_table_dtype = plan.text_layers[1].ple.embedding_scale
        ? DType::FP8_E4M3FN : DType::BF16;
}

} // namespace ninfer::models::qwen3_8_flash_next_125b_a6b
