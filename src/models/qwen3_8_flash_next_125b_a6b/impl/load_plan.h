#pragma once

#include "artifact/binder.h"
#include <ninfer/models/qwen3_8_flash_next/startup_features.h>

#include <array>
#include <cstddef>
#include <optional>

namespace ninfer::models::qwen3_8_flash_next_125b_a6b {

inline constexpr char kModelId[]   = "qwen3.8-flash-next-125b-a6b";
inline constexpr char kWeightsId[] = "nvfp4";
inline constexpr char kPleTableName[] =
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight";
inline constexpr std::size_t kTextLayers          = 48;
inline constexpr std::size_t kFullAttentionLayers = 12;
inline constexpr std::size_t kGdnLayers           = 36;

struct HyperConnectionPlan {
    artifact::ParameterReference block_inject;
    artifact::ParameterReference norm;
    artifact::ParameterReference down;
    artifact::ParameterReference up;
};

struct MoePlan {
    artifact::ParameterReference router;
    artifact::ParameterReference shared_gate;
    artifact::ParameterReference shared_up;
    artifact::ParameterReference shared_down;
    artifact::ParameterReference shared_scale;
    artifact::ParameterReference routed_gate_up;
    artifact::ParameterReference routed_gate_up_input_divisors;
    artifact::ParameterReference routed_down;
    artifact::ParameterReference routed_down_input_divisors;
    // Main banks are NVFP4. The MTP bank is BF16 unless the artifact stores NVFP4 drafter experts.
    bool nvfp4_experts = true;
};

struct FullAttentionPlan {
    artifact::ParameterReference query_gate;
    artifact::ParameterReference key;
    artifact::ParameterReference value;
    artifact::ParameterReference output;
    artifact::ParameterReference query_norm;
    artifact::ParameterReference key_norm;
    artifact::ParameterReference index_query_key;
    artifact::ParameterReference index_query_norm;
    artifact::ParameterReference index_key_norm;
};

struct GdnPlan {
    artifact::ParameterReference a_log;
    artifact::ParameterReference dt_bias;
    artifact::ParameterReference convolution;
    artifact::ParameterReference a_projection;
    artifact::ParameterReference b_projection;
    artifact::ParameterReference query_key_value;
    artifact::ParameterReference output_gate;
    artifact::ParameterReference norm;
    artifact::ParameterReference output;
};

struct PlePlan {
    artifact::ParameterReference convolution;
    artifact::ParameterReference key_projection;
    artifact::ParameterReference convolution_norm;
    artifact::ParameterReference key_norm;
    artifact::ParameterReference query_norm;
    artifact::ParameterReference value_projection;
    std::optional<artifact::ParameterReference> embedding_scale;
    artifact::ParameterReference embedding_table;
};

struct TextLayerPlan {
    HyperConnectionPlan attention_hc;
    FullAttentionPlan attention{};
    GdnPlan gdn{};
    bool is_full_attention = false;
    bool has_ple           = false;
    PlePlan ple{};
    HyperConnectionPlan mlp_hc;
    MoePlan moe;
};

struct FinalHyperConnectionPlan {
    artifact::ParameterReference norm;
    artifact::ParameterReference down;
    artifact::ParameterReference up;
};

struct MtpPlan {
    artifact::ParameterReference embedding_norm;
    artifact::ParameterReference hidden_norm;
    artifact::ParameterReference embedding_projection;
    artifact::ParameterReference hidden_projection;
    HyperConnectionPlan attention_hc;
    FullAttentionPlan attention;
    HyperConnectionPlan mlp_hc;
    MoePlan moe;
    FinalHyperConnectionPlan final_hc;
};

struct VisionPlan {
    artifact::ParameterReference patch_embedding;
    artifact::ParameterReference patch_embedding_bias;
    artifact::ParameterReference position_embedding;

    struct Layer {
        artifact::ParameterReference qkv;
        artifact::ParameterReference qkv_bias;
        artifact::ParameterReference output;
        artifact::ParameterReference output_bias;
        artifact::ParameterReference fc1;
        artifact::ParameterReference fc1_bias;
        artifact::ParameterReference fc2;
        artifact::ParameterReference fc2_bias;
        artifact::ParameterReference norm1_weight;
        artifact::ParameterReference norm1_bias;
        artifact::ParameterReference norm2_weight;
        artifact::ParameterReference norm2_bias;
    };

    std::array<Layer, 27> layers;
    artifact::ParameterReference merger_fc1;
    artifact::ParameterReference merger_fc1_bias;
    artifact::ParameterReference merger_fc2;
    artifact::ParameterReference merger_fc2_bias;
    artifact::ParameterReference merger_norm_weight;
    artifact::ParameterReference merger_norm_bias;
};

struct BindingPlan {
    qwen3_8_flash_next::StartupFeatures features;
    artifact::MappedRange ple_mapping;
    std::array<artifact::ObjectHandle, 6> frontend;
    artifact::ParameterReference token_embedding;
    std::array<TextLayerPlan, kTextLayers> text_layers;
    FinalHyperConnectionPlan final_hc;
    artifact::ParameterReference output_head;
    artifact::ParameterReference optimized_proposal_head;
    artifact::ParameterReference optimized_proposal_token_ids;
    MtpPlan mtp;
    VisionPlan vision;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MappedRange ple_table;
    artifact::MaterializationPlan materialization;
};

ArtifactLoadPlan plan_artifact(artifact::Binder& binder,
                               qwen3_8_flash_next::StartupFeatures features = {});

} // namespace ninfer::models::qwen3_8_flash_next_125b_a6b
