#pragma once

#include "artifact/materializer.h"
#include "ninfer/ops/flash_next_moe.h"
#include "ninfer/ops/flash_next_gdn.h"
#include "ninfer/ops/flash_next_ple.h"
#include "ninfer/ops/flash_next_qsa.h"
#include "ninfer/ops/hyperconnection.h"
#include "models/qwen3_8_flash_next_125b_a6b/impl/load_plan.h"
#include <ninfer/models/qwen3_8_flash_next_125b_a6b/package.h>
#include <ninfer/models/qwen3_8_flash_next/frontend_resources.h>
#include <ninfer/models/qwen3_8_flash_next/model_view.h>
#include <ninfer/models/qwen3_8_flash_next/startup_features.h>
#include <ninfer/models/qwen3_8_flash_next/vision.h>

#include <array>
#include <optional>
#include <span>
#include <utility>

namespace ninfer::models::qwen3_8_flash_next_125b_a6b {

using GdnWeights = ops::FlashNextGdnWeights;

struct FullLayerWeights {
    Tensor input_norm;
    ops::HyperConnectionWeights attention_hc;
    ops::FlashNextQsaWeights projection;
    Tensor query_norm;
    Tensor key_norm;
    Weight output;
    Tensor post_attention_norm;
    bool has_ple = false;
    ops::FlashNextPleWeights ple{};
    ops::HyperConnectionWeights mlp_hc;
    ops::FlashNextMoeWeights post_mixer;
};

struct GdnLayerWeights {
    Tensor input_norm;
    ops::HyperConnectionWeights attention_hc;
    GdnWeights projection;
    Tensor convolution;
    Tensor norm;
    Weight output;
    Tensor post_attention_norm;
    bool has_ple = false;
    ops::FlashNextPleWeights ple{};
    ops::HyperConnectionWeights mlp_hc;
    ops::FlashNextMoeWeights post_mixer;
};

struct FinalHyperConnectionWeights {
    Tensor norm;
    Weight down;
    Weight up;
};

struct MtpWeights {
    Weight input_projection;
    Tensor embedding_norm;
    Tensor hidden_norm;
    Weight embedding_projection;
    Weight hidden_projection;
    Tensor input_norm;
    ops::HyperConnectionWeights attention_hc;
    ops::FlashNextQsaWeights attention;
    Tensor query_norm;
    Tensor key_norm;
    Weight output;
    Tensor post_attention_norm;
    ops::HyperConnectionWeights mlp_hc;
    ops::FlashNextMoeWeights moe;
    ops::FlashNextMoeWeights post_mixer;
    FinalHyperConnectionWeights final_hc;
    Tensor final_norm;
};

using EmptyDFlashWeights = qwen3_8_flash_next::DFlashWeights<0>;

struct OptimizedProposalWeights {
    Weight head;
    Tensor token_ids;
};

struct RuntimeModelView {
    using FullLayer = FullLayerWeights;
    using GdnLayer  = GdnLayerWeights;
    using MtpLayer  = MtpWeights;
    using DFlash    = EmptyDFlashWeights;

    std::uint64_t weight_bytes = 0;
    Weight token_embedding;
    std::array<FullLayerWeights, kFullAttentionLayers> full_layers;
    std::array<GdnLayerWeights, kGdnLayers> gdn_layers;
    FinalHyperConnectionWeights final_hc;
    Tensor final_norm;
    Weight output_head;
    qwen3_8_flash_next::StartupFeatures features;
    std::optional<OptimizedProposalWeights> optimized_proposal;
    std::optional<MtpWeights> mtp;
    std::optional<EmptyDFlashWeights> dflash;
    std::optional<qwen3_8_flash_next::VisionWeights> vision;
    const artifact::MappedRange* ple_table = nullptr;
};

class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized);

    artifact::MaterializedArtifact backing;
    artifact::MappedRange ple_mapping;
    qwen3_8_flash_next::FrontendResources frontend;
    RuntimeModelView runtime;
};

namespace detail {
class LoadedModel::Impl {
public:
    Impl(WeightsProfile profile, BindingPlan plan, artifact::MaterializedArtifact materialized)
        : weights_profile(profile), data(std::move(plan), std::move(materialized)) {}

    WeightsProfile weights_profile;
    LoadedModelData data;
};
} // namespace detail

} // namespace ninfer::models::qwen3_8_flash_next_125b_a6b
