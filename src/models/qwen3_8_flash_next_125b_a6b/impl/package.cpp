#include <ninfer/models/qwen3_8_flash_next_125b_a6b/package.h>

#include "artifact/reader.h"
#include "models/qwen3_8_flash_next_125b_a6b/impl/load_plan.h"
#include "models/qwen3_8_flash_next_125b_a6b/impl/model.h"
#include "models/qwen3_8_flash_next_125b_a6b/impl/shard_rules.h"
#include "models/qwen3_8_flash_next_125b_a6b/impl/variant.h"

#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_8_flash_next_125b_a6b::detail {

class LoadPlan::Impl {
public:
    Impl(WeightsProfile profile, ArtifactLoadPlan value)
        : weights_profile(profile), plan(std::move(value)) {}

    WeightsProfile weights_profile;
    ArtifactLoadPlan plan;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

artifact::PreparedResources& LoadPlan::materialization() {
    if (!impl_) { throw std::logic_error("Flash-Next load plan is empty"); }
    return impl_->plan.materialization;
}

std::vector<artifact::ObjectHandle>& LoadPlan::device_objects() {
    if (!impl_) { throw std::logic_error("Flash-Next load plan is empty"); }
    return impl_->plan.device_objects;
}

std::vector<artifact::TierPlanInput>& LoadPlan::tier_inputs() {
    if (!impl_) { throw std::logic_error("Flash-Next load plan is empty"); }
    return impl_->plan.tier_inputs;
}

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadedModel::~LoadedModel() = default;

namespace {
// One MoE layer's tiered banks; the variant's non-FlashNext route has no tier state.
const ops::FlashNextMoeWeights* tier_moe(const MoeWeights& moe) noexcept {
    return std::get_if<ops::FlashNextMoeWeights>(&moe);
}
} // namespace

int LoadedModel::tier_bank_count() const noexcept {
    int count = 0;
    for (const auto& layer : impl_->data.runtime.full_layers) {
        if (tier_moe(layer.post_mixer) != nullptr) { ++count; }
    }
    for (const auto& layer : impl_->data.runtime.gdn_layers) {
        if (tier_moe(layer.post_mixer) != nullptr) { ++count; }
    }
    if (impl_->data.runtime.mtp.has_value() && tier_moe(impl_->data.runtime.mtp->post_mixer) != nullptr) {
        ++count;
    }
    return count;
}

const ops::FlashNextExpertBank* LoadedModel::tier_gate_up_bank(const int index) const noexcept {
    int ordinal = 0;
    auto pick = [&ordinal, index](const MoeWeights& moe) noexcept
        -> const ops::FlashNextExpertBank* {
        const auto* weights = tier_moe(moe);
        if (weights == nullptr) { return nullptr; }
        if (ordinal == index) { return &weights->routed_gate_up; }
        ++ordinal;
        return nullptr;
    };
    for (const auto& layer : impl_->data.runtime.full_layers) {
        if (auto* bank = pick(layer.post_mixer); bank != nullptr) { return bank; }
    }
    for (const auto& layer : impl_->data.runtime.gdn_layers) {
        if (auto* bank = pick(layer.post_mixer); bank != nullptr) { return bank; }
    }
    if (impl_->data.runtime.mtp.has_value()) {
        return pick(impl_->data.runtime.mtp->post_mixer);
    }
    return nullptr;
}

const ops::FlashNextExpertBank* LoadedModel::tier_down_bank(const int index) const noexcept {
    int ordinal = 0;
    auto pick = [&ordinal, index](const MoeWeights& moe) noexcept
        -> const ops::FlashNextExpertBank* {
        const auto* weights = tier_moe(moe);
        if (weights == nullptr) { return nullptr; }
        if (ordinal == index) { return &weights->routed_down; }
        ++ordinal;
        return nullptr;
    };
    for (const auto& layer : impl_->data.runtime.full_layers) {
        if (auto* bank = pick(layer.post_mixer); bank != nullptr) { return bank; }
    }
    for (const auto& layer : impl_->data.runtime.gdn_layers) {
        if (auto* bank = pick(layer.post_mixer); bank != nullptr) { return bank; }
    }
    if (impl_->data.runtime.mtp.has_value()) {
        return pick(impl_->data.runtime.mtp->post_mixer);
    }
    return nullptr;
}

} // namespace ninfer::models::qwen3_8_flash_next_125b_a6b::detail

namespace ninfer::models::qwen3_8_flash_next_125b_a6b {
namespace {
constexpr ModelSamplingDefaults kDefaults{
    .thinking     = {.temperature = 1.0F, .top_k = 20, .top_p = 0.95F},
    .non_thinking = {.temperature = 0.7F, .top_k = 20, .top_p = 0.80F, .presence_penalty = 1.5F},
};
} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model != model_id) { throw std::runtime_error("unknown Flash-Next model identity"); }
    return kDefaults;
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile profile) {
    if (profile != WeightsProfile::Nvfp4 && profile != WeightsProfile::Exl3) {
        throw std::logic_error("invalid Flash-Next profile");
    }
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        profile, plan_artifact(binder, qwen3_8_flash_next::startup_features(options))));
}

artifact::Binder::ShardTransform Package::shard_transform(int width, int rank) {
    return flash_next_125b_a6b::shard_transform(width, rank);
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (!plan.impl_) { throw std::invalid_argument("Flash-Next load plan is empty"); }
    auto impl = std::make_unique<LoadedModel::Impl>(
        plan.impl_->weights_profile, std::move(plan.impl_->plan.bindings), std::move(materialized));
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (!model.impl_) { throw std::invalid_argument("Flash-Next loaded model is empty"); }
    return qwen3_8_flash_next::make_frontend(
        model.impl_->data.frontend, {.vision_enabled    = model.impl_->data.runtime.features.vision,
                                     .max_context       = options.max_context,
                                     .media_cache_bytes = options.media_cache_bytes,
                                     .media_live_bytes  = options.media_live_bytes,
                                     .media_preprocess_threads = options.media_preprocess_threads});
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile profile) {
    return qwen3_8_flash_next::make_sequence_planner<detail::Variant>(device, options, profile);
}

std::unique_ptr<Package::Program> Package::create_program(const LoadedModel& model,
                                                          SequencePlan&& plan,
                                                          DeviceContext& device,
                                                          const StartupObserver& observer) {
    if (!model.impl_) { throw std::invalid_argument("Flash-Next loaded model is empty"); }
    return qwen3_8_flash_next::create_program<detail::Variant>(
        model.impl_->data.runtime, model.impl_->weights_profile, std::move(plan), device, observer);
}

} // namespace ninfer::models::qwen3_8_flash_next_125b_a6b
