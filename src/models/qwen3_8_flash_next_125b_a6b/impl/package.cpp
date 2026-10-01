#include <ninfer/models/qwen3_8_flash_next_125b_a6b/package.h>

#include "artifact/reader.h"
#include "models/qwen3_8_flash_next_125b_a6b/impl/load_plan.h"
#include "models/qwen3_8_flash_next_125b_a6b/impl/model.h"
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

artifact::MaterializationPlan& LoadPlan::materialization() {
    if (!impl_) { throw std::logic_error("Flash-Next load plan is empty"); }
    return impl_->plan.materialization;
}

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadedModel::~LoadedModel() = default;

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

Package::WeightsProfile Package::weights_profile(const artifact::Reader& reader) {
    const auto& binding = reader.directory().bindings.at(kPleTableName);
    if (!binding.whole_object || binding.parts.size() != 1) {
        throw artifact::ArtifactError("PLE requires a complete raw table");
    }
    const auto& table = reader.directory().tensor(binding.parts.front().object);
    if (table.format == "fp8_e4m3fn") { return WeightsProfile::Nvfp4; }
    if (table.format == "bf16") { return WeightsProfile::Nvfp4Bf16Ple; }
    throw artifact::ArtifactError("unsupported PLE table format");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile profile) {
    if (profile != weights_profile(binder.reader())) {
        throw std::logic_error("invalid Flash-Next profile");
    }
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        profile, plan_artifact(binder, qwen3_8_flash_next::startup_features(options))));
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
