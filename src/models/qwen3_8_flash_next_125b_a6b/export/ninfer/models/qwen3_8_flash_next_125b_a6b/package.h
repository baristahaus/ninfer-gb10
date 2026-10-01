#pragma once

#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/request.h"
#include "runtime/contract/resources.h"
#include <ninfer/models/qwen3_8_flash_next/frontend.h>
#include <ninfer/models/qwen3_8_flash_next/runtime.h>

#include <cstdint>
#include <memory>
#include <string_view>

namespace ninfer {
struct DeviceContext;

namespace artifact {
class Binder;
class Reader;
class MaterializedArtifact;
struct MaterializationPlan;
} // namespace artifact

namespace models::qwen3_8_flash_next_125b_a6b {

struct Package;

namespace detail {
struct Variant;
enum class WeightsProfile : std::uint8_t { Nvfp4, Nvfp4Bf16Ple };
using Frontend        = qwen3_8_flash_next::Frontend;
using PreparedPrompt  = qwen3_8_flash_next::PreparedPrompt;
using OutputSession   = qwen3_8_flash_next::OutputSession;
using PublishedOutput = qwen3_8_flash_next::PublishedOutput;

class LoadPlan {
public:
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    ~LoadPlan();
    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;
    [[nodiscard]] artifact::MaterializationPlan& materialization();
private:
    class Impl;
    explicit LoadPlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend struct qwen3_8_flash_next_125b_a6b::Package;
};

class LoadedModel {
public:
    ~LoadedModel();
    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
private:
    class Impl;
    explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend struct qwen3_8_flash_next_125b_a6b::Package;
};
} // namespace detail

struct Package {
    static constexpr std::string_view model_id   = "qwen3.8-flash-next-125b-a6b";
    static constexpr std::string_view target_key = "qwen3_8_flash_next_125b_a6b";
    using WeightsProfile                         = detail::WeightsProfile;
    using LoadPlan                               = detail::LoadPlan;
    using LoadedModel                            = detail::LoadedModel;
    using Frontend                               = detail::Frontend;
    using PreparedPrompt                         = detail::PreparedPrompt;
    using OutputSession                          = detail::OutputSession;
    using PublishedOutput                        = detail::PublishedOutput;
    using SequencePlanner         = qwen3_8_flash_next::SequencePlanner<detail::Variant>;
    using SequencePlan            = qwen3_8_flash_next::SequencePlan<detail::Variant>;
    using RequestBasePlan         = qwen3_8_flash_next::RequestBasePlan<detail::Variant>;
    using AdmissionCandidate      = qwen3_8_flash_next::AdmissionCandidate<detail::Variant>;
    using ResourcePlan            = qwen3_8_flash_next::ResourcePlan<detail::Variant>;
    using PersistentBackfillProof = qwen3_8_flash_next::PersistentBackfillProof<detail::Variant>;
    using SequenceHandle          = qwen3_8_flash_next::SequenceHandle<detail::Variant>;
    using ContinuationHandle      = qwen3_8_flash_next::ContinuationHandle<detail::Variant>;
    using SharedPrefixHandle      = qwen3_8_flash_next::SharedPrefixHandle<detail::Variant>;
    using CaptureOffer            = qwen3_8_flash_next::CaptureOffer<detail::Variant>;
    using CacheSessionKey         = qwen3_8_flash_next::PreparedSessionKey;
    using ContinuationSummary     = qwen3_8_flash_next::ContinuationSummary;
    using SharedPrefixSummary     = qwen3_8_flash_next::SharedPrefixSummary;
    using PressurePlanningSession = qwen3_8_flash_next::PressurePlanningSession<detail::Variant>;
    using PressureTargetHandle    = qwen3_8_flash_next::PressureTargetHandle;
    using AssessedPressureTarget  = qwen3_8_flash_next::AssessedPressureTarget<detail::Variant>;
    using CapturePressurePlan     = qwen3_8_flash_next::CapturePressurePlan<detail::Variant>;
    using MaterializationResult   = qwen3_8_flash_next::MaterializationResult<detail::Variant>;
    using ContextTransactionProgress =
        qwen3_8_flash_next::ContextTransactionProgress<detail::Variant>;
    using CaptureAssessment   = qwen3_8_flash_next::CaptureAssessment;
    using ActiveCaptureResult = qwen3_8_flash_next::ActiveCaptureResult<detail::Variant>;
    using PendingBatch        = qwen3_8_flash_next::PendingBatch<detail::Variant>;
    using StartResult         = qwen3_8_flash_next::StartResult<detail::Variant>;
    using PrefillProgress     = qwen3_8_flash_next::PrefillProgress<detail::Variant>;
    using CommitResult        = qwen3_8_flash_next::CommitResult<detail::Variant>;
    using DiscardResult       = qwen3_8_flash_next::DiscardResult<detail::Variant>;
    using FinishResult        = qwen3_8_flash_next::FinishResult<detail::Variant>;
    using AbortResult         = qwen3_8_flash_next::AbortResult<detail::Variant>;
    using ReleaseResult       = qwen3_8_flash_next::ReleaseResult<detail::Variant>;
    using Program             = qwen3_8_flash_next::Program<detail::Variant>;

    [[nodiscard]] static ModelSamplingDefaults sampling_defaults(std::string_view model);
    [[nodiscard]] static WeightsProfile weights_profile(const artifact::Reader&);
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder&, const EngineOptions&,
                                            WeightsProfile);
    [[nodiscard]] static std::unique_ptr<LoadedModel>
    construct_loaded_model(LoadPlan&&, artifact::MaterializedArtifact&&);
    [[nodiscard]] static Frontend make_frontend(const LoadedModel&, const EngineOptions&);
    [[nodiscard]] static SequencePlanner make_sequence_planner(DeviceContext&, const EngineOptions&,
                                                               WeightsProfile);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel&, SequencePlan&&, DeviceContext&, const StartupObserver&);
};

} // namespace models::qwen3_8_flash_next_125b_a6b
} // namespace ninfer
