#pragma once

#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/request.h"
#include "runtime/contract/resources.h"
#include "artifact/binder.h"
#include <ninfer/models/qwen3_8_flash_next/frontend.h>
#include <ninfer/models/qwen3_8_flash_next/runtime.h>
#include <ninfer/ops/flash_next_moe.h>

#include <cstdint>
#include <memory>
#include <string_view>

namespace ninfer {
struct DeviceContext;

namespace artifact {
class Binder;
class MaterializedArtifact;
struct MaterializationPlan;
} // namespace artifact

namespace models::qwen3_8_flash_next_125b_a6b {

struct Package;

namespace detail {
struct Variant;
enum class WeightsProfile : std::uint8_t { Nvfp4, Exl3 };
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
    [[nodiscard]] artifact::PreparedResources& materialization();
    [[nodiscard]] std::vector<artifact::ObjectHandle>& device_objects();
    // SP-4 tiered banks: the binder's tier state from the artifact's tier plan; empty for an
    // untiered artifact.
    [[nodiscard]] std::vector<artifact::TierPlanInput>& tier_inputs();
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

    // SP-4 tier-policy support. The tiered expert banks in stable model order: the
    // full-attention layers, then the GDN layers, then the MTP post-mixer when present.
    // A null bank means that layer's MoE is not tiered in this artifact. The counters and
    // residency live in the FlashNextExpertBank fields; the engine applies the policy's
    // swaps via ops::swap_expert_tier on both banks of a layer with the same expert pair.
    [[nodiscard]] int tier_bank_count() const noexcept;
    [[nodiscard]] const ops::FlashNextExpertBank* tier_gate_up_bank(int index) const noexcept;
    [[nodiscard]] const ops::FlashNextExpertBank* tier_down_bank(int index) const noexcept;

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
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder&, const EngineOptions&,
                                            WeightsProfile);
    // The binder transform that binds `rank` of `width` ranks. The placement policy is the model's
    // (impl/shard_rules.h); a caller that loads a device group installs this and binds once per
    // rank, so no layer above the model names a shard axis. Width 1 returns an empty transform,
    // which is the single-device lane's contract.
    [[nodiscard]] static artifact::Binder::ShardTransform shard_transform(int width, int rank);
    [[nodiscard]] static std::unique_ptr<LoadedModel>
    construct_loaded_model(LoadPlan&&, artifact::MaterializedArtifact&&);
    [[nodiscard]] static Frontend make_frontend(const LoadedModel&, const EngineOptions&);
    [[nodiscard]] static SequencePlanner make_sequence_planner(DeviceContext&, const EngineOptions&,
                                                               WeightsProfile);
    [[nodiscard]] static std::unique_ptr<Program>
    // `peer_device` is rank 1's device on the two-GPU lane: the Program then builds rank 1's
    // persistent storage and attaches the KV and StateImage mirrors. Null is the single-device lane.
    create_program(const LoadedModel&, SequencePlan&&, DeviceContext&, const StartupObserver&,
                   DeviceContext* peer_device = nullptr);
};

} // namespace models::qwen3_8_flash_next_125b_a6b
} // namespace ninfer
