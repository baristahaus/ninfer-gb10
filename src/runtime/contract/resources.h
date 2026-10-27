#pragma once

#include "core/transfer_work.h"
#include "runtime/contract/request.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

namespace ninfer {
struct KVPageGeometry;
}

namespace ninfer::runtime {

enum class CheckpointRole : std::uint8_t { InputReplay, LongAnchor, SharedPrefix, Continuation };

struct KvPoolId {
    std::uint16_t index                                           = 0;
    friend constexpr bool operator==(KvPoolId, KvPoolId) noexcept = default;
};

// Pool IDs are contiguous and fixed for a Program's lifetime. Labels are diagnostic only.
struct KvPoolDescriptor {
    KvPoolId id;
    std::string_view label;
    const KVPageGeometry* geometry   = nullptr;
    std::uint32_t device_page_groups = 0;
    std::size_t host_page_stride     = 0;
};

// Physical quantities are sampled from stores. Each architecture declares its pool capacity;
// inactive pools have zero capacity. Page groups from different pools never offset each other.
template <std::size_t PoolCapacity>
struct ContextResources {
    static constexpr std::size_t pool_capacity = PoolCapacity;
    std::uint32_t state_slots                  = 0;
    std::array<std::uint32_t, PoolCapacity> kv_page_groups{};
    std::size_t host_bytes = 0;

    [[nodiscard]] std::optional<KvPoolId> first_kv_deficit() const noexcept {
        for (std::size_t i = 0; i < PoolCapacity; ++i) {
            if (kv_page_groups[i]) return KvPoolId{static_cast<std::uint16_t>(i)};
        }
        return std::nullopt;
    }

    [[nodiscard]] std::uint64_t useful(const ContextResources& shortage) const noexcept {
        if (shortage.state_slots) return std::min(state_slots, shortage.state_slots);
        if (const auto pool = shortage.first_kv_deficit()) {
            return std::min(kv_page_groups[pool->index], shortage.kv_page_groups[pool->index]);
        }
        return std::min(host_bytes, shortage.host_bytes);
    }

    [[nodiscard]] std::array<std::uint64_t, PoolCapacity + 2> quantities() const noexcept {
        std::array<std::uint64_t, PoolCapacity + 2> result{};
        result[0] = state_slots;
        for (std::size_t i = 0; i < PoolCapacity; ++i) result[i + 1] = kv_page_groups[i];
        result.back() = host_bytes;
        return result;
    }

    friend bool operator==(const ContextResources&, const ContextResources&) noexcept = default;
};

template <class Resources>
struct ResourceReservation {
    bool reserved = false;
    Resources shortage;

    explicit operator bool() const noexcept { return reserved; }
};

// Exact features for the startup-selected static prefill cost model. They describe only the
// suffix rebuilt after a selected prefix and remain separate from Scheduler service work.
struct PrefillWork {
    std::uint64_t chunks          = 0;
    std::uint64_t tokens          = 0;
    std::uint64_t attention_pairs = 0;
    std::uint64_t vision_items    = 0;
    std::uint64_t vision_patches  = 0;

    [[nodiscard]] friend constexpr bool operator==(PrefillWork, PrefillWork) noexcept = default;
};

// Exact prefill feature definition for a suffix beginning after prefix_tokens. Attention work is
// prefix*suffix + suffix*(suffix+1)/2 and all arithmetic saturates.
[[nodiscard]] inline PrefillWork make_prefill_work(std::uint64_t prefix_tokens,
                                                   std::uint64_t suffix_tokens,
                                                   std::uint64_t vision_items,
                                                   std::uint64_t vision_patches,
                                                   std::uint32_t prefill_chunk) noexcept {
    PrefillWork result;
    result.chunks =
        suffix_tokens == 0 || prefill_chunk == 0 ? 0 : 1U + (suffix_tokens - 1U) / prefill_chunk;
    result.tokens                       = suffix_tokens;
    result.vision_items                 = vision_items;
    result.vision_patches               = vision_patches;
    const unsigned __int128 suffix      = suffix_tokens;
    const unsigned __int128 linear      = static_cast<unsigned __int128>(prefix_tokens) * suffix;
    const unsigned __int128 triangular  = suffix * (suffix + 1U) / 2U;
    constexpr unsigned __int128 maximum = ~static_cast<unsigned __int128>(0);
    const unsigned __int128 attention =
        triangular > maximum - linear ? maximum : linear + triangular;
    result.attention_pairs = attention > std::numeric_limits<std::uint64_t>::max()
                                 ? std::numeric_limits<std::uint64_t>::max()
                                 : static_cast<std::uint64_t>(attention);
    return result;
}

struct ContextTransferResource {
    std::optional<KvPoolId> pool;

    [[nodiscard]] constexpr bool is_state() const noexcept { return !pool; }

    [[nodiscard]] constexpr std::size_t index() const noexcept {
        return pool ? 1U + pool->index : 0;
    }

    friend constexpr bool operator==(ContextTransferResource,
                                     ContextTransferResource) noexcept = default;
};

enum class ContextTransferDirection : std::uint8_t {
    DeviceToHost,
    HostToDevice,
    DeviceToDevice,
};

enum class ContextResourceClass : std::uint8_t {
    State,
    MainKV,
    BackendKV,
};

struct ContextTransferObservation {
    ContextResourceClass resource        = ContextResourceClass::State;
    ContextTransferDirection direction   = ContextTransferDirection::DeviceToHost;
    std::uint64_t units                  = 0; // State images for State; bytes for typed KV.
    std::uint32_t page_count             = 0;
    TransferWork work;
    std::uint64_t elapsed_ns = 0;
};

struct ContextTransferRequirement {
    ContextResourceClass resource      = ContextResourceClass::State;
    ContextTransferDirection direction = ContextTransferDirection::DeviceToHost;
    std::uint64_t units                = 0;
    std::uint32_t page_count           = 0;
    TransferWork work;

    [[nodiscard]] friend constexpr bool operator==(ContextTransferRequirement,
                                                   ContextTransferRequirement) noexcept = default;
};

struct ContextOperationCounts {
    std::uint64_t state_moves            = 0;
    std::uint64_t state_forks            = 0;
    std::uint64_t state_restores         = 0;
    std::uint64_t pressure_spill_pages   = 0;
    std::uint64_t partial_tail_cow_pages = 0;
    std::uint64_t historical_fork_hits   = 0;
};

struct SequenceCapacityCurve {
    std::uint32_t main_page_tokens                   = 0;
    std::uint32_t minimum_main_page_groups           = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::size_t minimum_device_reservation_bytes        = 0;
    std::size_t bytes_per_additional_main_page_group = 0;

    [[nodiscard]] std::size_t reservation_bytes(std::uint32_t main_page_groups) const;
    [[nodiscard]] std::uint32_t resolved_tokens(std::uint32_t main_page_groups) const;
};

struct KvCapacityResolution {
    KvCapacityMode mode                                 = KvCapacityMode::Explicit;
    std::uint32_t main_page_groups                   = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::uint32_t resolved_tokens                       = 0;
    std::size_t minimum_runtime_reservation_bytes       = 0;
    std::size_t bytes_per_additional_main_page_group = 0;
    std::size_t runtime_reservation_bytes               = 0;
    std::size_t available_after_weights_bytes           = 0;
    std::size_t available_after_startup_bytes           = 0;
    std::size_t automatic_headroom_bytes                = 0;
    std::size_t planned_slack_bytes                     = 0;
};


inline constexpr std::size_t kContextTransferDirectionCount = 3;
using CoalescedTransferWork = std::array<TransferWork, kContextTransferDirectionCount>;

// Master-generation contract vocabulary: the pressure, materialization, checkpoint and
// retention planning types the Flash-Next family programs against. Additive to the
// restructured pool-descriptor contract above.
enum class Readiness : std::uint8_t {
    Ready,
    NeedsTransfer,
    TemporarilyBlocked,
    PermanentlyInfeasible,
};

enum class ContextTransactionStatus : std::uint8_t {
    InProgress,
    Published,
    Aborted,
};

enum class ContextTransactionReserveStatus : std::uint8_t {
    Reserved,
    Aborted,
};

struct ContextTransactionInProgress {};

enum class ContextTransactionKind : std::uint8_t {
    Materialization,
    ActiveCapture,
};

enum class PreflightStatus : std::uint8_t {
    Ready,
    StalePolicyState,
    InvariantFailure,
};

enum class CheckpointKind : std::uint8_t {
    SessionEndpoint,
    TurnClosure,
    ResponseReplay,
    SharedStablePrefix,
    LongAnchor,
};

enum class CheckpointScope : std::uint8_t {
    Private,
    Shared,
};

enum class ReplicaResidency : std::uint8_t {
    DeviceOnly,
    HostOnly,
    Both,
};

enum class RetentionClass : std::uint8_t {
    SharedStable,
    LiveSession,
    RecentPrivate,
    Disposable,
};

enum class LogicalOwnerKind : std::uint8_t {
    PrivateContinuation,
    SharedPrefix,
};

struct LogicalOwnerKey {
    LogicalOwnerKind kind = LogicalOwnerKind::PrivateContinuation;
    std::uint64_t id      = 0;

    [[nodiscard]] friend constexpr bool operator==(LogicalOwnerKey,
                                                   LogicalOwnerKey) noexcept = default;
};

struct CatalogCapability {
    LogicalOwnerKey owner;
    std::uint32_t slot       = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t generation = 0;

    [[nodiscard]] friend constexpr bool operator==(CatalogCapability,
                                                   CatalogCapability) noexcept = default;
};

struct PlanningOwnerId {
    std::uint32_t value = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]] friend constexpr bool operator==(PlanningOwnerId,
                                                   PlanningOwnerId) noexcept = default;
};

struct PlanningCandidateId {
    std::uint32_t value = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]] friend constexpr bool operator==(PlanningCandidateId,
                                                   PlanningCandidateId) noexcept = default;
};

struct ProgramResourceRevision {
    std::uint64_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(ProgramResourceRevision,
                                                   ProgramResourceRevision) noexcept = default;
};

// ResourceManager-owned final schedule policy. Program validates and consumes this view while
// sealing the already assessed physical target; ResourcePlan is immutable after that boundary.
struct FinalScheduleIntent {
    std::span<const std::uint32_t> shared_capture_frontiers;
};

enum class PrivateSourceMode : std::uint8_t {
    Retain,
    ConsumeToActive,
};

enum class VictimDisposition : std::uint8_t {
    Retained,
    Evicted,
};

enum class FinishDisposition : std::uint8_t {
    Catalogued,
    Released,
};

struct CheckpointRef {
    CheckpointKind kind    = CheckpointKind::SessionEndpoint;
    std::uint32_t frontier = 0;
    // Singleton checkpoint kinds use zero. LongAnchor uses a nonzero, per-continuation slot.
    std::uint32_t ordinal = 0;

    [[nodiscard]] friend constexpr bool operator==(CheckpointRef, CheckpointRef) noexcept = default;
};

struct Revision {
    std::uint64_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(Revision, Revision) noexcept = default;
};

enum class MaterializationPhysicalStatus : std::uint8_t {
    Feasible,
    Infeasible,
    StructuralInvalid,
};

// Exact, unpriced machine work for one complete materialization projection. Program owns this
// physical fact; the common search runner applies the immutable planning cost model exactly once.
// `optimistic_candidate_transfers` is ordering evidence only and never proves feasibility.
struct MaterializationMachineWork {
    CoalescedTransferWork pressure_transfers;
    CoalescedTransferWork candidate_transfers;
    CoalescedTransferWork optimistic_candidate_transfers;
    PrefillWork remaining_prefill_work;
    std::uint32_t reused_prompt_tokens = 0;

    [[nodiscard]] friend constexpr bool
    operator==(const MaterializationMachineWork&,
               const MaterializationMachineWork&) noexcept = default;
};

// One exact recovery recipe. Program enumerates every supported alternative; pricing policy
// selects the cheapest alternative without changing physical legality or the target graph.
struct CheckpointRecoveryAlternativeWork {
    CoalescedTransferWork transfers;
    PrefillWork prefill;

    [[nodiscard]] friend constexpr bool
    operator==(const CheckpointRecoveryAlternativeWork&,
               const CheckpointRecoveryAlternativeWork&) noexcept = default;
};

struct IdentityMaterializationAssessment {
    MaterializationPhysicalStatus physical_status =
        MaterializationPhysicalStatus::StructuralInvalid;
    PrivateSourceMode source_mode = PrivateSourceMode::ConsumeToActive;
    MaterializationMachineWork machine_work;
    bool pressure_may_change_machine_work = false;
    bool expandable                       = false;
    std::uint64_t projection_work         = 0;
    std::uint64_t assessment_digest       = 0;

    [[nodiscard]] friend constexpr bool
    operator==(const IdentityMaterializationAssessment&,
               const IdentityMaterializationAssessment&) noexcept = default;
};

struct PressureCheckpointRecoveryImpact {
    PlanningOwnerId owner;
    CheckpointRef checkpoint;
    std::span<const CheckpointRecoveryAlternativeWork> target_recovery_work;
    bool survives = true;

    [[nodiscard]] friend constexpr bool
    operator==(const PressureCheckpointRecoveryImpact&,
               const PressureCheckpointRecoveryImpact&) noexcept = default;
};

struct PressureCheckpointOutcome {
    PlanningOwnerId owner;
    CheckpointRef checkpoint;
    bool survives = true;

    [[nodiscard]] friend constexpr bool operator==(PressureCheckpointOutcome,
                                                   PressureCheckpointOutcome) noexcept = default;
};

struct PressureOwnerOutcome {
    PlanningOwnerId owner;
    VictimDisposition disposition     = VictimDisposition::Retained;
    std::uint32_t degradation_units   = 0;
    std::uint32_t dropped_checkpoints = 0;

    [[nodiscard]] friend constexpr bool operator==(const PressureOwnerOutcome&,
                                                   const PressureOwnerOutcome&) noexcept = default;
};

// Cheap, target-neutral ordering evidence for an unassessed pressure target.  This is deliberately
// not a feasibility certificate: only PressureTargetAssessment may admit or seal a target.  The
// Program owns the physical projection and the common planner combines the owner outcomes with its
// retention policy.
struct PressurePhysicalGuidance {
    std::uint32_t unsatisfied_constraints   = 0;
    std::uint32_t estimated_remaining_steps = 0;
    std::uint64_t normalized_residual_q20   = 0;
    // Aggregate byte relief cannot resolve an extent/ordered-stage geometry question.
    bool requires_exact_feedback = false;
};

struct PressureOwnerRecoveryGuidance {
    PlanningOwnerId owner;
    CoalescedTransferWork additional_restore;
};

struct PressureConstructionOptionId {
    std::uint32_t cursor_generation = 0;
    std::uint32_t scan_generation   = 0;
    std::uint32_t index             = 0;
};

// The spans are borrowed from a PressurePlanningSession scratch generation and remain valid only
// until the next session operation.  The common planner folds them immediately into owning values.
struct PressureTargetGuidance {
    PressurePhysicalGuidance physical;
    MaterializationMachineWork estimated_machine_work;
    std::span<const PressureOwnerOutcome> owner_outcomes;
    PlanningCandidateId candidate;
    std::uint32_t stable_target_ordinal = 0;
    std::uint32_t degradation_units     = 0;
    std::uint32_t dropped_checkpoints   = 0;
    PrivateSourceMode source_mode       = PrivateSourceMode::ConsumeToActive;
    std::span<const PressureCheckpointOutcome> checkpoint_changes;
    std::span<const PressureOwnerRecoveryGuidance> recovery_estimates;
    bool recovery_estimate_complete = false;
};

// One resumable scan operation: one owner's successor generation or one option summary.
// Guidance spans expire at the next session call; the ID remains valid until choose/reset.
struct PressureConstructionStep {
    std::optional<PressureTargetGuidance> guidance;
    PressureConstructionOptionId option;
    bool exhausted = false;
};

// The spans are borrowed from a PressurePlanningSession scratch generation and remain valid only
// until the next session mutation. The common planner folds them immediately into owning values.
struct PressureTargetAssessment {
    MaterializationPhysicalStatus physical_status =
        MaterializationPhysicalStatus::StructuralInvalid;
    PrivateSourceMode source_mode = PrivateSourceMode::ConsumeToActive;
    MaterializationMachineWork machine_work;
    std::span<const PressureOwnerOutcome> owner_outcomes;
    std::span<const PressureCheckpointRecoveryImpact> checkpoint_impacts;
    PlanningCandidateId candidate;
    std::uint32_t stable_target_ordinal = 0;
    std::uint32_t degradation_units     = 0;
    std::uint32_t dropped_checkpoints   = 0;
    std::uint64_t projection_work       = 0;
    std::uint64_t assessment_digest     = 0;
    bool expandable                     = false;
    bool root_maximal                   = false;
};

} // namespace ninfer::runtime
