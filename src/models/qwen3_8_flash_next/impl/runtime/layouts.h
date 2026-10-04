#pragma once
#include "models/qwen3_8_flash_next/impl/runtime/instance.h"
// Qwen3.8 family runtime implementation; instantiated only by exact variants.

#include "core/cyclic_kv_cache.h"
#include "core/dtype.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/tensor.h"
#include <ninfer/models/qwen3_8_flash_next/decoder_state.h>
#include <ninfer/models/qwen3_8_flash_next/round_state.h>
#include <ninfer/models/qwen3_8_flash_next/state_image.h>
#include <ninfer/models/qwen3_8_flash_next/startup_features.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace ninfer::models::qwen3_8_flash_next::detail::NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS {

using TensorLayout                              = TensorRegion;
inline constexpr std::uint32_t kCausalScoreTile = 1024;

struct DFlashPersistentLayout {
    qwen3_8_flash_next::PagedKVCacheLayout full;
    TensorLayout prefill_features;
    TensorLayout prefill_positions;
    TensorLayout pending_features;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept { return full.payload_bytes(); }
};

struct PersistentLayout {
    qwen3_8_flash_next::DecoderStateLayout decoder;
    qwen3_8_flash_next::StateImageDeviceLayout state_images;
    std::optional<GdnReplayRecordLayout> replay_records;
    // MTP: the previous round's records, copied at the round head so the verify can fold them
    // while it overwrites replay_records.
    std::optional<GdnReplayRecordLayout> replay_snapshot;
    std::optional<DFlashPersistentLayout> dflash;
    qwen3_8_flash_next::RoundStateLayout round;
    TensorLayout prefill_hidden;
    std::optional<TensorLayout> mtp_prefill_hidden;
    std::optional<TensorLayout> flash_decode_ple;
    std::optional<TensorLayout> flash_ple_records;
    std::optional<TensorLayout> score_hidden;
    TensorLayout token_counts;
    TensorLayout sampling_config;
    std::size_t bytes            = 0;
    std::size_t kv_payload_bytes = 0;
};

struct VisionWorkspacePlan {
    std::uint32_t max_merged_tokens    = 0;
    std::size_t general_capacity_bytes = 0;
    std::size_t encode_peak_bytes      = 0;
    std::size_t handoff_offset_bytes   = 0;
    std::size_t handoff_capacity_bytes = 0;
    std::size_t capacity_bytes         = 0;
};

struct WorkspacePlan {
    std::size_t text_prefill     = 0;
    std::size_t ordinary_round   = 0;
    std::size_t mtp_prefill      = 0;
    std::size_t mtp_round        = 0;
    std::size_t dflash_context   = 0;
    std::size_t dflash_round     = 0;
    std::size_t causal_score     = 0;
    std::size_t general_capacity = 0;
    std::optional<VisionWorkspacePlan> vision;
    std::size_t capacity = 0;
};

struct SequencePlanningInputs {
    WeightsProfile weights_profile;
    std::uint32_t capacity                 = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    KvCacheStorage kv_storage              = KvCacheStorage::BFloat16;
    ProposalHead proposal_head             = ProposalHead::Full;
    bool prompt_lookup                     = false;
    StartupFeatures features;
    bool use_cuda_graph = true;
    bool causal_scoring = false;
    int device          = 0;
    ContextCacheOptions context_cache;
    // Report a target-model log probability for every published generated token. Fixed at load.
    bool token_logprobs = false;
};

} // namespace ninfer::models::qwen3_8_flash_next::detail::NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS

namespace ninfer::models::qwen3_8_flash_next::detail {

template <>
struct SequencePlanImpl<NINFER_QWEN38_FLASH_NEXT_VARIANT> {
    typename NINFER_QWEN38_FLASH_NEXT_VARIANT::WeightsProfile weights_profile;
    std::uint32_t capacity                 = 0;
    std::uint32_t kv_capacity              = 0;
    std::uint32_t main_page_groups         = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    KvCacheStorage kv_storage              = KvCacheStorage::BFloat16;
    ProposalHead proposal_head             = ProposalHead::Full;
    bool prompt_lookup                     = false;
    StartupFeatures features;
    bool use_cuda_graph = true;
    bool causal_scoring = false;
    int device          = 0;
    ContextCacheOptions context_cache;
    bool token_logprobs = false;
    NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS::PersistentLayout persistent;
    NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS::WorkspacePlan workspace;
    std::size_t graph_allowance_bytes    = 0;
    std::size_t device_reservation_bytes = 0;
};

template <>
struct SequencePlannerImpl<NINFER_QWEN38_FLASH_NEXT_VARIANT> {
    NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS::SequencePlanningInputs inputs;
    runtime::SequenceCapacityCurve curve;
    std::unique_ptr<SequencePlanImpl<NINFER_QWEN38_FLASH_NEXT_VARIANT>> minimum;
};

} // namespace ninfer::models::qwen3_8_flash_next::detail

namespace ninfer::models::qwen3_8_flash_next::detail::NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS {

using SequencePlanImpl = qwen3_8_flash_next::detail::SequencePlanImpl<Variant>;

[[nodiscard]] std::unique_ptr<qwen3_8_flash_next::detail::SequencePlannerImpl<Variant>>
make_sequence_planner_impl(DeviceContext& device, const EngineOptions& options,
                           WeightsProfile weights_profile);
[[nodiscard]] std::unique_ptr<SequencePlanImpl> finalize_sequence_plan_impl(
    std::unique_ptr<qwen3_8_flash_next::detail::SequencePlannerImpl<Variant>> planner,
    std::uint32_t main_page_groups);

} // namespace ninfer::models::qwen3_8_flash_next::detail::NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS
