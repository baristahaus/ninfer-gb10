#pragma once

#include "core/layout.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/target_logprobs.h"
#include "ninfer/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::models::qwen3_8_flash_next {

inline constexpr std::uint32_t kMtpDecodeMaximumDrafts    = 5;
inline constexpr std::uint32_t kMtpDecodeMaximumWidth     = kMtpDecodeMaximumDrafts + 1;
inline constexpr std::uint32_t kDFlashDecodeMaximumDrafts = 15;
inline constexpr std::uint32_t kDFlashDecodeMaximumWidth  = kDFlashDecodeMaximumDrafts + 1;

struct RoundStateSpec {
    std::int32_t hidden = 0;
    // Zero retains the family MTP contract where predictor and sampled hidden
    // have the same width. A positive value enables a distinct predictor state.
    std::int32_t mtp_hidden      = 0;
    std::int32_t output_rows     = 0;
    std::uint32_t batch_capacity = 1;
    std::uint32_t draft_window   = 0;
    bool enable_mtp              = false;
    bool enable_dflash           = false;
};
// Pinned/device transfer format for the single token a prefill step publishes. Prefill keeps its
// own tiny egress rather than a decode frame's, so the probability report travels beside it.
struct PrefillRoundReport {
    float token_logprob = 0.0F;
    std::array<std::int32_t, ops::kMaxReportedLogprobRanks> top_ids{};
    std::array<float, ops::kMaxReportedLogprobRanks> top_logprobs{};
};

// Stable pinned/device transfer format for ordinary decode. The full fixed-size object is copied
// once per round; only its exact-B prefixes are consumed by the model schedule.
struct OrdinaryDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> tokens{};
    std::array<std::int32_t, kMaximumConcurrency> cache_positions{};
    std::array<std::int32_t, kMaximumConcurrency> rope_positions{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> state_source_slots{};
    std::array<std::int32_t, kMaximumConcurrency> state_destination_slots{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
};

struct OrdinaryDecodeEgress {
    std::array<TokenId, kMaximumConcurrency> sampled_tokens{};
    // Target-model probability report for each sampled token, indexed exactly like sampled_tokens.
    // Ordinary decode has one column per lane, so the ingress sampling array is already the
    // per-column array the probability Op needs.
    std::array<float, kMaximumConcurrency> token_logprobs{};
    std::array<std::int32_t, ops::kMaxReportedLogprobRanks * kMaximumConcurrency> top_ids{};
    std::array<float, ops::kMaxReportedLogprobRanks * kMaximumConcurrency> top_logprobs{};
};

// Stable pinned/device transfer formats for concurrent MTP decode. The arrays use the maximum
// product domain; RoundState binds only the configured [K,C] and [K+1,C] prefixes.
struct MtpDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> anchors{};
    std::array<std::int32_t, kMaximumConcurrency> base_frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> remaining_budgets{};
    std::array<std::int32_t, kMaximumConcurrency> current_extents{};
    std::array<std::int32_t, kMaximumConcurrency> target_valid_columns{};
    std::array<TokenId, kMaximumConcurrency * kMtpDecodeMaximumDrafts> current_drafts{};
    std::array<std::int32_t, kMaximumConcurrency * kMtpDecodeMaximumWidth> target_rope_positions{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> mtp_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> state_source_slots{};
    std::array<std::int32_t, kMaximumConcurrency> state_destination_slots{};
    std::array<std::int32_t, kMaximumConcurrency> rope_deltas{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
};

struct MtpDecodeEgress {
    std::array<TokenId, kMaximumConcurrency * kMtpDecodeMaximumWidth> licensed_tokens{};
    std::array<std::int32_t, kMaximumConcurrency> licensed_counts{};
    std::array<std::int32_t, kMaximumConcurrency> accepted_drafts{};
    // Step-major: all B rows for proposal step 0, followed by all B rows for step 1, etc.
    std::array<TokenId, kMaximumConcurrency * kMtpDecodeMaximumDrafts> next_drafts{};
    std::array<std::int32_t, kMaximumConcurrency> next_extents{};
    // Probability report for every verify column, indexed exactly like licensed_tokens; ranks of
    // column g occupy [g*ranks, (g+1)*ranks). Filled only when the engine was loaded with
    // EngineOptions::token_logprobs.
    std::array<float, kMaximumConcurrency * kMtpDecodeMaximumWidth> token_logprobs{};
    std::array<std::int32_t,
               ops::kMaxReportedLogprobRanks * kMaximumConcurrency * kMtpDecodeMaximumWidth>
        top_ids{};
    std::array<float,
               ops::kMaxReportedLogprobRanks * kMaximumConcurrency * kMtpDecodeMaximumWidth>
        top_logprobs{};
};

// Stable pinned/device transfer formats for one exact-B DFlash transaction. The proposal is
// produced and verified in the same round, so no draft state crosses the round boundary.
struct DFlashDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> anchors{};
    std::array<std::int32_t, kMaximumConcurrency> execution_frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> context_frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> proposal_extents{};
    std::array<std::int32_t, kMaximumConcurrency> target_valid_columns{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> dflash_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> active_lanes{};
    std::array<std::int32_t, kMaximumConcurrency> state_source_slots{};
    std::array<std::int32_t, kMaximumConcurrency> state_destination_slots{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
};

struct DFlashDecodeEgress {
    std::array<TokenId, kMaximumConcurrency * kDFlashDecodeMaximumWidth> licensed_tokens{};
    std::array<std::int32_t, kMaximumConcurrency> licensed_counts{};
    std::array<std::int32_t, kMaximumConcurrency> accepted_drafts{};
};

struct OrdinaryDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion logits;
    TensorRegion hidden;
};

struct MtpPrefillStateLayout {
    TensorRegion position;
    TensorRegion ar_hidden;
    TensorRegion draft_tokens;
    TensorRegion target_input_ids;
    TensorRegion target_positions;
};

struct DFlashPrefillStateLayout {
    TensorRegion produced_count;
};

struct MtpDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion verify_ids;
    TensorRegion target_positions;
    TensorRegion target_argmax;
    TensorRegion target_logits;
    TensorRegion target_hidden;
    TensorRegion target_continuation_hidden;
    std::optional<TensorRegion> target_mtp_hidden;
    std::optional<TensorRegion> target_continuation_mtp_hidden;
    TensorRegion proposal_logits;
    TensorRegion alignment_ids;
    TensorRegion alignment_hidden;
    std::optional<TensorRegion> alignment_mtp_hidden;
    std::optional<TensorRegion> alignment_qsa_indices;
    TensorRegion ar_hidden;
    std::optional<TensorRegion> ar_mtp_hidden;
    std::optional<TensorRegion> ar_qsa_indices;
    TensorRegion next_hidden;
    std::optional<TensorRegion> next_mtp_hidden;
    TensorRegion ar_positions;
    TensorRegion ar_rope_positions;
    TensorRegion ar_valid_columns;
};

struct DFlashDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion proposal_ids;
    TensorRegion proposal_positions;
    TensorRegion append_positions;
    TensorRegion append_counts;
    TensorRegion draft_tokens;
    TensorRegion verify_ids;
    TensorRegion target_argmax;
    TensorRegion target_logits;
    TensorRegion target_hidden;
    TensorRegion target_continuation_hidden;
};

struct RoundStateLayout {
    RoundStateSpec spec;
    std::optional<OrdinaryDecodeStateLayout> ordinary;
    TensorRegion token;
    TensorRegion pos;
    TensorRegion rope_pos;
    TensorRegion rope_delta;
    TensorRegion logits;
    TensorRegion text_kv_table_row;
    TensorRegion backend_kv_table_row;
    LayoutRegion report;
    std::optional<MtpPrefillStateLayout> mtp;
    std::optional<DFlashPrefillStateLayout> dflash_prefill;
    std::optional<MtpDecodeStateLayout> mtp_decode;
    std::optional<DFlashDecodeStateLayout> dflash_decode;
    bool complete = false;
};

struct OrdinaryDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor tokens;
    Tensor cache_positions;
    Tensor rope_positions;
    Tensor text_kv_table_rows;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor sampled_tokens;
    Tensor token_logprobs;
    Tensor top_ids;
    Tensor top_logprobs;
    Tensor logits;
    Tensor hidden;

    OrdinaryDecodeState() = default;
    OrdinaryDecodeState(DeviceSpan backing, const OrdinaryDecodeStateLayout& layout,
                        std::uint32_t batch_capacity);
};

// The two planning calls expose one deliberate exact-target extension seam after scalar logits.
// This lets a target retain its schedule-sized prefill activation at the established physical
// address without making that activation part of the family round contract.
[[nodiscard]] RoundStateLayout begin_round_state_layout(LayoutBuilder& builder,
                                                        const RoundStateSpec& spec);
void complete_round_state_layout(LayoutBuilder& builder, RoundStateLayout& layout);

struct MtpPrefillState {
    Tensor position;
    Tensor ar_hidden;
    Tensor draft_tokens;
    Tensor target_input_ids;
    Tensor target_positions;

    MtpPrefillState() = default;
    MtpPrefillState(DeviceSpan backing, const MtpPrefillStateLayout& layout);
};

struct DFlashPrefillState {
    Tensor produced_count;

    DFlashPrefillState() = default;
    DFlashPrefillState(DeviceSpan backing, const DFlashPrefillStateLayout& layout);
};

struct MtpDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor anchors;
    Tensor base_frontiers;
    Tensor remaining_budgets;
    Tensor current_extents;
    Tensor target_valid_columns;
    Tensor current_drafts;
    Tensor target_rope_positions;
    Tensor text_kv_table_rows;
    Tensor mtp_kv_table_rows;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    Tensor rope_deltas;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor next_drafts;
    Tensor next_extents;
    // One report per verify column; column w of lane b is b * width + w.
    Tensor token_logprobs;
    Tensor top_ids;
    Tensor top_logprobs;
    Tensor verify_ids;
    Tensor target_positions;
    Tensor target_argmax;
    Tensor target_logits;
    Tensor target_hidden;
    Tensor target_continuation_hidden;
    std::optional<Tensor> target_mtp_hidden;
    std::optional<Tensor> target_continuation_mtp_hidden;
    Tensor proposal_logits;
    Tensor alignment_ids;
    Tensor alignment_hidden;
    std::optional<Tensor> alignment_mtp_hidden;
    std::optional<Tensor> alignment_qsa_indices;
    Tensor ar_hidden;
    std::optional<Tensor> ar_mtp_hidden;
    std::optional<Tensor> ar_qsa_indices;
    Tensor next_hidden;
    std::optional<Tensor> next_mtp_hidden;
    Tensor ar_positions;
    Tensor ar_rope_positions;
    Tensor ar_valid_columns;

    MtpDecodeState() = default;
    MtpDecodeState(DeviceSpan backing, const MtpDecodeStateLayout& layout,
                   std::uint32_t batch_capacity, std::uint32_t draft_window);
};

struct DFlashDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor anchors;
    Tensor execution_frontiers;
    Tensor context_frontiers;
    Tensor proposal_extents;
    Tensor target_valid_columns;
    Tensor text_kv_table_rows;
    Tensor dflash_kv_table_rows;
    Tensor active_lanes;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor proposal_ids;
    Tensor proposal_positions;
    Tensor append_positions;
    Tensor append_counts;
    Tensor draft_tokens;
    Tensor verify_ids;
    Tensor target_argmax;
    Tensor target_logits;
    Tensor target_hidden;
    Tensor target_continuation_hidden;

    DFlashDecodeState() = default;
    DFlashDecodeState(DeviceSpan backing, const DFlashDecodeStateLayout& layout,
                      std::uint32_t batch_capacity, std::uint32_t draft_window);
};

struct RoundState {
    std::optional<OrdinaryDecodeState> ordinary;
    Tensor token;
    Tensor pos;
    Tensor rope_pos;
    Tensor rope_delta;
    Tensor logits;
    Tensor text_kv_table_row;
    Tensor backend_kv_table_row;
    DeviceSpan report;
    Tensor report_logprob;
    Tensor report_top_ids;
    Tensor report_top_logprobs;
    std::optional<MtpPrefillState> mtp;
    std::optional<DFlashPrefillState> dflash_prefill;
    std::optional<MtpDecodeState> mtp_decode;
    std::optional<DFlashDecodeState> dflash_decode;
    // Selected once at load from EngineOptions::token_logprobs. It is read while a decode graph is
    // captured, so the probability report is either present in a captured round or absent; a round
    // never branches on it.
    bool report_token_logprobs = false;

    RoundState() = default;
    RoundState(DeviceSpan backing, const RoundStateLayout& layout);
};

} // namespace ninfer::models::qwen3_8_flash_next
