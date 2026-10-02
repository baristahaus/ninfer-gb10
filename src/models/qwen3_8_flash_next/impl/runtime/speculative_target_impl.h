#include "models/qwen3_8_flash_next/impl/runtime/instance.h"
#include "models/qwen3_8_flash_next/impl/runtime/schedule.h"

#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

namespace ninfer::models::qwen3_8_flash_next::detail::NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS::
    schedule {

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::CausalAttentionExecutionEnvelope envelope) {
    if (frame.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    card.set_gdn_pending_fold(frame.pending_records,
                              frame.pending_records != nullptr ? &frame.pending_folds : nullptr);
    card.set_ple_replay_records(execution.ple_records);
    card.set_flash_predictor_hidden_output(
        frame.target_mtp_hidden.data != nullptr ? &frame.target_mtp_hidden : nullptr);
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.state_source_slots,
                                 envelope, frame.target_hidden, frame.target_logits,
                                 frame.target_tokens, *frame.feature_sink);
    } else {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.state_source_slots,
                                 envelope, frame.target_hidden, frame.target_logits,
                                 frame.target_tokens);
    }
    ops::speculative_accept_greedy_drafts(frame.target_tokens, frame.target_logits, frame.drafts,
                                          frame.current_extents, frame.frontiers, frame.anchors,
                                          frame.licensed_tokens, frame.licensed_counts,
                                          frame.accepted_drafts, TextConfig::token_domain,
                                          frame.sampling, execution.work, execution.device.stream);
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.selected_hidden, execution.device.stream);
    ops::scatter(frame.selected_hidden, frame.state_destination_slots, continuation_hidden_store,
                 execution.device.stream);
    if (frame.target_mtp_hidden.data != nullptr) {
        if (execution.mtp_continuation_hidden_store == nullptr ||
            frame.selected_mtp_hidden.data == nullptr) {
            throw std::logic_error("speculative target MTP continuation storage is unavailable");
        }
        ops::speculative_select_accepted_hidden(frame.target_mtp_hidden, frame.accepted_drafts,
                                                frame.selected_mtp_hidden, execution.device.stream);
        ops::scatter(frame.selected_mtp_hidden, frame.state_destination_slots,
                     *execution.mtp_continuation_hidden_store, execution.device.stream);
    }
    card.set_flash_predictor_hidden_output(nullptr);
    card.set_gdn_pending_fold(nullptr, nullptr);
}

} // namespace
  // ninfer::models::qwen3_8_flash_next::detail::NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS::schedule
