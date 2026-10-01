#include "models/qwen3_8_flash_next/impl/runtime/instance.h"
#include "models/qwen3_8_flash_next/impl/runtime/schedule.h"

#include "core/nvtx.h"
#include "ninfer/ops/flash_next_ple.h"
#include "ninfer/ops/mtp_round.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ninfer::models::qwen3_8_flash_next::detail::NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS::
    schedule {
void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, std::int32_t position,
                            std::span<const std::int32_t> rope_position, bool build_proposal,
                            const Tensor* next_embedding) {
    if (!state.mtp_kv.valid() || !state.execution.io.mtp) {
        throw std::logic_error("MTP bridge requires MTP storage");
    }
    if (rope_position.size() != 3) {
        throw std::invalid_argument("MTP bridge requires one three-axis rope position");
    }
    state.execution.work.reset();
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache,
                     state.execution.ple_state);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);

    Tensor position_view = state.execution.io.mtp->target_positions.slice(0, 0, 1);
    ops::set_i32_scalar(position_view, position, state.execution.device.stream);
    Tensor mtp_hidden         = state.execution.io.mtp->ar_hidden;
    Tensor logits             = state.execution.io.logits.slice(1, 0, 1);
    Tensor draft0             = state.execution.io.mtp->draft_tokens.slice(0, 0, 1);
    Tensor rope_position_view = state.execution.work.alloc(DType::I32, {1, 3});
    CUDA_CHECK(cudaMemcpyAsync(rope_position_view.data, rope_position.data(),
                               rope_position.size_bytes(), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    const auto bridge_visible = static_cast<std::uint32_t>(position + 1);
    const ops::CausalAttentionExecutionEnvelope bridge_envelope{bridge_visible, bridge_visible};
    if constexpr (Variant::flash_next) {
        Tensor sample_hidden = state.execution.work.alloc(DType::BF16, {TextConfig::hidden, 1});
        card.mtp_forward_batch(
            next_token, previous_hidden, position_view, bridge_envelope, sample_hidden,
            build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
            build_proposal ? &draft0 : nullptr, &rope_position_view, next_embedding, &mtp_hidden);
    } else {
        card.mtp_forward_batch(
            next_token, previous_hidden, position_view, bridge_envelope, mtp_hidden,
            build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
            build_proposal ? &draft0 : nullptr, &rope_position_view, next_embedding);
    }
    if (!build_proposal) { return; }

    if (state.mtp_proposal_extent == 0 ||
        state.mtp_proposal_extent >
            static_cast<std::uint32_t>(state.execution.io.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP bridge proposal extent is outside the configured window");
    }

    Tensor ar_position = state.execution.io.mtp->position.slice(0, 0, 1);
    ops::set_i32_scalar(ar_position, position + 1, state.execution.device.stream);
    for (int i = 1; i < static_cast<int>(state.mtp_proposal_extent); ++i) {
        Tensor previous_token = state.execution.io.mtp->draft_tokens.slice(0, i - 1, 1);
        Tensor next_draft     = state.execution.io.mtp->draft_tokens.slice(0, i, 1);
        Tensor next_hidden    = state.execution.prefill_hidden.slice(1, i, 1);
        const auto visible    = static_cast<std::uint32_t>(position + i + 1);
        const ops::CausalAttentionExecutionEnvelope envelope{visible, visible};
        if constexpr (Variant::flash_next) {
            if (state.execution.mtp_prefill_hidden == nullptr) {
                throw std::logic_error("Flash-Next bridge predictor scratch is unavailable");
            }
            Tensor next_predictor = state.execution.mtp_prefill_hidden->slice(1, i, 1);
            card.mtp_forward_ar_step(previous_token, state.execution.io.mtp->ar_hidden, ar_position,
                                     envelope, next_hidden, logits, next_draft, &next_predictor);
            CUDA_CHECK(cudaMemcpyAsync(state.execution.io.mtp->ar_hidden.data, next_predictor.data,
                                       next_predictor.bytes(), cudaMemcpyDeviceToDevice,
                                       state.execution.device.stream));
        } else {
            card.mtp_forward_ar_step(previous_token, state.execution.io.mtp->ar_hidden, ar_position,
                                     envelope, next_hidden, logits, next_draft);
            CUDA_CHECK(cudaMemcpyAsync(state.execution.io.mtp->ar_hidden.data, next_hidden.data,
                                       state.execution.io.mtp->ar_hidden.bytes(),
                                       cudaMemcpyDeviceToDevice, state.execution.device.stream));
        }
        ops::increment_i32_scalar(ar_position, state.execution.device.stream);
    }
}

auto mtp_decode_batch_body(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                           MtpCausalAttentionEnvelopes envelopes) {
    return [&state, batch_size, k, envelopes] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kMtpDecodeMaximumDrafts) {
            throw std::logic_error("MTP decode batch state is incomplete");
        }

        qwen3_8_flash_next::MtpDecodeState& frame = state.frame;
        const std::int32_t width                  = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_8_flash_next::MtpDecodeIngress),
                                   cudaMemcpyHostToDevice, state.execution.device.stream));
        // The previous round's deferred commit: its accepted columns fold into the state slots
        // before this round's verify reads them and overwrites the records they come from.
        if (state.pending_fold != nullptr && state.pending_fold_rows > 0) {
            state.pending_fold->execute_device(frame.pending_folds, state.pending_fold_rows,
                                               state.execution.device.stream);
            if constexpr (Variant::flash_next) {
                if (state.execution.ple_records == nullptr ||
                    state.execution.ple_state == nullptr) {
                    throw std::logic_error("Flash-Next deferred PLE fold storage is unavailable");
                }
                ops::flash_next_ple_replay_fold_device(
                    *state.execution.ple_records, *state.execution.ple_state, frame.pending_folds,
                    state.pending_fold_rows, state.execution.device.stream);
            }
        }

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, &state.mtp_cache, state.execution.ple_state);
        Tensor anchors            = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers          = frame.base_frontiers.slice(0, 0, batch_size);
        Tensor budgets            = frame.remaining_budgets.slice(0, 0, batch_size);
        Tensor current_extents    = frame.current_extents.slice(0, 0, batch_size);
        Tensor target_valid       = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor current_drafts     = frame.current_drafts.slice(1, 0, batch_size);
        Tensor target_rope        = frame.target_rope_positions.slice(1, 0, batch_size);
        Tensor text_rows          = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor mtp_rows           = frame.mtp_kv_table_rows.slice(0, 0, batch_size);
        Tensor state_sources      = frame.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor rope_deltas        = frame.rope_deltas.slice(0, 0, batch_size);
        Tensor verify_ids         = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions   = frame.target_positions.slice(1, 0, batch_size);
        Tensor target_tokens      = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits      = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden      = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden    = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor target_mtp_hidden;
        Tensor selected_mtp_hidden;
        Tensor alignment_mtp_hidden;
        Tensor ar_mtp_hidden;
        Tensor next_mtp_hidden;
        Tensor alignment_qsa_indices;
        Tensor ar_qsa_indices;
        if constexpr (Variant::flash_next) {
            if (!frame.target_mtp_hidden || !frame.target_continuation_mtp_hidden ||
                !frame.alignment_mtp_hidden || !frame.ar_mtp_hidden || !frame.next_mtp_hidden ||
                !frame.alignment_qsa_indices || !frame.ar_qsa_indices) {
                throw std::logic_error("Flash-Next MTP frame has no predictor hidden storage");
            }
            target_mtp_hidden     = frame.target_mtp_hidden->slice(2, 0, batch_size);
            selected_mtp_hidden   = frame.target_continuation_mtp_hidden->slice(1, 0, batch_size);
            alignment_mtp_hidden  = frame.alignment_mtp_hidden->slice(2, 0, batch_size);
            ar_mtp_hidden         = frame.ar_mtp_hidden->slice(1, 0, batch_size);
            next_mtp_hidden       = frame.next_mtp_hidden->slice(1, 0, batch_size);
            alignment_qsa_indices = frame.alignment_qsa_indices->slice(2, 0, batch_size);
            ar_qsa_indices        = frame.ar_qsa_indices->slice(1, 0, batch_size);
        }
        Tensor licensed_tokens   = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts   = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted          = frame.accepted_drafts.slice(0, 0, batch_size);
        Tensor next_extents      = frame.next_extents.slice(0, 0, batch_size);
        Tensor alignment_ids     = frame.alignment_ids.slice(1, 0, batch_size);
        Tensor alignment_hidden  = frame.alignment_hidden.slice(2, 0, batch_size);
        Tensor ar_hidden         = frame.ar_hidden.slice(1, 0, batch_size);
        Tensor next_hidden       = frame.next_hidden.slice(1, 0, batch_size);
        Tensor ar_positions      = frame.ar_positions.slice(0, 0, batch_size);
        Tensor ar_rope_positions = frame.ar_rope_positions.slice(0, 0, batch_size);
        Tensor ar_valid_columns  = frame.ar_valid_columns.slice(0, 0, batch_size);
        Tensor next_drafts       = frame.next_drafts.slice(0, 0, batch_size);

        Tensor ple_embeddings;
        if (state.ple_embeddings != nullptr) {
            ple_embeddings = state.ple_embeddings->slice(2, 0, batch_size);
            card.set_ple_embeddings(&ple_embeddings);
        }

        ops::speculative_prepare_verify_inputs(anchors, current_drafts, frontiers, current_extents,
                                               verify_ids, target_positions,
                                               state.execution.device.stream);
        {
            nvtx::ScopedRange target_range(nvtx::Name::DecodeMtpTarget, nvtx::Category::Mtp,
                                           static_cast<std::uint64_t>(width) * batch_size);
            target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                 TargetVerifyFrameView{
                                     .ids                     = verify_ids,
                                     .cache_positions         = target_positions,
                                     .rope_positions          = target_rope,
                                     .valid_columns           = target_valid,
                                     .kv_table_rows           = text_rows,
                                     .state_source_slots      = state_sources,
                                     .state_destination_slots = state_destinations,
                                     .target_hidden           = target_hidden,
                                     .target_mtp_hidden       = target_mtp_hidden,
                                     .target_logits           = target_logits,
                                     .target_tokens           = target_tokens,
                                     .drafts                  = current_drafts,
                                     .current_extents         = current_extents,
                                     .frontiers               = frontiers,
                                     .anchors                 = anchors,
                                     .licensed_tokens         = licensed_tokens,
                                     .licensed_counts         = licensed_counts,
                                     .accepted_drafts         = accepted,
                                     .selected_hidden         = selected_hidden,
                                     .selected_mtp_hidden     = selected_mtp_hidden,
                                     .replay_records          = state.execution.replay_records,
                                     .sampling                = frame.sampling,
                                 },
                                 envelopes.target_verify);
        }

        if (state.execution.io.report_token_logprobs) {
            // target_logits is [rows, width, batch]; column-major storage makes the same memory a
            // [rows, width*batch] matrix whose column g is lane b's verify column w under
            // g = w + b*width. licensed_tokens is bound as [width, batch] and read by the host as
            // [row * width + column], so the two already share one column order.
            const std::int32_t report_columns = width * batch_size;
            Tensor verify_logits(target_logits.data, DType::BF16,
                                 {target_logits.ne[0], report_columns});
            Tensor published         = licensed_tokens.view({report_columns});
            Tensor chosen            = frame.token_logprobs.slice(0, 0, report_columns);
            Tensor reported_ids      = frame.top_ids.slice(1, 0, report_columns);
            Tensor reported_logprobs = frame.top_logprobs.slice(1, 0, report_columns);

            ops::target_logprobs(verify_logits, published, TextConfig::token_domain, chosen,
                                 &reported_ids, &reported_logprobs,
                                 state.execution.device.stream);
        }

        {
            nvtx::ScopedRange draft_range(nvtx::Name::DecodeMtpDraft, nvtx::Category::Mtp,
                                          static_cast<std::uint64_t>(k) * batch_size);
            ops::mtp_prepare_next_round(verify_ids, anchors, accepted, frontiers, budgets,
                                        licensed_counts, rope_deltas, alignment_ids, next_extents,
                                        ar_positions, ar_rope_positions, ar_valid_columns,
                                        static_cast<std::int32_t>(state.text_cache.max_context()),
                                        state.execution.device.stream);
            if constexpr (Variant::flash_next) {
                card.mtp_forward_decode_batch(
                    alignment_ids, target_mtp_hidden, target_positions, target_rope,
                    licensed_counts, mtp_rows, envelopes.batch, alignment_hidden,
                    &alignment_mtp_hidden, &alignment_qsa_indices, nullptr);
            } else {
                card.mtp_forward_decode_batch(alignment_ids, target_hidden, target_positions,
                                              target_rope, licensed_counts, mtp_rows,
                                              envelopes.batch, alignment_hidden);
            }
            ops::speculative_select_accepted_hidden(alignment_hidden, accepted, ar_hidden,
                                                    state.execution.device.stream);
            if constexpr (Variant::flash_next) {
                ops::speculative_select_accepted_hidden(
                    alignment_mtp_hidden, accepted, ar_mtp_hidden, state.execution.device.stream);
                ops::flash_next_qsa_select_indices(alignment_qsa_indices, accepted, ar_qsa_indices,
                                                   state.execution.device.stream);
            }

            Tensor proposal_logits = frame.proposal_logits.slice(1, 0, batch_size);
            Tensor draft0          = next_drafts.slice(1, 0, 1).view({batch_size});
            card.mtp_propose_batch(ar_hidden, proposal_logits, draft0);
            for (std::uint32_t step = 0; step + 1 < k; ++step) {
                Tensor previous =
                    next_drafts.slice(1, static_cast<std::int32_t>(step), 1).view({batch_size});
                Tensor next =
                    next_drafts.slice(1, static_cast<std::int32_t>(step + 1), 1).view({batch_size});
                Tensor position =
                    ar_positions.slice(1, static_cast<std::int32_t>(step), 1).view({1, batch_size});
                Tensor rope  = ar_rope_positions.slice(1, static_cast<std::int32_t>(step), 1)
                                   .view({1, batch_size});
                Tensor valid = ar_valid_columns.slice(1, static_cast<std::int32_t>(step), 1)
                                   .view({batch_size});
                Tensor previous_batch    = previous.view({1, batch_size});
                Tensor hidden_batch      = ar_hidden.view({TextConfig::hidden, 1, batch_size});
                Tensor next_hidden_batch = next_hidden.view({TextConfig::hidden, 1, batch_size});
                if constexpr (Variant::flash_next) {
                    Tensor predictor_batch =
                        ar_mtp_hidden.view({4 * TextConfig::hidden, 1, batch_size});
                    Tensor next_predictor_batch =
                        next_mtp_hidden.view({4 * TextConfig::hidden, 1, batch_size});
                    card.mtp_forward_decode_batch(previous_batch, predictor_batch, position, rope,
                                                  valid, mtp_rows, envelopes.ar[step],
                                                  next_hidden_batch, &next_predictor_batch, nullptr,
                                                  &ar_qsa_indices);
                } else {
                    card.mtp_forward_decode_batch(previous_batch, hidden_batch, position, rope,
                                                  valid, mtp_rows, envelopes.ar[step],
                                                  next_hidden_batch);
                }
                card.mtp_propose_batch(next_hidden, proposal_logits, next);
                CUDA_CHECK(cudaMemcpyAsync(ar_hidden.data, next_hidden.data, ar_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice,
                                           state.execution.device.stream));
                if constexpr (Variant::flash_next) {
                    CUDA_CHECK(cudaMemcpyAsync(ar_mtp_hidden.data, next_mtp_hidden.data,
                                               ar_mtp_hidden.bytes(), cudaMemcpyDeviceToDevice,
                                               state.execution.device.stream));
                }
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_8_flash_next::MtpDecodeEgress),
                                   cudaMemcpyDeviceToHost, state.execution.device.stream));
    };
}

void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              MtpCausalAttentionEnvelopes envelopes,
                              DecodeGraphDefinition& definition) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    capture_graph(state, definition, body);
}

void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      MtpCausalAttentionEnvelopes envelopes, DecodeGraphExecutable* executable) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    run_prepared(state, executable, body);
}

} // namespace
  // ninfer::models::qwen3_8_flash_next::detail::NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS::schedule
