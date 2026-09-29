#include "models/qwen3_8_flash_next/impl/runtime/instance.h"
#include "models/qwen3_8_flash_next/impl/runtime/schedule.h"

#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scatter.h"

#include <stdexcept>

namespace ninfer::models::qwen3_8_flash_next::detail::NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS::
    schedule {
namespace {

auto ordinary_batch_body(OrdinaryBatchContext& state, std::int32_t batch_size,
                         ops::CausalAttentionExecutionEnvelope envelope) {
    return [&state, batch_size, envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency)) {
            throw std::logic_error("ordinary decode batch state is incomplete");
        }

        qwen3_8_flash_next::OrdinaryDecodeState& ordinary = state.frame;
        CUDA_CHECK(cudaMemcpyAsync(ordinary.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_8_flash_next::OrdinaryDecodeIngress),
                                   cudaMemcpyHostToDevice, state.execution.device.stream));

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, nullptr, state.execution.ple_state);

        Tensor tokens             = ordinary.tokens.slice(0, 0, batch_size);
        Tensor cache_positions    = ordinary.cache_positions.slice(0, 0, batch_size);
        Tensor rope_positions     = ordinary.rope_positions.slice(0, 0, batch_size);
        Tensor kv_rows            = ordinary.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor state_sources      = ordinary.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = ordinary.state_destination_slots.slice(0, 0, batch_size);
        Tensor hidden             = ordinary.hidden.slice(1, 0, batch_size);
        Tensor logits             = ordinary.logits.slice(1, 0, batch_size);
        Tensor sampled            = ordinary.sampled_tokens.slice(0, 0, batch_size);

        Tensor predictor_hidden;
        if constexpr (Variant::flash_next) {
            if (state.execution.mtp_prefill_hidden != nullptr) {
                if (state.execution.mtp_continuation_hidden_store == nullptr) {
                    throw std::logic_error(
                        "Flash-Next ordinary decode has no predictor continuation store");
                }
                predictor_hidden = state.execution.mtp_prefill_hidden->slice(1, 0, batch_size);
                card.set_flash_predictor_hidden_output(&predictor_hidden);
            }
        }

        Tensor ple_embeddings;
        const Tensor* active_ple_embeddings = nullptr;
        if (state.ple_embeddings != nullptr) {
            ple_embeddings = state.ple_embeddings->slice(2, 0, batch_size);
            active_ple_embeddings = &ple_embeddings;
        }
        card.ordinary_decode_batch(tokens, cache_positions, rope_positions, kv_rows, state_sources,
                                   state_destinations, envelope, hidden, logits,
                                   active_ple_embeddings);
        ops::scatter(hidden, state_destinations, state.continuation_hidden_store,
                     state.execution.device.stream);
        if (predictor_hidden.data != nullptr) {
            ops::scatter(predictor_hidden, state_destinations,
                         *state.execution.mtp_continuation_hidden_store,
                         state.execution.device.stream);
            card.set_flash_predictor_hidden_output(nullptr);
        }
        ops::sample(logits, sampled, TextConfig::token_domain, ordinary.sampling, cache_positions,
                    ops::kSamplePurposeDecode, state.execution.work, state.execution.device.stream);
        // The report is the target model's own distribution at this position: no temperature, no
        // penalties. It is therefore a checkpoint property rather than a sampling artefact, and it
        // reads no sampler state, no committed-token counts and no round-local token array - which is
        // also why an external scorer can agree with it.
        if (state.execution.io.report_token_logprobs) {
            Tensor chosen            = ordinary.token_logprobs.slice(0, 0, batch_size);
            Tensor reported_ids      = ordinary.top_ids.slice(1, 0, batch_size);
            Tensor reported_logprobs = ordinary.top_logprobs.slice(1, 0, batch_size);
            ops::target_logprobs(logits, sampled, TextConfig::token_domain, chosen, &reported_ids,
                                 &reported_logprobs, state.execution.device.stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, ordinary.egress.data,
                                   sizeof(qwen3_8_flash_next::OrdinaryDecodeEgress),
                                   cudaMemcpyDeviceToHost, state.execution.device.stream));
    };
}

} // namespace

void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   DecodeGraphDefinition& definition) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    capture_graph(state, definition, body);
}

void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           ops::CausalAttentionExecutionEnvelope envelope,
                           DecodeGraphExecutable* executable) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    run_prepared(state, executable, body);
}

} // namespace
  // ninfer::models::qwen3_8_flash_next::detail::NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS::schedule
