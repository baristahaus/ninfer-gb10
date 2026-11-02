#include "models/qwen3_8_flash_next/impl/runtime/instance.h"
#include "models/qwen3_8_flash_next/impl/runtime/schedule.h"
#include "models/qwen3_8_flash_next/impl/runtime/workspace_recipe.h"

#include "core/nvtx.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/speculative_round.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_8_flash_next::detail::NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS::
    schedule {
namespace {

void require_dflash_state(const PrefillContext& state) {
    if (state.dflash == nullptr ||
        (!state.execution.model.dflash.has_value() && !state.execution.model.dflash2.has_value())) {
        throw std::logic_error("DFlash schedule requires DFlash weights and state");
    }
}

DFlashPersistentState& dflash_state(PrefillContext& state) {
    require_dflash_state(state);
    return *state.dflash;
}

DFlashPersistentState& dflash_state(DFlashBatchContext& state) { return state.dflash; }

DFlashPersistentState& dflash_state(DFlashAppendContext& state) { return state.dflash; }

template <class V>
DFlashFeatureSink prefill_feature_sink_impl(PrefillContext& state,
                                            DFlashFeatureSink::PrefillConsumer consume_prefill) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        require_dflash_state(state);
        using Config = typename V::DFlashConfig;
        return DFlashFeatureSink{
            .features        = &dflash_state(state).prefill_features,
            .positions       = &dflash_state(state).prefill_positions,
            .layers          = std::span<const int>(Config::target_feature_layers),
            .consume_prefill = std::move(consume_prefill),
        };
    }
}

template <class V>
DFlashFeatureSink batch_feature_sink_impl(DFlashBatchContext& state, const Tensor& lanes,
                                          const Tensor& valid_columns, std::int32_t width,
                                          std::int32_t batch_size) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        using Config = typename V::DFlashConfig;
        return DFlashFeatureSink{
            .batch_features      = &dflash_state(state).pending_features,
            .batch_lanes         = &lanes,
            .batch_valid_columns = &valid_columns,
            .batch_width         = width,
            .batch_size          = batch_size,
            .layers              = std::span<const int>(Config::target_feature_layers),
        };
    }
}

template <class V, class Context>
void append_context_impl(Context& state, const Tensor& features, const Tensor& positions,
                         const Tensor& commit_counts, const Tensor& lanes, const Tensor& table_rows,
                         ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash context append is unavailable for this target");
    } else {
        using Config               = typename V::DFlashConfig;
        const std::int32_t width   = features.ne[1];
        const std::int32_t batch   = features.ne[2];
        const std::int32_t columns = width * batch;
        nvtx::ScopedRange append_range(nvtx::Name::DFlashContextAppend, nvtx::Category::DFlash,
                                       static_cast<std::uint64_t>(columns));
        if (width <= 0 || batch <= 0 || features.dtype != DType::BF16 ||
            features.ne[0] != Config::feature_rows || features.ne[3] != 1 ||
            positions.dtype != DType::I32 || positions.ne[0] != width || positions.ne[1] != batch ||
            commit_counts.dtype != DType::I32 || commit_counts.ne[0] != batch ||
            lanes.dtype != DType::I32 || lanes.ne[0] != batch || table_rows.dtype != DType::I32 ||
            table_rows.ne[0] != batch) {
            throw std::invalid_argument("DFlash context append inputs are invalid");
        }
        const bool replace_local_window = batch == 1 && width > Config::local_capacity;
        if (replace_local_window && (envelope.min_count != static_cast<std::uint32_t>(width) ||
                                     envelope.max_count != static_cast<std::uint32_t>(width))) {
            throw std::invalid_argument(
                "DFlash oversized local append requires an exact full-prefix commit");
        }
        const int local_offset = replace_local_window ? width - Config::local_capacity : 0;
        const int local_width  = replace_local_window ? Config::local_capacity : width;
        const ops::KVCacheAppendPrefixExecutionEnvelope local_envelope{
            replace_local_window ? static_cast<std::uint32_t>(Config::local_capacity)
                                 : envelope.min_count,
            replace_local_window ? static_cast<std::uint32_t>(Config::local_capacity)
                                 : envelope.max_count,
        };
        Tensor local_counts = commit_counts;
        if (replace_local_window) {
            if (!state.execution.io.dflash_prefill) {
                throw std::logic_error("DFlash prefill count storage is unavailable");
            }
            local_counts = state.execution.io.dflash_prefill->produced_count;
            ops::set_i32_scalar(local_counts, Config::local_capacity,
                                state.execution.device.execution_view());
        }

        const auto context_roots =
            workspace_recipe::dflash_context<Config>(state.execution.work, columns);
        Tensor projected = context_roots.projected;
        // The v1 DFlash draft and the DFlash2 draft share the context projection, the layer
        // context projections, and the norm names; only the model-view source differs.
        const auto project_and_norm = [&](const auto& draft) {
            ops::linear(features.view({Config::feature_rows, columns}), draft.feature_projection,
                        projected, state.execution.device.execution_view());
            Tensor normalized = context_roots.normalized;
            ops::rmsnorm(projected, draft.context_norm, Config::rms_epsilon, false, normalized,
                         state.execution.device.execution_view());
            return normalized;
        };
        const auto dump_append = std::getenv("NINFER_DFLASH2_DUMP") != nullptr;
        const auto append_layers = [&](const auto& draft) {
        Tensor context = project_and_norm(draft);
        if (dump_append) {
            CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream()));
            std::vector<std::uint16_t> all(static_cast<std::size_t>(context.bytes()) / 2);
            CUDA_CHECK(cudaMemcpy(all.data(), context.data, context.bytes(),
                                  cudaMemcpyDeviceToHost));
            std::size_t nan = 0, zero = 0;
            double magnitude = 0.0;
            for (auto b16 : all) {
                if ((b16 & 0x7f80u) == 0x7f80u && (b16 & 0x007fu) != 0) { ++nan; }
                else if (b16 == 0) { ++zero; }
                magnitude += std::fabs(static_cast<float>(static_cast<std::int16_t>(b16)));
            }
            std::fprintf(stderr, "[dflash2] context scan: n=%zu nan=%zu zero=%zu mean|x|=%.5f first:",
                         all.size(), nan, zero, magnitude / static_cast<double>(all.size()));
            for (std::size_t i = 0; i < 6 && i < all.size(); ++i) {
                std::fprintf(stderr, " %04x", all[i]);
            }
            std::fprintf(stderr, "\n");
        }
        for (int layer = 0; layer < Config::layers; ++layer) {
            auto layer_scope = state.execution.work.scope();
            const auto& weight = draft.layers.at(static_cast<std::size_t>(layer));
            const bool local_layer  = layer < Config::local_layers;
            const int layer_width   = local_layer ? local_width : width;
            const int layer_columns = layer_width * batch;
            Tensor layer_context    = local_layer && replace_local_window
                                          ? context.slice(1, local_offset, local_width)
                                          : context;
            Tensor layer_positions  = local_layer && replace_local_window
                                          ? positions.slice(0, local_offset, local_width)
                                          : positions;
            auto layer_roots =
                workspace_recipe::dflash_context_layer<Config>(state.execution.work, layer_columns);
            Tensor key_raw =
                layer_roots.key_raw.view({Config::head_dim, Config::kv_heads, layer_columns});
            Tensor value =
                layer_roots.value.view({Config::head_dim, Config::kv_heads, layer_columns});
            Tensor key_flat   = key_raw.view({Config::kv_size, layer_columns});
            Tensor value_flat = value.view({Config::kv_size, layer_columns});
            // The v1 draft's context projections are quantized and run as one pair; the
            // DFlash2 draft's are BF16 and run as two registered dense projections.
            if (weight.context_key.qtype == QType::BF16) {
                ops::linear(layer_context, weight.context_key, key_flat,
                            state.execution.device.execution_view());
                ops::linear(layer_context, weight.context_value, value_flat,
                            state.execution.device.execution_view());
            } else {
                ops::linear_pair(layer_context, weight.context_key, weight.context_value,
                                 key_flat, value_flat, state.execution.device.execution_view());
            }
            Tensor key = layer_roots.key.view({Config::head_dim, Config::kv_heads, layer_columns});
            ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                         state.execution.device.execution_view());
            ops::rope(layer_positions.view({layer_columns}), Config::head_dim, Config::rope_theta,
                      key, state.execution.device.execution_view());
            Tensor key_batch = key.view({Config::head_dim, Config::kv_heads, layer_width, batch});
            Tensor value_batch =
                value.view({Config::head_dim, Config::kv_heads, layer_width, batch});
            Tensor position_batch = layer_positions.view({layer_width, batch});
            if (local_layer) {
                ops::kv_cache_append_prefix(
                    key_batch, value_batch, position_batch, local_counts, lanes, local_envelope,
                    dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                    state.execution.device.execution_view());
            } else {
                ops::kv_cache_append_prefix(
                    key_batch, value_batch, position_batch, commit_counts, table_rows, envelope,
                    dflash_state(state).full_batch_layer(0), state.execution.device.execution_view());
            }
        }
        };
        if (state.execution.model.dflash2.has_value()) {
            append_layers(*state.execution.model.dflash2);
        } else {
            append_layers(*state.execution.model.dflash);
        }
        if (std::getenv("NINFER_DFLASH2_DUMP") != nullptr) {
            // The draft's context must land in the lane-local cyclic caches; scan the first
            // layer's K plane for a live window.
            CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream()));
            const auto cache = dflash_state(state).local_layer(0);
            const std::size_t count = std::min<std::size_t>(
                static_cast<std::size_t>(cache.k.bytes()) / 2, 2560u * 4u);
            std::vector<std::uint16_t> all(count);
            CUDA_CHECK(cudaMemcpy(all.data(), cache.k.data, count * 2, cudaMemcpyDeviceToHost));
            std::size_t nan = 0, zero = 0;
            for (auto b16 : all) {
                if ((b16 & 0x7f80u) == 0x7f80u && (b16 & 0x007fu) != 0) { ++nan; }
                else if (b16 == 0) { ++zero; }
            }
            std::fprintf(stderr, "[dflash2] context cache K scan: n=%zu nan=%zu zero=%zu first:",
                         count, nan, zero);
            for (std::size_t i = 0; i < 8 && i < count; ++i) {
                std::fprintf(stderr, " %04x", all[i]);
            }
            std::fprintf(stderr, "\n");
        }
    }
}

template <class V>
void propose_batch_impl(DFlashBatchContext& state, qwen3_8_flash_next::DFlashDecodeState& frame,
                        std::int32_t batch_size, std::uint32_t k, DFlashEnvelopes envelopes) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash proposal is unavailable for this target");
    } else {
        using Config               = typename V::DFlashConfig;
        const std::int32_t width   = static_cast<std::int32_t>(k) + 1;
        const std::int32_t columns = width * batch_size;
        nvtx::ScopedRange proposal_range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash,
                                         static_cast<std::uint64_t>(columns));
        Tensor anchors            = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers          = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor valid_columns      = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor full_rows          = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor ids                = frame.proposal_ids.slice(1, 0, batch_size);
        Tensor positions          = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor drafts             = frame.draft_tokens.slice(1, 0, batch_size);

        state.execution.work.reset();
        ops::prepare_masked_block(anchors, frontiers, valid_columns, Config::mask_token, ids,
                                  positions, state.execution.device.stream());
        Tensor residual = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
        ops::embedding(ids.view({columns}), state.execution.model.token_embedding, residual,
                       state.execution.device.execution_view());

        for (int layer = 0; layer < Config::layers; ++layer) {
            nvtx::ScopedRange layer_range(nvtx::Name::DFlashLayer, nvtx::Category::DFlash,
                                          static_cast<std::uint64_t>(layer));
            const auto& weight =
                state.execution.model.dflash->layers.at(static_cast<std::size_t>(layer));
            {
                nvtx::ScopedRange attention_range(nvtx::Name::DFlashAttention,
                                                  nvtx::Category::Attention,
                                                  static_cast<std::uint64_t>(layer));
                auto attention_scope = state.execution.work.scope();
                auto roots =
                    workspace_recipe::dflash_attention<Config>(state.execution.work, columns);
                ops::rmsnorm(residual, weight.input_norm, Config::rms_epsilon, false, roots.hidden,
                             state.execution.device.execution_view());
                Tensor query_raw =
                    roots.query_raw.view({Config::head_dim, Config::query_heads, columns});
                Tensor key_raw = roots.key_raw.view({Config::head_dim, Config::kv_heads, columns});
                Tensor value   = roots.value.view({Config::head_dim, Config::kv_heads, columns});
                Tensor query_flat = query_raw.view({Config::query_size, columns});
                Tensor key_flat   = key_raw.view({Config::kv_size, columns});
                Tensor value_flat = value.view({Config::kv_size, columns});
                ops::attn_input_proj(roots.hidden, weight.query_key_value, query_flat, key_flat,
                                     value_flat, state.execution.device.stream());
                Tensor query = roots.query.view({Config::head_dim, Config::query_heads, columns});
                Tensor key   = roots.key.view({Config::head_dim, Config::kv_heads, columns});
                ops::rmsnorm(query_raw, weight.query_norm, Config::rms_epsilon, false, query,
                             state.execution.device.execution_view());
                ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                             state.execution.device.execution_view());
                ops::rope(positions.view({columns}), Config::head_dim, Config::rope_theta, query,
                          key, state.execution.device.execution_view());
                Tensor query_batch =
                    query.view({Config::head_dim, Config::query_heads, width, batch_size});
                Tensor key_batch =
                    key.view({Config::head_dim, Config::kv_heads, width, batch_size});
                Tensor value_batch =
                    value.view({Config::head_dim, Config::kv_heads, width, batch_size});
                Tensor attention_batch = roots.attention.view(
                    {Config::head_dim, Config::query_heads, width, batch_size});
                if (layer < Config::local_layers) {
                    ops::sliding_window_attention(
                        query_batch, key_batch, value_batch, positions, valid_columns,
                        state_destinations,
                        {Config::head_dim, Config::query_heads, Config::kv_heads},
                        Config::local_capacity, Config::attention_scale,
                        dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                        envelopes.local, state.execution.work, attention_batch,
                        state.execution.device.execution_view());
                } else {
                    ops::context_softmax_attention(
                        query_batch, key_batch, value_batch, frontiers, valid_columns, full_rows,
                        {Config::head_dim, Config::query_heads, Config::kv_heads},
                        Config::attention_scale, dflash_state(state).full_batch_layer(0),
                        envelopes.full, state.execution.work, attention_batch,
                        state.execution.device.execution_view());
                }
                ops::linear_add(roots.attention.view({Config::query_size, columns}),
                                weight.attention_output, residual, state.execution.work,
                                state.execution.device.execution_view());
            }
            {
                nvtx::ScopedRange mlp_range(nvtx::Name::DFlashMlp, nvtx::Category::PostMixer,
                                            static_cast<std::uint64_t>(layer));
                auto mlp_scope = state.execution.work.scope();
                auto roots = workspace_recipe::dflash_mlp<Config>(state.execution.work, columns);
                ops::rmsnorm(residual, weight.post_attention_norm, Config::rms_epsilon, false,
                             roots.hidden, state.execution.device.execution_view());
                ops::linear_swiglu(roots.hidden, weight.gate_up, roots.intermediate,
                                   state.execution.work, state.execution.device.execution_view());
                ops::linear_add(roots.intermediate, weight.down, residual, state.execution.work,
                                state.execution.device.execution_view());
            }
        }

        Tensor packed = state.execution.work.alloc(
            DType::BF16, {Config::hidden, static_cast<std::int32_t>(k) * batch_size});
        const std::size_t element_bytes = dtype_size(DType::BF16);
        const std::size_t row_bytes =
            static_cast<std::size_t>(Config::hidden) * static_cast<std::size_t>(k) * element_bytes;
        const std::size_t source_pitch =
            static_cast<std::size_t>(Config::hidden) * width * element_bytes;
        const auto* source = static_cast<const std::byte*>(residual.data) +
                             static_cast<std::size_t>(Config::hidden) * element_bytes;
        CUDA_CHECK(cudaMemcpy2DAsync(packed.data, row_bytes, source, source_pitch, row_bytes,
                                     static_cast<std::size_t>(batch_size), cudaMemcpyDeviceToDevice,
                                     state.execution.device.stream()));
        Tensor proposal_hidden = state.execution.work.alloc(
            DType::BF16, {Config::hidden, static_cast<std::int32_t>(k) * batch_size});
        ops::rmsnorm(packed, state.execution.model.dflash->final_norm, Config::rms_epsilon, false,
                     proposal_hidden, state.execution.device.execution_view());
        Tensor flat_drafts = drafts.view({static_cast<std::int32_t>(k) * batch_size});
        if (state.execution.proposal_head == ProposalHead::Full) {
            Tensor logits = state.execution.work.alloc(
                DType::BF16, {TextConfig::output_rows, static_cast<std::int32_t>(k) * batch_size});
            ops::linear(proposal_hidden, state.execution.model.output_head, logits,
                        state.execution.device.execution_view());
            ops::argmax(logits, flat_drafts, TextConfig::token_domain,
                        state.execution.device.execution_view());
        } else {
            if (!state.execution.model.optimized_proposal.has_value()) {
                throw std::logic_error("optimized DFlash proposal head is unavailable");
            }
            const auto& proposal = *state.execution.model.optimized_proposal;
            Tensor logits        = state.execution.work.alloc(
                DType::BF16, {V::draft_head_rows, static_cast<std::int32_t>(k) * batch_size});
            ops::linear(proposal_hidden, proposal.head, logits, state.execution.device.execution_view());
            ops::argmax(logits, flat_drafts, V::draft_head_rows, state.execution.device.execution_view());
            ops::proposal_remap_token_ids(flat_drafts,
                                          static_cast<const std::int32_t*>(proposal.token_ids.data),
                                          V::draft_head_rows, state.execution.device.stream());
        }
        state.execution.work.reset();
    }
}

// Defined beside the v1 proposal in dflash2_impl.h; declared here because the decode body
// dispatches on the variant's speculative backend and this header is included first.
template <class V>
void propose_dflash2_batch_impl(DFlashBatchContext& state,
                                qwen3_8_flash_next::DFlashDecodeState& frame,
                                std::int32_t batch_size, std::uint32_t k,
                                DFlashEnvelopes envelopes);

auto dflash_decode_batch_body(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              DFlashEnvelopes envelopes,
                              ops::CausalAttentionExecutionEnvelope target_envelope) {
    return [&state, batch_size, k, envelopes, target_envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kDFlashDecodeMaximumDrafts) {
            throw std::logic_error("DFlash decode batch state is incomplete");
        }
        qwen3_8_flash_next::DFlashDecodeState& frame = state.frame;
        const std::int32_t width                     = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_8_flash_next::DFlashDecodeIngress),
                                   cudaMemcpyHostToDevice, state.execution.device.stream()));

        Tensor anchors            = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers          = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor context_starts     = frame.context_frontiers.slice(0, 0, batch_size);
        Tensor extents            = frame.proposal_extents.slice(0, 0, batch_size);
        Tensor valid_columns      = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor text_rows          = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor dflash_rows        = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor active_lanes       = frame.active_lanes.slice(0, 0, batch_size);
        Tensor state_sources      = frame.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor append_positions   = frame.append_positions.slice(1, 0, batch_size);
        Tensor append_counts      = frame.append_counts.slice(0, 0, batch_size);
        Tensor drafts             = frame.draft_tokens.slice(1, 0, batch_size);
        Tensor verify_ids         = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions   = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor target_tokens      = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits      = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden      = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden    = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens    = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts    = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted           = frame.accepted_drafts.slice(0, 0, batch_size);

        state.execution.work.reset();
        Tensor compact_features = state.execution.work.alloc(
            DType::BF16, {Variant::DFlashConfig::feature_rows, width, batch_size});
        ops::prepare_ragged_prefix(dflash_state(state).pending_features, active_lanes,
                                   context_starts, frontiers, compact_features, append_positions,
                                   append_counts, state.execution.device.stream());
        append_context_impl<Variant>(state, compact_features, append_positions, append_counts,
                                     state_destinations, dflash_rows, envelopes.append);
        if (std::getenv("NINFER_DFLASH2_DUMP") != nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream()));
            std::vector<std::int32_t> append_pos(static_cast<std::size_t>(width) * batch_size);
            CUDA_CHECK(cudaMemcpy(append_pos.data(), append_positions.data,
                                  append_positions.bytes(), cudaMemcpyDeviceToHost));
            std::vector<std::int32_t> append_count(static_cast<std::size_t>(batch_size));
            CUDA_CHECK(cudaMemcpy(append_count.data(), append_counts.data, append_counts.bytes(),
                                  cudaMemcpyDeviceToHost));
            std::fprintf(stderr,
                         "[dflash2] round ingress: anchor=%d E=%d CF=%d extent=%d lane=%d dest=%d | append pos:",
                         state.host_ingress.anchors[0],
                         state.host_ingress.execution_frontiers[0],
                         state.host_ingress.context_frontiers[0],
                         state.host_ingress.proposal_extents[0],
                         state.host_ingress.active_lanes[0],
                         state.host_ingress.state_destination_slots[0]);
            for (int i = 0; i < width * batch_size; ++i) {
                std::fprintf(stderr, " %d", append_pos[static_cast<std::size_t>(i)]);
            }
            std::fprintf(stderr, " count:");
            for (int i = 0; i < batch_size; ++i) {
                std::fprintf(stderr, " %d", append_count[static_cast<std::size_t>(i)]);
            }
            {
                std::vector<std::uint16_t> feats(compact_features.numel());
                CUDA_CHECK(cudaMemcpy(feats.data(), compact_features.data, compact_features.bytes(),
                                      cudaMemcpyDeviceToHost));
                static int feats_index = 0;
                const std::string ft_path = "/tmp/oracle_feats_" + std::to_string(feats_index++) + ".bin";
                FILE* f = std::fopen(ft_path.c_str(), "wb");
                if (f != nullptr) {
                    std::fwrite(feats.data(), 2, feats.size(), f);
                    std::fclose(f);
                }
                std::fprintf(stderr, "[dflash2] feats dumped n=%zu\n", feats.size());
                const auto cache = dflash_state(state).local_layer(0);
                std::vector<std::uint16_t> ks(static_cast<std::size_t>(cache.k.ne[0]) * 32);
                CUDA_CHECK(cudaMemcpy(ks.data(), cache.k.data, ks.size() * 2,
                                      cudaMemcpyDeviceToHost));
                f = std::fopen("/tmp/oracle_cachek.bin", "wb");
                if (f != nullptr) {
                    std::fwrite(ks.data(), 2, ks.size(), f);
                    std::fclose(f);
                }
                std::fprintf(stderr, "[dflash2] cache k dumped ne0=%d ne1=%d ne2=%d\n", cache.k.ne[0],
                             cache.k.ne[1], cache.k.ne[2]);
            }
            std::fprintf(stderr, "\n");
        }

        if (state.execution.model.features.speculative == SpeculativeBackend::DFlash2) {
            propose_dflash2_batch_impl<Variant>(state, frame, batch_size, k, envelopes);
        } else {
            propose_batch_impl<Variant>(state, frame, batch_size, k, envelopes);
        }
        ops::speculative_prepare_verify_ids(anchors, drafts, extents, verify_ids,
                                            state.execution.device.stream());
        if (std::getenv("NINFER_DFLASH2_DUMP") != nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream()));
            std::vector<std::int32_t> dh(drafts.numel()), vh(verify_ids.numel()),
                xh(extents.numel());
            CUDA_CHECK(cudaMemcpy(dh.data(), drafts.data, drafts.bytes(), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(vh.data(), verify_ids.data, verify_ids.bytes(),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(xh.data(), extents.data, extents.bytes(),
                                  cudaMemcpyDeviceToHost));
            std::fprintf(stderr, "[dflash2] accept inputs: extent=%d drafts:", xh[0]);
            for (auto v : dh) { std::fprintf(stderr, " %d", v); }
            std::fprintf(stderr, " verify:");
            for (auto v : vh) { std::fprintf(stderr, " %d", v); }
            std::fprintf(stderr, "\n");
        }

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, nullptr, state.execution.ple_state);
        // The target's Flash-Next layers consume PLE rows for their verify columns.
        Tensor ple_embeddings;
        if (state.ple_embeddings != nullptr) {
            ple_embeddings = state.ple_embeddings->slice(2, 0, batch_size);
            card.set_ple_embeddings(&ple_embeddings);
        }
        if (state.ple_stage_mailbox != nullptr) {
            Tensor ple_history = frame.ple_history.slice(1, 0, batch_size);
            ops::flash_next_ple_publish_ids(verify_ids, ple_history, state.ple_stage_mailbox,
                                            state.execution.device.stream());
            card.set_ple_stage(state.ple_stage_mailbox, state.ple_stage_staging);
        }
        DFlashFeatureSink sink =
            batch_feature_sink_impl<Variant>(state, active_lanes, valid_columns, width, batch_size);
        {
            nvtx::ScopedRange target_range(nvtx::Name::DecodeDFlashTarget, nvtx::Category::DFlash,
                                           static_cast<std::uint64_t>(width) * batch_size);
            target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                 TargetVerifyFrameView{
                                     .ids                     = verify_ids,
                                     .cache_positions         = target_positions,
                                     .rope_positions          = target_positions,
                                     .valid_columns           = valid_columns,
                                     .kv_table_rows           = text_rows,
                                     .state_source_slots      = state_sources,
                                     .state_destination_slots = state_destinations,
                                     .target_hidden           = target_hidden,
                                     .target_logits           = target_logits,
                                     .target_tokens           = target_tokens,
                                     .drafts                  = drafts,
                                     .current_extents         = extents,
                                     .frontiers               = frontiers,
                                     .anchors                 = anchors,
                                     .licensed_tokens         = licensed_tokens,
                                     .licensed_counts         = licensed_counts,
                                     .accepted_drafts         = accepted,
                                     .selected_hidden         = selected_hidden,
                                     .replay_records          = state.execution.replay_records,
                                     .sampling                = frame.sampling,
                                     .feature_sink            = &sink,
                                 },
                                 target_envelope);
        }
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_8_flash_next::DFlashDecodeEgress),
                                   cudaMemcpyDeviceToHost, state.execution.device.stream()));
        if (std::getenv("NINFER_DFLASH2_DUMP") != nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream()));
            std::array<std::int32_t, 8> accepted_host{};
            CUDA_CHECK(cudaMemcpy(accepted_host.data(), accepted.data,
                                  std::min<std::size_t>(sizeof(std::int32_t) * batch_size,
                                                        static_cast<std::size_t>(accepted.bytes())),
                                  cudaMemcpyDeviceToHost));
            std::array<std::int32_t, 8> licensed{};
            CUDA_CHECK(cudaMemcpy(licensed.data(), licensed_counts.data,
                                  std::min<std::size_t>(sizeof(std::int32_t) * batch_size,
                                                        static_cast<std::size_t>(licensed_counts.bytes())),
                                  cudaMemcpyDeviceToHost));
            {
                std::array<std::int32_t, 8> argmax{};
                CUDA_CHECK(cudaMemcpy(argmax.data(), frame.target_argmax.data,
                                      std::min<std::size_t>(sizeof(std::int32_t) * width * batch_size,
                                                            frame.target_argmax.bytes()),
                                      cudaMemcpyDeviceToHost));
                std::fprintf(stderr, "[dflash2] target argmax:");
                for (int i = 0; i < width * batch_size; ++i) { std::fprintf(stderr, " %d", argmax[i]); }
                std::fprintf(stderr, "\n");
            }
            static int round_index = 0;
            {
                const std::string tl_path = "/tmp/oracle_target_logits_" + std::to_string(round_index) + ".bin";
                FILE* f = std::fopen(tl_path.c_str(), "wb");
                if (f != nullptr) {
                    std::vector<std::uint16_t> all(frame.target_logits.numel());
                    CUDA_CHECK(cudaMemcpy(all.data(), frame.target_logits.data,
                                          frame.target_logits.bytes(), cudaMemcpyDeviceToHost));
                    std::fwrite(all.data(), 2, all.size(), f);
                    std::fclose(f);
                }
                const std::string th_path = "/tmp/oracle_target_hidden_" + std::to_string(round_index) + ".bin";
                f = std::fopen(th_path.c_str(), "wb");
                if (f != nullptr) {
                    std::vector<std::uint16_t> all(target_hidden.numel());
                    CUDA_CHECK(cudaMemcpy(all.data(), target_hidden.data, target_hidden.bytes(),
                                          cudaMemcpyDeviceToHost));
                    std::fwrite(all.data(), 2, all.size(), f);
                    std::fclose(f);
                }
            }
            std::fprintf(stderr, "[dflash2] round %d k=%u accepted:", round_index++, k);
            for (int i = 0; i < batch_size; ++i) {
                std::fprintf(stderr, " %d(licensed %d)", accepted_host[i], licensed[i]);
            }
            std::fprintf(stderr, "\n");
        }
    };
}

} // namespace

DFlashFeatureSink dflash_feature_sink(PrefillContext& state,
                                      DFlashFeatureSink::PrefillConsumer consume_prefill) {
    return prefill_feature_sink_impl<Variant>(state, std::move(consume_prefill));
}

void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl<Variant>(state, features, positions, commit_counts, lanes, table_rows,
                                 envelope);
}

void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl<Variant>(state, features, positions, commit_counts, lanes, table_rows,
                                 envelope);
}

void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t k, DFlashEnvelopes envelopes,
                                 ops::CausalAttentionExecutionEnvelope target_envelope,
                                 DecodeGraphDefinition& definition) {
    auto body = dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
    capture_graph(state, definition, body);
}

void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         DFlashEnvelopes envelopes,
                         ops::CausalAttentionExecutionEnvelope target_envelope,
                         DecodeGraphExecutable* executable) {
    auto body = dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
    run_prepared(state, executable, body);
}

} // namespace
  // ninfer::models::qwen3_8_flash_next::detail::NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS::schedule
