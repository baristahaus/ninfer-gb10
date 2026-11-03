#include "models/qwen3_8_flash_next/impl/runtime/instance.h"
#include "models/qwen3_8_flash_next/impl/runtime/schedule.h"

#include "core/nvtx.h"
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/column_topk.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rmsnorm_pack_tail.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/sliding_window_attention.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdexcept>

namespace ninfer::models::qwen3_8_flash_next::detail::NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS::
    schedule {
namespace {

// One dynamic-conv branch: the RMSNorm'd input with its two-tap coefficients prepared, and the
// finish delta the projection's grouped convolution reuses.
struct DFlash2Branch {
    Tensor prepared;
    Tensor finish_delta;
};

template <class Config>
DFlash2Branch dflash2_branch(MemoryArena& work, std::int32_t width, std::int32_t batch_size) {
    return {work.alloc(DType::BF16, {Config::hidden, width, batch_size}),
            work.alloc(DType::BF16, {Config::groups, 2, width, batch_size})};
}

template <class Config>
void prepare_branch(const Tensor& residual, const Tensor& norm, const Tensor& base_kernel,
                    const Weight& kernel_projection, DFlash2Branch& branch, MemoryArena& work,
                    CudaExecutionView execution) {
    const std::int32_t width = residual.ne[1], batch_size = residual.ne[2];
    MemoryArena scratch(work.alloc_bytes(
        ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(width, width,
                                                                          batch_size, batch_size)));
    ops::rmsnorm_dynamic_grouped_conv_prepare(residual, norm, Config::rms_epsilon, base_kernel,
                                              kernel_projection, branch.prepared,
                                              branch.finish_delta, scratch, execution);
}

template <class Config>
void finish_branch(const Tensor& input, const Weight& projection, const Tensor& base_kernel,
                   DFlash2Branch& branch, Tensor& residual, MemoryArena& work,
                   CudaExecutionView execution) {
    const std::int32_t width = input.ne[1], batch_size = input.ne[2];
    MemoryArena scratch(work.alloc_bytes(
        ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
            input.ne[0], width, width, batch_size, batch_size)));
    ops::linear_dynamic_grouped_conv_add(input, projection, base_kernel, branch.finish_delta,
                                         residual, scratch, execution);
}

// The DFlash2 draft's proposal pass: masked block drafting with the two-tap dynamic
// convolutions, then the candidate selector's coherent path. The draft reuses the target's
// token embedding and output head; the context K/V come from the target's tap layers through
// the feature projection, exactly as the v1 draft's do.
template <class V>
void propose_dflash2_batch_impl(DFlashBatchContext& state,
                                qwen3_8_flash_next::DFlashDecodeState& frame,
                                std::int32_t batch_size, std::uint32_t k,
                                DFlashEnvelopes envelopes) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash2 proposal is unavailable for this target");
    } else {
        using Config             = typename V::DFlashConfig;
        const std::int32_t width = static_cast<std::int32_t>(k) + 1;
        const std::int32_t columns = width * batch_size;
        nvtx::ScopedRange proposal_range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash,
                                         static_cast<std::uint64_t>(columns));
        Tensor anchors            = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers          = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor valid_columns      = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor ids                = frame.proposal_ids.slice(1, 0, batch_size);
        Tensor positions          = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor drafts             = frame.draft_tokens.slice(1, 0, batch_size);

        auto& work = state.execution.work;
        work.reset();
        ;
        ;
        ops::prepare_masked_block(anchors, frontiers, valid_columns, Config::mask_token, ids,
                                  positions, state.execution.device.stream());
        Tensor residual = work.alloc(DType::BF16, {Config::hidden, width, batch_size});
        Tensor flat_residual = residual.view({Config::hidden, columns});
        ops::embedding(ids.view({columns}), state.execution.model.token_embedding, flat_residual,
                       state.execution.device.execution_view());
        for (int layer = 0; layer < Config::layers; ++layer) {
            nvtx::ScopedRange layer_range(nvtx::Name::DFlashLayer, nvtx::Category::DFlash,
                                          static_cast<std::uint64_t>(layer));
            const auto& weight =
                state.execution.model.dflash2->layers.at(static_cast<std::size_t>(layer));
            {
                nvtx::ScopedRange attention_range(nvtx::Name::DFlashAttention,
                                                  nvtx::Category::Attention,
                                                  static_cast<std::uint64_t>(layer));
                auto attention_scope = work.scope();
                auto branch          = dflash2_branch<Config>(work, width, batch_size);
                prepare_branch<Config>(residual, weight.input_norm,
                                       weight.attention_conv_base_kernel,
                                       weight.attention_conv_kernel_projection, branch, work,
                                       state.execution.device.execution_view());
                Tensor query =
                    work.alloc(DType::BF16, {Config::head_dim, Config::query_heads, width,
                                             batch_size});
                Tensor key =
                    work.alloc(DType::BF16, {Config::head_dim, Config::kv_heads, width, batch_size});
                Tensor value =
                    work.alloc(DType::BF16, {Config::head_dim, Config::kv_heads, width, batch_size});
                Tensor query_flat = query.view({Config::query_size, columns});
                Tensor key_flat   = key.view({Config::kv_size, columns});
                Tensor value_flat = value.view({Config::kv_size, columns});
                std::array outputs{query_flat, key_flat, value_flat};
                ops::linear_split(branch.prepared.view({Config::hidden, columns}),
                                  weight.query_key_value, outputs,
                                  state.execution.device.execution_view());
                ops::rmsnorm_rope(positions, weight.query_norm, weight.key_norm, query, key,
                                  Config::rope_theta,
                                  state.execution.device.execution_view());
                Tensor attention =
                    work.alloc(DType::BF16, {Config::head_dim, Config::query_heads, width,
                                             batch_size});
                ops::sliding_window_attention(
                    query, key, value, positions, valid_columns, state_destinations,
                    {Config::head_dim, Config::query_heads, Config::kv_heads},
                    Config::local_capacity, Config::attention_scale,
                    dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                    envelopes.local, work, attention,
                    state.execution.device.execution_view());
                finish_branch<Config>(
                    attention.view({Config::query_size, width, batch_size}), weight.output,
                    weight.attention_conv_base_kernel, branch, residual, work,
                    state.execution.device.execution_view());
                }
            {
                nvtx::ScopedRange mlp_range(nvtx::Name::DFlashMlp, nvtx::Category::PostMixer,
                                            static_cast<std::uint64_t>(layer));
                auto mlp_scope = work.scope();
                auto branch    = dflash2_branch<Config>(work, width, batch_size);
                prepare_branch<Config>(residual, weight.post_attention_norm,
                                       weight.mlp_conv_base_kernel,
                                       weight.mlp_conv_kernel_projection, branch, work,
                                       state.execution.device.execution_view());
                Tensor intermediate =
                    work.alloc(DType::BF16, {Config::intermediate, width, batch_size});
                Tensor intermediate_flat = intermediate.view({Config::intermediate, columns});
                Tensor prepared_flat     = branch.prepared.view({Config::hidden, columns});
                ops::linear_swiglu(prepared_flat, weight.mlp_gate_up, intermediate_flat, work,
                                   state.execution.device.execution_view());
                finish_branch<Config>(intermediate, weight.mlp_down, weight.mlp_conv_base_kernel,
                                      branch, residual, work,
                                      state.execution.device.execution_view());
                }
        }

        const std::int32_t mask_columns = static_cast<std::int32_t>(k) * batch_size;
        Tensor proposal_hidden = work.alloc(DType::BF16, {Config::hidden, mask_columns});
        ops::rmsnorm_pack_tail(residual, state.execution.model.dflash2->final_norm,
                               proposal_hidden, state.execution.device.execution_view());
        Tensor candidates = frame.proposal_candidates.slice(2, 0, batch_size);
        Tensor scores     = frame.proposal_scores.slice(2, 0, batch_size);
        Tensor logits     = work.alloc(DType::BF16, {TextConfig::output_rows, mask_columns});

        // The BF16 draft head has no fused linear_topk profile; project the logits and take the
        // top candidates per column.
        ops::linear(proposal_hidden, state.execution.model.output_head, logits,
                    state.execution.device.execution_view());
        ops::column_topk(logits.view({TextConfig::output_rows, static_cast<std::int32_t>(k),
                                      batch_size}),
                         candidates, scores, state.execution.device.execution_view());
        Tensor projected = work.alloc(DType::BF16, {Config::selector_rank, mask_columns});
        ops::linear(proposal_hidden, state.execution.model.dflash2->selector_hidden_projection,
                    projected, state.execution.device.execution_view());

        if (std::getenv("NINFER_DFLASH2_BYPASS") != nullptr) {
            // Diagnostic: zero the selector's projected hidden so the edge score collapses to
            // the unary head score and the selector picks the per-column head argmax.
            CUDA_CHECK(cudaMemsetAsync(projected.data, 0,
                                       static_cast<std::size_t>(projected.bytes()),
                                       state.execution.device.stream()));
        }
        Tensor proposal_q = frame.proposal_q.slice(2, 0, batch_size);
        ops::candidate_selector_path(
            candidates, scores,
            projected.view({Config::selector_rank, static_cast<std::int32_t>(k), batch_size}),
            anchors, state.execution.model.dflash2->selector_predecessor_codebook,
            state.execution.model.dflash2->selector_successor_codebook, frontiers,
            frame.sampling, drafts, proposal_q, work,
            state.execution.device.execution_view());
        work.reset();
    }
}

} // namespace
} // namespace ninfer::models::qwen3_8_flash_next::detail::NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS::schedule
