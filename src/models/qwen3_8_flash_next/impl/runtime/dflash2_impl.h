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
        const bool dump_enabled = std::getenv("NINFER_DFLASH2_DUMP") != nullptr;
        const auto scan = [&](const char* label, const Tensor& t) {
            CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream()));
            std::vector<std::uint16_t> all(static_cast<std::size_t>(t.bytes()) / 2);
            CUDA_CHECK(cudaMemcpy(all.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost));
            std::size_t nan = 0, zero = 0;
            std::int64_t first_nan = -1;
            for (std::size_t i = 0; i < all.size(); ++i) {
                const auto b16 = all[i];
                if ((b16 & 0x7f80u) == 0x7f80u && (b16 & 0x007fu) != 0) {
                    ++nan;
                    if (first_nan < 0) { first_nan = static_cast<std::int64_t>(i); }
                } else if (b16 == 0) {
                    ++zero;
                }
            }
            std::fprintf(stderr, "[dflash2] %s scan: n=%zu nan=%zu zero=%zu first_nan=%lld\n",
                         label, all.size(), nan, zero, static_cast<long long>(first_nan));
        };
        const auto probe = [&](const char* label, const Tensor& t, int count) {
            // The compute stream is non-blocking; order the host read after it.
            CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream()));
            std::array<std::uint16_t, 16> bits{};
            const auto bytes = std::min<std::size_t>(bits.size() * sizeof(std::uint16_t),
                                                     static_cast<std::size_t>(t.bytes()));
            CUDA_CHECK(cudaMemcpy(bits.data(), t.data, bytes, cudaMemcpyDeviceToHost));
            if (dump_enabled && label[0] == 'l') {
                std::vector<std::uint16_t> all(static_cast<std::size_t>(t.bytes()) / 2);
                CUDA_CHECK(cudaMemcpy(all.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost));
                std::size_t nan = 0, zero = 0;
                std::int64_t first_nan = -1;
                for (std::size_t i = 0; i < all.size(); ++i) {
                    const auto bits16 = all[i];
                    if ((bits16 & 0x7f80u) == 0x7f80u && (bits16 & 0x007fu) != 0) {
                        ++nan;
                        if (first_nan < 0) { first_nan = static_cast<std::int64_t>(i); }
                    } else if (bits16 == 0) { ++zero; }
                }
                std::fprintf(stderr, "[dflash2] %s scan: elements=%zu nan=%zu zero=%zu first_nan=%lld\n",
                             label, all.size(), nan, zero, static_cast<long long>(first_nan));
            }
            std::fprintf(stderr, "[dflash2] %s (bf16 bits):", label);
            for (std::size_t i = 0; i < bytes / sizeof(std::uint16_t); ++i) {
                std::fprintf(stderr, " %04x", bits[i]);
            }
            std::fprintf(stderr, "\n");
        };
        ops::prepare_masked_block(anchors, frontiers, valid_columns, Config::mask_token, ids,
                                  positions, state.execution.device.stream());
        if (dump_enabled) {
            CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream()));
            std::vector<std::int32_t> block_ids(static_cast<std::size_t>(columns)),
                block_positions(static_cast<std::size_t>(columns));
            CUDA_CHECK(cudaMemcpy(block_ids.data(), ids.data, ids.bytes(), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(block_positions.data(), positions.data, positions.bytes(),
                                  cudaMemcpyDeviceToHost));
            std::fprintf(stderr, "[dflash2] block ids:");
            for (auto v : block_ids) { std::fprintf(stderr, " %d", v); }
            std::fprintf(stderr, " positions:");
            for (auto v : block_positions) { std::fprintf(stderr, " %d", v); }
            std::fprintf(stderr, "\n");
        }
        Tensor residual = work.alloc(DType::BF16, {Config::hidden, width, batch_size});
        Tensor flat_residual = residual.view({Config::hidden, columns});
        ops::embedding(ids.view({columns}), state.execution.model.token_embedding, flat_residual,
                       state.execution.device.execution_view());
        if (dump_enabled) { probe("residual@embed", residual, 6); }

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
                if (dump_enabled && layer == 0) {
                    probe("prepared@0", branch.prepared, 6);
                    probe("finish_delta@0", branch.finish_delta, 6);
                    // Oracle dump: one column of the residual, the norm weight, the prepared
                    // value, and the conv base kernel + projection, for a host-side check of
                    // the dynamic-conv prepare math.
                    CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream()));
                    const auto dump_column = [&](const char* path, const Tensor& t,
                                                 int rows, int stride, int column) {
                        std::vector<std::uint16_t> host(static_cast<std::size_t>(rows));
                        for (int r = 0; r < rows; ++r) {
                            CUDA_CHECK(cudaMemcpy(host.data() + r,
                                                  static_cast<const std::uint8_t*>(t.data) +
                                                      (static_cast<std::size_t>(r) * stride + column) * 2,
                                                  2, cudaMemcpyDeviceToHost));
                        }
                        FILE* f = std::fopen(path, "wb");
                        if (f != nullptr) {
                            std::fwrite(host.data(), 2, host.size(), f);
                            std::fclose(f);
                        }
                    };
                    dump_column("/tmp/oracle_residual.bin", residual, residual.numel(), 1, 0);
                    dump_column("/tmp/oracle_prepared.bin", branch.prepared, branch.prepared.numel(),
                                1, 0);
                    {
                        std::vector<std::uint16_t> norm(2560);
                        CUDA_CHECK(cudaMemcpy(norm.data(), weight.input_norm.data, 5120,
                                              cudaMemcpyDeviceToHost));
                        FILE* f = std::fopen("/tmp/oracle_norm.bin", "wb");
                        if (f != nullptr) { std::fwrite(norm.data(), 2, norm.size(), f); std::fclose(f); }
                        std::vector<std::uint16_t> base(2560 * 4);
                        CUDA_CHECK(cudaMemcpy(base.data(), weight.attention_conv_base_kernel.data,
                                              5120 * 4, cudaMemcpyDeviceToHost));
                        f = std::fopen("/tmp/oracle_base.bin", "wb");
                        if (f != nullptr) { std::fwrite(base.data(), 2, base.size(), f); std::fclose(f); }
                        std::vector<std::uint16_t> proj(640 * 2560);
                        CUDA_CHECK(cudaMemcpy(proj.data(), weight.attention_conv_kernel_projection.qdata,
                                              640 * 2560 * 2, cudaMemcpyDeviceToHost));
                        f = std::fopen("/tmp/oracle_proj.bin", "wb");
                        if (f != nullptr) { std::fwrite(proj.data(), 2, proj.size(), f); std::fclose(f); }
                        std::vector<std::uint16_t> delta(branch.finish_delta.numel());
                        CUDA_CHECK(cudaMemcpy(delta.data(), branch.finish_delta.data,
                                              branch.finish_delta.bytes(), cudaMemcpyDeviceToHost));
                        f = std::fopen("/tmp/oracle_delta.bin", "wb");
                        if (f != nullptr) { std::fwrite(delta.data(), 2, delta.size(), f); std::fclose(f); }
                    }
                }
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
                if (dump_enabled && layer == 0) {
                    probe("q@0", query_flat, 6);
                    probe("k@0", key_flat, 6);
                    probe("v@0", value_flat, 6);
                }
                if (dump_enabled && layer == 0) {
                    CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream()));
                    const auto dump_full = [&](const char* path, const Tensor& t) {
                        std::vector<std::uint16_t> host(static_cast<std::size_t>(t.bytes()) / 2);
                        CUDA_CHECK(cudaMemcpy(host.data(), t.data, t.bytes(),
                                              cudaMemcpyDeviceToHost));
                        FILE* f = std::fopen(path, "wb");
                        if (f != nullptr) { std::fwrite(host.data(), 2, host.size(), f); std::fclose(f); }
                    };
                    dump_full("/tmp/l0_prepared.bin", branch.prepared);
                    dump_full("/tmp/l0_q.bin", query_flat);
                    dump_full("/tmp/l0_k.bin", key_flat);
                    dump_full("/tmp/l0_v.bin", value_flat);
                    {
                        std::vector<std::uint16_t> norm(2560);
                        CUDA_CHECK(cudaMemcpy(norm.data(), weight.query_norm.data, 5120,
                                              cudaMemcpyDeviceToHost));
                        FILE* f = std::fopen("/tmp/l0_qnorm.bin", "wb");
                        if (f != nullptr) { std::fwrite(norm.data(), 2, norm.size(), f); std::fclose(f); }
                        CUDA_CHECK(cudaMemcpy(norm.data(), weight.key_norm.data, 5120,
                                              cudaMemcpyDeviceToHost));
                        f = std::fopen("/tmp/l0_knorm.bin", "wb");
                        if (f != nullptr) { std::fwrite(norm.data(), 2, norm.size(), f); std::fclose(f); }
                        std::vector<std::int32_t> pos(columns);
                        CUDA_CHECK(cudaMemcpy(pos.data(), positions.data, positions.bytes(),
                                              cudaMemcpyDeviceToHost));
                        f = std::fopen("/tmp/l0_positions.bin", "wb");
                        if (f != nullptr) { std::fwrite(pos.data(), 4, pos.size(), f); std::fclose(f); }
                    }
                }
                ops::rmsnorm_rope(positions, weight.query_norm, weight.key_norm, query, key,
                                  Config::rope_theta,
                                  state.execution.device.execution_view());
                if (dump_enabled && layer == 0) {
                    CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream()));
                    const auto dump_full2 = [&](const char* path, const Tensor& t) {
                        std::vector<std::uint16_t> host(static_cast<std::size_t>(t.bytes()) / 2);
                        CUDA_CHECK(cudaMemcpy(host.data(), t.data, t.bytes(),
                                              cudaMemcpyDeviceToHost));
                        FILE* f = std::fopen(path, "wb");
                        if (f != nullptr) { std::fwrite(host.data(), 2, host.size(), f); std::fclose(f); }
                    };
                    dump_full2("/tmp/l0_q_roped.bin", query_flat);
                    dump_full2("/tmp/l0_k_roped.bin", key_flat);
                }
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
                if (dump_enabled && layer == 0) { probe("attention@0", attention, 6); }
                if (dump_enabled && layer == 0) {
                    CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream()));
                    const auto dump_full3 = [&](const char* path, const Tensor& t) {
                        std::vector<std::uint16_t> host(static_cast<std::size_t>(t.bytes()) / 2);
                        CUDA_CHECK(cudaMemcpy(host.data(), t.data, t.bytes(),
                                              cudaMemcpyDeviceToHost));
                        FILE* f = std::fopen(path, "wb");
                        if (f != nullptr) { std::fwrite(host.data(), 2, host.size(), f); std::fclose(f); }
                    };
                    dump_full3("/tmp/l0_attention.bin", attention);
                }
                finish_branch<Config>(
                    attention.view({Config::query_size, width, batch_size}), weight.output,
                    weight.attention_conv_base_kernel, branch, residual, work,
                    state.execution.device.execution_view());
                if (dump_enabled && layer == 0) {
                    probe("residual@attn0", residual, 6);
                    CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream()));
                    std::vector<std::uint16_t> host(static_cast<std::size_t>(residual.bytes()) / 2);
                    CUDA_CHECK(cudaMemcpy(host.data(), residual.data, residual.bytes(),
                                          cudaMemcpyDeviceToHost));
                    FILE* f = std::fopen("/tmp/l0_residual_after_attn.bin", "wb");
                    if (f != nullptr) { std::fwrite(host.data(), 2, host.size(), f); std::fclose(f); }
                }
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
                if (dump_enabled && layer == 0) {
                    scan("intermediate@L0", intermediate);
                    scan("prepared_mlp@L0", branch.prepared);
                }
                finish_branch<Config>(intermediate, weight.mlp_down, weight.mlp_conv_base_kernel,
                                      branch, residual, work,
                                      state.execution.device.execution_view());
                if (dump_enabled) {
                    char label[32];
                    std::snprintf(label, sizeof(label), "residual@L%d", layer);
                    scan(label, residual);
                }
            }
        }

        const std::int32_t mask_columns = static_cast<std::int32_t>(k) * batch_size;
        Tensor proposal_hidden = work.alloc(DType::BF16, {Config::hidden, mask_columns});
        if (dump_enabled) { scan("residual@final", residual); }
        ops::rmsnorm_pack_tail(residual, state.execution.model.dflash2->final_norm,
                               proposal_hidden, state.execution.device.execution_view());
        if (dump_enabled) { scan("hidden@final", proposal_hidden); }
        Tensor candidates = frame.proposal_candidates.slice(2, 0, batch_size);
        Tensor scores     = frame.proposal_scores.slice(2, 0, batch_size);
        Tensor logits     = work.alloc(DType::BF16, {TextConfig::output_rows, mask_columns});

        // The BF16 draft head has no fused linear_topk profile; project the logits and take the
        // top candidates per column.
        if (dump_enabled) {
            // proposal_hidden is [D, columns] row-major: a column is strided by `columns`.
            std::array<std::uint16_t, 18> cols{};
            const auto* base = static_cast<const std::uint8_t*>(proposal_hidden.data);
            for (int col = 0; col < static_cast<int>(k) * batch_size && col < 3; ++col) {
                for (int row = 0; row < 6; ++row) {
                    CUDA_CHECK(cudaMemcpy(cols.data() + col * 6 + row,
                                          base + (static_cast<std::size_t>(row) * (k * batch_size) + col) * 2,
                                          sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
                }
            }
            std::fprintf(stderr, "[dflash2] hidden columns (bf16 bits):");
            for (auto b : cols) { std::fprintf(stderr, " %04x", b); }
            std::fprintf(stderr, "\n");
        }
        if (dump_enabled) {
            // Sentinel fill: tells whether the projection wrote at all.
            CUDA_CHECK(cudaMemsetAsync(logits.data, 0, static_cast<std::size_t>(logits.bytes()),
                                       state.execution.device.stream()));
        }
        ops::linear(proposal_hidden, state.execution.model.output_head, logits,
                    state.execution.device.execution_view());
        if (dump_enabled) {
            CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream()));
            // Token-major vocabulary columns: column c occupies [c*V, (c+1)*V).
            std::vector<std::uint16_t> all(static_cast<std::size_t>(logits.bytes()) / 2);
            CUDA_CHECK(cudaMemcpy(all.data(), logits.data, logits.bytes(),
                                  cudaMemcpyDeviceToHost));
            std::size_t nan = 0, zero = 0;
            std::int64_t first_nan = -1;
            for (std::size_t i = 0; i < all.size(); ++i) {
                const auto b16 = all[i];
                if ((b16 & 0x7f80u) == 0x7f80u && (b16 & 0x007fu) != 0) {
                    ++nan;
                    if (first_nan < 0) { first_nan = static_cast<std::int64_t>(i); }
                } else if (b16 == 0) {
                    ++zero;
                }
            }
            std::fprintf(stderr,
                         "[dflash2] logits scan: n=%zu nan=%zu zero=%zu first_nan=%lld cols:",
                         all.size(), nan, zero, static_cast<long long>(first_nan));
            for (int c = 0; c < static_cast<int>(k) * batch_size && c < 3; ++c) {
                std::fprintf(stderr, " [%d]", c);
                for (int r = 0; r < 4; ++r) {
                    std::fprintf(stderr, " %04x", all[static_cast<std::size_t>(c) * 248320 + r]);
                }
            }
            std::fprintf(stderr, "\n");
        }
        ops::column_topk(logits.view({TextConfig::output_rows, static_cast<std::int32_t>(k),
                                      batch_size}),
                         candidates, scores, state.execution.device.execution_view());
        if (std::getenv("NINFER_DFLASH2_DUMP") != nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream()));
            {
                static int prop_index = 0;
                const std::string lg_path = "/tmp/oracle_prop_logits_" + std::to_string(prop_index) + ".bin";
                FILE* f = std::fopen(lg_path.c_str(), "wb");
                if (f != nullptr) {
                    std::vector<std::uint16_t> all(logits.numel());
                    CUDA_CHECK(cudaMemcpy(all.data(), logits.data, logits.bytes(),
                                          cudaMemcpyDeviceToHost));
                    std::fwrite(all.data(), 2, all.size(), f);
                    std::fclose(f);
                }
                f = std::fopen("/tmp/oracle_candidates.bin", "wb");
                if (f != nullptr) {
                    std::vector<std::int32_t> all(candidates.numel());
                    CUDA_CHECK(cudaMemcpy(all.data(), candidates.data, candidates.bytes(),
                                          cudaMemcpyDeviceToHost));
                    std::fwrite(all.data(), 4, all.size(), f);
                    std::fclose(f);
                }
                f = std::fopen("/tmp/oracle_scores.bin", "wb");
                if (f != nullptr) {
                    std::vector<float> all(scores.numel());
                    CUDA_CHECK(cudaMemcpy(all.data(), scores.data, scores.bytes(),
                                          cudaMemcpyDeviceToHost));
                    std::fwrite(all.data(), 4, all.size(), f);
                    std::fclose(f);
                }
                std::vector<std::int32_t> anchor_host(batch_size);
                CUDA_CHECK(cudaMemcpy(anchor_host.data(), anchors.data, anchors.bytes(),
                                      cudaMemcpyDeviceToHost));
                std::fprintf(stderr, "[dflash2] prop %d topk anchor0=%d k=%u b=%d\n", prop_index++, anchor_host[0], k,
                             batch_size);
            }
        }
        if (std::getenv("NINFER_DFLASH2_DUMP") != nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream()));
            const auto dump = [&](const char* label, const Tensor& t, int count) {
                std::array<float, 64> host{};
                std::array<std::int32_t, 64> ints{};
                std::fprintf(stderr, "[dflash2] %s:", label);
                if (t.dtype == DType::FP32) {
                    CUDA_CHECK(cudaMemcpy(host.data(), t.data,
                                          std::min<std::size_t>(sizeof(float) * count,
                                                                static_cast<std::size_t>(t.bytes())),
                                          cudaMemcpyDeviceToHost));
                    for (int i = 0; i < count; ++i) { std::fprintf(stderr, " %.4g", host[i]); }
                } else {
                    CUDA_CHECK(cudaMemcpy(ints.data(), t.data,
                                          std::min<std::size_t>(sizeof(std::int32_t) * count,
                                                                static_cast<std::size_t>(t.bytes())),
                                          cudaMemcpyDeviceToHost));
                    for (int i = 0; i < count; ++i) { std::fprintf(stderr, " %d", ints[i]); }
                }
                std::fprintf(stderr, "\n");
            };
            dump("candidates", candidates, 16);
            dump("scores", scores, static_cast<int>(k) * batch_size);
            dump("hidden", proposal_hidden, 6);
            dump("logits", logits, 6);
        }

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
        if (std::getenv("NINFER_DFLASH2_DUMP") != nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream()));
            std::array<std::int32_t, 64> host{};
            const auto bytes = std::min<std::size_t>(host.size() * sizeof(std::int32_t),
                                                     drafts.numel() * sizeof(std::int32_t));
            CUDA_CHECK(cudaMemcpy(host.data(), drafts.data, bytes, cudaMemcpyDeviceToHost));
            std::fprintf(stderr, "[dflash2] drafts k=%u b=%d:", k, batch_size);
            for (std::size_t i = 0; i < bytes / sizeof(std::int32_t); ++i) {
                std::fprintf(stderr, " %d", host[i]);
            }
            std::fprintf(stderr, "\n");
        }
        work.reset();
    }
}

} // namespace
} // namespace ninfer::models::qwen3_8_flash_next::detail::NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS::schedule
