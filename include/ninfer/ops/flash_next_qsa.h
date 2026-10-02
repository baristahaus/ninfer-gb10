#pragma once
#include "core/weight.h"

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

class Bf16GemmContext;

inline constexpr std::int32_t kFlashNextQsaSelectedTokens = 2051;

struct FlashNextQsaIndexControl {
    // Optional persistent [2051,T] destination for the indices selected by this call.
    Tensor* selected_indices = nullptr;
    // Optional persistent [2051,T] indices selected by a target-aligned call. When present,
    // index-query projection, scoring, and top-k selection are skipped while K/V/index-key
    // cache publication still occurs for the current tokens.
    const Tensor* reused_indices = nullptr;
};

struct FlashNextQsaWeights {
    // Per-head packed rows: 24 repetitions of [query(256), gate(256)].
    Weight query_gate;
    Weight key;
    Weight value;
    Weight output;
    Tensor query_norm;
    Tensor key_norm;
    Weight index_query;
    Weight index_key;
    Tensor index_query_norm;
    Tensor index_key_norm;
};

// Workspace for any call of 1..tokens rows whose cache holds at most max_context keys: a
// workspace sized for a prefill chunk also serves the chunk's short tail and decode calls.
[[nodiscard]] std::size_t flash_next_qsa_workspace_capacity_bytes(std::int32_t tokens,
                                                                  std::uint32_t max_context);

// The checkpoint stores Q and its sigmoid gate interleaved per head as
// 24 repetitions of [query(256), gate(256)]. Project and unpack that physical
// representation into the two logical contiguous outputs.
[[nodiscard]] std::size_t flash_next_query_gate_workspace_capacity_bytes(std::int32_t tokens);
void flash_next_project_query_gate(const Tensor& input, const Weight& query_gate,
                                   Tensor& query, Tensor& gate,
                                   WorkspaceArena& workspace, cudaStream_t stream,
                                   Bf16GemmContext* bf16_gemm = nullptr);

void flash_next_expand_text_positions(const Tensor& positions, Tensor& mrope_positions,
                                      cudaStream_t stream);

// Complete exact QSA attention leaf. `cache_positions` is I32 [W,B], `rope_positions`
// is I32 [W,B,3], `valid_columns` and `table_rows` are I32 [B]. The cache owns BF16 or
// row-scaled FP8 E4M3 main K/V followed by auxiliary raw-index-key BF16 and MRoPE-position I32
// planes.
void flash_next_qsa(const Tensor& input, const Tensor& cache_positions,
                    const Tensor& rope_positions, const Tensor& valid_columns,
                    const Tensor& table_rows, const FlashNextQsaWeights& weights,
                    PagedKVBatchLayerView cache, CausalAttentionExecutionEnvelope envelope,
                    Tensor& destination,
                    WorkspaceArena& workspace, cudaStream_t stream,
                    Bf16GemmContext* bf16_gemm = nullptr,
                    FlashNextQsaIndexControl index_control = {});

// Select one target-aligned QSA index row per request. `indices` is [2051,W,B], selectors is
// I32 [B], and destination is [2051,B]. This is the persistent MTP index-sharing transition.
void flash_next_qsa_select_indices(const Tensor& indices, const Tensor& selectors,
                                   Tensor& destination, cudaStream_t stream);

} // namespace ninfer::ops
