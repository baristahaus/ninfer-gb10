#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/sparse_moe.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class SparseMoeSmallTD3Schedule : std::uint8_t;
enum class SparseMoeSmallTD4Schedule : std::uint8_t;

struct SparseMoeDecodePlan {
    std::size_t workspace_bytes = 0;
    // L2 cache hint resolved from the caller's SparseMoeHints. No numeric effect.
    const void* next_weight_prefetch       = nullptr;
    std::size_t next_weight_prefetch_bytes = 0;
};

struct SparseMoeDecodeWorkspace {
    Tensor ids;
    Tensor alpha;
    Tensor shared_scale;
    Tensor scratch;
    // SP-7 EXL3 (single-token decode route): the gate/up inner-product vectors D3a leaves
    // for D4a's output transforms, and the per-path down results D4a stages for D4b's
    // block-diagonal output transform, route-weighted sum and residual epilogue.
    Tensor exl3_z;
    Tensor exl3_down;
};

template <class Arena>
SparseMoeDecodeWorkspace allocate_sparse_moe_decode_workspace(Arena& arena, bool exl3 = false,
                                                              const SparseMoeShape& shape =
                                                                  kSparseMoe35B) {
    const int top_k        = shape.top_k;
    const int hidden       = shape.hidden;
    const int intermediate = shape.intermediate;
    SparseMoeDecodeWorkspace out;
    out.ids          = arena.alloc(DType::I32, {top_k}, 16);
    out.alpha        = arena.alloc(DType::FP32, {top_k}, 16);
    out.shared_scale = arena.alloc(DType::FP32, {1}, 4);
    // D1 uses the first experts+1 values as scores. D3 then reuses the same lifetime for the
    // SwiGLU results consumed by D4. On the EXL3 route only the shared expert writes its
    // activation slot here; the routed experts transform inside D4a. The score row stays
    // inside the intermediate-sized row 0: 35B needs 257 <= 512 and 125B needs 513 <= 640.
    out.scratch = arena.alloc(DType::FP32, {top_k + 1, intermediate}, 256);
    if (exl3) {
        out.exl3_z    = arena.alloc(DType::FP32, {top_k, 2, intermediate}, 256);
        out.exl3_down = arena.alloc(DType::FP32, {top_k + 1, hidden}, 256);
    }
    return out;
}

[[nodiscard]] std::size_t sparse_moe_decode_workspace_bytes();
[[nodiscard]] std::size_t
sparse_moe_exl3_decode_workspace_bytes(const SparseMoeShape& shape = kSparseMoe35B);
[[nodiscard]] SparseMoeDecodePlan resolve_sparse_moe_decode_plan(QType routed_gate_up,
                                                                 QType routed_down,
                                                                 const SparseMoeHints& hints = {},
                                                                 const SparseMoeShape& shape =
                                                                     kSparseMoe35B);

void sparse_moe_decode_launch_d3_small_t(const Tensor& x, const SparseMoeWeights& weights,
                                         const int* token_ids, float* token_activations,
                                         std::int32_t tokens, SparseMoeSmallTD3Schedule schedule,
                                         cudaStream_t stream,
                                         const int* adaptive_route_jobs = nullptr);
void sparse_moe_decode_launch_d4_small_t(const SparseMoeWeights& weights, Tensor& destination,
                                         const int* token_ids, const float* token_alpha,
                                         const float* shared_scale, const float* token_activations,
                                         std::int32_t tokens, SparseMoeSmallTD4Schedule schedule,
                                         cudaStream_t stream,
                                         const int* adaptive_route_jobs = nullptr);
void sparse_moe_decode_launch(const Tensor& x, const SparseMoeWeights& weights, Tensor& destination,
                              const SparseMoeDecodePlan& plan,
                              const SparseMoeDecodeWorkspace& workspace,
                              const SparseMoeShape& shape, cudaStream_t stream);

} // namespace ninfer::ops::detail
