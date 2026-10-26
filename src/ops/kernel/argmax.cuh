#pragma once

// Implements: include/ninfer/ops/argmax.h
// Match: contiguous BF16 [physical_rows,C], valid_rows <= physical_rows, with
// tuned routes qualified for the 248077-row and 131072-row vocabularies.
// Algorithm assumptions: one CTA reduces each blockDim.x-row tile and an atomic
// winner selects the exact value/lower-id maximum per column.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <climits>
#include <math_constants.h>

namespace ninfer::ops {

// Small-column execution uses 512 threads to reduce atomic contenders. Aggregate
// execution dispatches registered vocab profiles to a smaller block so the much
// larger 2-D grid exposes enough resident CTAs without oversized reductions.
inline constexpr int kArgmaxBlock          = 512;
inline constexpr int kArgmaxItemsPerThread = 1;
inline constexpr int kShortlistRerankTile = 512;
inline constexpr int kShortlistCandidatesPerTile = 2;

__device__ __forceinline__ bool argmax_better(float value, std::int32_t index, float best_value,
                                              std::int32_t best_index) {
    return value > best_value || (value == best_value && index < best_index);
}

__device__ __forceinline__ void argmax_warp_reduce(float& value, std::int32_t& index) {
    constexpr unsigned int kMask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_value        = __shfl_down_sync(kMask, value, offset);
        const std::int32_t other_index = __shfl_down_sync(kMask, index, offset);
        if (argmax_better(other_value, other_index, value, index)) {
            value = other_value;
            index = other_index;
        }
    }
}

__device__ __forceinline__ void argmax_block_reduce(float& value, std::int32_t& index) {
    __shared__ float warp_values[16];
    __shared__ std::int32_t warp_indices[16];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    argmax_warp_reduce(value, index);
    if (lane == 0) {
        warp_values[warp]  = value;
        warp_indices[warp] = index;
    }
    __syncthreads();

    value = (lane < (blockDim.x >> 5)) ? warp_values[lane] : -CUDART_INF_F;
    index = (lane < (blockDim.x >> 5)) ? warp_indices[lane] : INT32_MAX;
    if (warp == 0) { argmax_warp_reduce(value, index); }
}

__launch_bounds__(kArgmaxBlock) __global__
    void argmax_kernel(const __nv_bfloat16* logits, std::int32_t* out, std::int32_t valid_rows,
                       std::int32_t physical_rows) {
    const std::int32_t t    = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(t) * physical_rows;

    float best_value        = __bfloat162float(logits[base]);
    std::int32_t best_index = 0;
    for (std::int32_t v = static_cast<std::int32_t>(threadIdx.x); v < valid_rows; v += blockDim.x) {
        const float value = __bfloat162float(logits[base + v]);
        if (argmax_better(value, v, best_value, best_index)) {
            best_value = value;
            best_index = v;
        }
    }

    __shared__ float values[kArgmaxBlock];
    __shared__ std::int32_t indices[kArgmaxBlock];
    values[threadIdx.x]  = best_value;
    indices[threadIdx.x] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other_value        = values[threadIdx.x + stride];
            const std::int32_t other_index = indices[threadIdx.x + stride];
            if (argmax_better(other_value, other_index, values[threadIdx.x],
                              indices[threadIdx.x])) {
                values[threadIdx.x]  = other_value;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) { out[t] = indices[0]; }
}

__launch_bounds__(kArgmaxBlock) __global__
    void argmax_tiled_atomic_kernel(const __nv_bfloat16* logits, std::int32_t* out,
                                    std::int32_t valid_rows, std::int32_t physical_rows) {
    const std::int32_t t    = static_cast<std::int32_t>(blockIdx.y);
    const std::int64_t base = static_cast<std::int64_t>(t) * physical_rows;
    const std::int32_t tile_start =
        static_cast<std::int32_t>(blockIdx.x) * blockDim.x * kArgmaxItemsPerThread;

    float best_value        = -CUDART_INF_F;
    std::int32_t best_index = INT32_MAX;
#pragma unroll
    for (int item = 0; item < kArgmaxItemsPerThread; ++item) {
        const std::int32_t v = tile_start + threadIdx.x + item * blockDim.x;
        if (v < valid_rows) {
            const float value = __bfloat162float(logits[base + v]);
            if (argmax_better(value, v, best_value, best_index)) {
                best_value = value;
                best_index = v;
            }
        }
    }

    argmax_block_reduce(best_value, best_index);
    if (threadIdx.x != 0 || best_index == INT32_MAX) { return; }

    int current = out[t];
    while (true) {
        const float current_value = __bfloat162float(logits[base + current]);
        if (!argmax_better(best_value, best_index, current_value, current)) { break; }
        const int observed = atomicCAS(reinterpret_cast<int*>(out + t), current, best_index);
        if (observed == current) { break; }
        current = observed;
    }
}

__launch_bounds__(kShortlistRerankTile) __global__ void shortlist_tile_candidates_kernel(
    const __nv_bfloat16* logits, const std::int32_t* id_map, std::int32_t* candidate_ids,
    std::int32_t shortlist_rows, std::int32_t physical_rows, std::int32_t candidate_rows) {
    const int tile = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int row = tile * kShortlistRerankTile + static_cast<int>(threadIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(token) * physical_rows;
    const float original_value =
        row < shortlist_rows ? __bfloat162float(logits[base + row]) : -CUDART_INF_F;
    const std::int32_t original_row = row < shortlist_rows ? row : INT32_MAX;
    float best_value = original_value;
    std::int32_t best_row = original_row;
    argmax_block_reduce(best_value, best_row);
    __shared__ std::int32_t first_row;
    if (threadIdx.x == 0) { first_row = best_row; }
    __syncthreads();

    best_value = original_row == first_row ? -CUDART_INF_F : original_value;
    best_row = original_row == first_row ? INT32_MAX : original_row;
    argmax_block_reduce(best_value, best_row);
    const std::int64_t output_base = static_cast<std::int64_t>(token) * candidate_rows +
                                     tile * kShortlistCandidatesPerTile;
    if (threadIdx.x == 0) {
        candidate_ids[output_base] = id_map[first_row];
        candidate_ids[output_base + 1] = id_map[best_row];
    }
}

__launch_bounds__(256) __global__ void shortlist_exact_scores_kernel(
    const __nv_bfloat16* hidden, const __nv_bfloat16* exact_head,
    const std::int32_t* candidate_ids, float* candidate_scores, std::int32_t hidden_rows,
    std::int32_t candidate_rows) {
    const int candidate = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const std::int64_t candidate_index = static_cast<std::int64_t>(token) * candidate_rows +
                                         candidate;
    const int token_id = candidate_ids[candidate_index];
    const __nv_bfloat16* x = hidden + static_cast<std::int64_t>(token) * hidden_rows;
    const __nv_bfloat16* weight = exact_head + static_cast<std::int64_t>(token_id) * hidden_rows;
    float sum = 0.0F;
    for (int k = static_cast<int>(threadIdx.x); k < hidden_rows; k += blockDim.x) {
        sum = fmaf(__bfloat162float(x[k]), __bfloat162float(weight[k]), sum);
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { partial[threadIdx.x] += partial[threadIdx.x + stride]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { candidate_scores[candidate_index] = partial[0]; }
}

// Row-scaled FP8 exact head: each E4M3 code decodes exactly, the FP32 dot product accumulates
// against the BF16 hidden state, and the row's BF16 multiplier scales the complete sum once.
__launch_bounds__(256) __global__ void shortlist_exact_scores_fp8_kernel(
    const __nv_bfloat16* hidden, const std::uint8_t* exact_codes,
    const __nv_bfloat16* exact_row_scales, const std::int32_t* candidate_ids,
    float* candidate_scores, std::int32_t hidden_rows, std::int32_t candidate_rows) {
    const int candidate = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const std::int64_t candidate_index = static_cast<std::int64_t>(token) * candidate_rows +
                                         candidate;
    const int token_id = candidate_ids[candidate_index];
    const __nv_bfloat16* x = hidden + static_cast<std::int64_t>(token) * hidden_rows;
    const std::uint8_t* codes = exact_codes + static_cast<std::int64_t>(token_id) * hidden_rows;
    float sum = 0.0F;
    for (int k = static_cast<int>(threadIdx.x); k < hidden_rows; k += blockDim.x) {
        __nv_fp8_e4m3 code;
        code.__x = codes[k];
        sum = fmaf(__bfloat162float(x[k]), static_cast<float>(code), sum);
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { partial[threadIdx.x] += partial[threadIdx.x + stride]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        candidate_scores[candidate_index] =
            partial[0] * __bfloat162float(exact_row_scales[token_id]);
    }
}

__launch_bounds__(kShortlistRerankTile) __global__ void shortlist_exact_select_kernel(
    const float* candidate_scores, const std::int32_t* candidate_ids, std::int32_t* out,
    std::int32_t candidate_rows) {
    const int token = static_cast<int>(blockIdx.x);
    const int candidate = static_cast<int>(threadIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(token) * candidate_rows;
    float best_value = -CUDART_INF_F;
    std::int32_t best_index = INT32_MAX;
    for (int row = candidate; row < candidate_rows; row += blockDim.x) {
        const float value = candidate_scores[base + row];
        const std::int32_t index = candidate_ids[base + row];
        if (argmax_better(value, index, best_value, best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    argmax_block_reduce(best_value, best_index);
    if (threadIdx.x == 0) { out[token] = best_index; }
}

} // namespace ninfer::ops
