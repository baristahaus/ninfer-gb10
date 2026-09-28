#pragma once

// Implements: include/ninfer/ops/target_logprobs.h
// Match: contiguous BF16 [physical_rows,C], I32 [C], and FP32 [C] plus optional [top_k,C] outputs.
// Algorithm assumptions: one 256-thread CTA per published column; a two-pass scaled logsumexp and,
// when ranking is compiled in, a per-thread bounded ranking merged by a shared pairwise merge tree.

#include "ninfer/ops/target_logprobs.h"

#include "ops/common/sampling_workspace.h"
#include "ops/common/warp.cuh"
#include "ops/kernel/sampling_order.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <climits>
#include <cstdint>

namespace ninfer::ops {

inline constexpr int kTargetLogprobsBlock = 256;

// The reported ranking ceiling is the sampler's own candidate cap: a wider report would describe
// candidates the sampler could never have drawn from.
static_assert(kMaxReportedLogprobRanks == kSamplerCandidateCap);

template <int BlockSize>
__device__ __forceinline__ float target_logprobs_block_max(float value) {
    static_assert(BlockSize >= kWarpSize && BlockSize <= 1024);
    static_assert((BlockSize & (BlockSize - 1)) == 0);
    constexpr int kWarps = BlockSize / kWarpSize;
    __shared__ float warp_maxima[kWarps];
    __shared__ float result;

    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    value          = warp_max(value);
    if (lane == 0) { warp_maxima[warp] = value; }
    __syncthreads();

    if (warp == 0) {
        value = lane < kWarps ? warp_maxima[lane] : -CUDART_INF_F;
        value = warp_max(value);
        if (lane == 0) { result = value; }
    }
    __syncthreads();
    return result;
}

// One column's penalty-adjusted, temperature-scaled logit, evaluated against the committed token
// counts as they stood when this column was drawn. The penalty terms mirror sampling_adjusted_logit
// exactly; the only difference is that the draw and every later token of the same round are removed
// first, because the sampler and the accept step have already added them. Ordering by this value and
// by the reported log-probability are the same relation, the scale being positive.
__device__ __forceinline__ float target_logprob_scaled_value(
    const __nv_bfloat16* logits, std::int64_t base, std::int32_t row, const SamplingConfig& config,
    float inverse_temperature, const std::int32_t* round_tokens, std::int32_t subtract_begin,
    std::int32_t subtract_end) {
    float value = __bfloat162float(logits[base + row]);
    if (config.presence_penalty != 0.0f || config.frequency_penalty != 0.0f) {
        int count = config.token_counts != nullptr ? config.token_counts[row] : 0;
        for (std::int32_t index = subtract_begin; index < subtract_end; ++index) {
            if (round_tokens[index] == row) { --count; }
        }
        if (count > 0) { value -= config.presence_penalty; }
        if (config.frequency_penalty != 0.0f) {
            value -= config.frequency_penalty * static_cast<float>(count);
        }
    }
    return value * inverse_temperature;
}

// Merges two descending ranking lists of exactly kMaxReportedLogprobRanks entries into the first,
// under the ordering rule of ops::sample (higher value wins, lower id breaks exact ties). The caller
// holds both lists in registers, so each shared row is read once and written once.
__device__ __forceinline__ void target_logprob_merge_lists(
    float (&left)[kMaxReportedLogprobRanks], int (&left_id)[kMaxReportedLogprobRanks],
    const float (&right)[kMaxReportedLogprobRanks],
    const int (&right_id)[kMaxReportedLogprobRanks]) {
    float merged[kMaxReportedLogprobRanks];
    int merged_id[kMaxReportedLogprobRanks];
    int head_l = 0;
    int head_r = 0;
    for (int rank = 0; rank < kMaxReportedLogprobRanks; ++rank) {
        const bool exhausted_r = head_r >= kMaxReportedLogprobRanks;
        const bool take_left   = head_l < kMaxReportedLogprobRanks &&
                               (exhausted_r || sampling_better(left[head_l], left_id[head_l],
                                                               right[head_r], right_id[head_r]));
        if (take_left) {
            merged[rank]    = left[head_l];
            merged_id[rank] = left_id[head_l];
            ++head_l;
        } else {
            merged[rank]    = right[head_r];
            merged_id[rank] = right_id[head_r];
            ++head_r;
        }
    }
#pragma unroll
    for (int rank = 0; rank < kMaxReportedLogprobRanks; ++rank) {
        left[rank]    = merged[rank];
        left_id[rank] = merged_id[rank];
    }
}

template <int BlockSize, bool ReportRanking>
__launch_bounds__(BlockSize) __global__ void target_logprobs_kernel(
    const __nv_bfloat16* logits, const std::int32_t* target_ids, float* output,
    std::int32_t valid_rows, std::int32_t physical_rows, const SamplingConfig* configs,
    std::int32_t columns_per_lane, const std::int32_t* round_tokens,
    const std::int32_t* round_produced, std::int32_t top_k, std::int32_t* top_ids,
    float* top_logprobs) {

    constexpr int kRanks = kMaxReportedLogprobRanks;
    __shared__ float list_value[ReportRanking ? BlockSize : 1][ReportRanking ? kRanks : 1];
    __shared__ int list_id[ReportRanking ? BlockSize : 1][ReportRanking ? kRanks : 1];

    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base   = static_cast<std::int64_t>(column) * physical_rows;
    const int tid             = static_cast<int>(threadIdx.x);

    SamplingConfig config;
    if (configs != nullptr) { config = configs[column / columns_per_lane]; }
    const float inverse_temperature = config.temperature > 0.0f ? 1.0f / config.temperature : 1.0f;

    const std::int32_t lane   = column / columns_per_lane;
    const std::int32_t within = column - lane * columns_per_lane;
    const std::int32_t* lane_tokens =
        round_tokens != nullptr ? round_tokens + static_cast<std::int64_t>(lane) * columns_per_lane
                                : nullptr;
    const std::int32_t produced =
        round_tokens == nullptr
            ? within
            : (round_produced != nullptr ? round_produced[lane] : columns_per_lane);
    const std::int32_t subtract_begin = within < produced ? within : produced;
    const std::int32_t subtract_end   = produced;

    if constexpr (ReportRanking) {
#pragma unroll
        for (int rank = 0; rank < kRanks; ++rank) {
            list_value[tid][rank] = -CUDART_INF_F;
            list_id[tid][rank]    = INT_MAX;
        }
    }

    float local_max = -CUDART_INF_F;
    for (std::int32_t row = tid; row < valid_rows; row += BlockSize) {
        const float value =
            target_logprob_scaled_value(logits, base, row, config, inverse_temperature, lane_tokens,
                                        subtract_begin, subtract_end);
        local_max = fmaxf(local_max, value);
        if constexpr (ReportRanking) {
            sampling_insert_candidate(list_value[tid], list_id[tid], static_cast<int>(top_k), value,
                                      static_cast<int>(row));
        }
    }
    const float maximum = target_logprobs_block_max<BlockSize>(local_max);

    float local_sum = 0.0f;
    for (std::int32_t row = tid; row < valid_rows; row += BlockSize) {
        local_sum += expf(target_logprob_scaled_value(logits, base, row, config, inverse_temperature,
                                                      lane_tokens, subtract_begin, subtract_end) -
                          maximum);
    }
    __shared__ float warp_sums[BlockSize / kWarpSize];
    const float sum = block_reduce_sum<BlockSize>(local_sum, warp_sums);

    if constexpr (ReportRanking) {
        // The reductions above synchronized, so every thread's list is visible. Merge the BlockSize
        // sorted lists pairwise in shared memory: round stride merges lists stride apart, and every
        // merged list lands at a multiple of 2*stride for the next round to consume.
        for (int stride = 1; stride < BlockSize; stride <<= 1) {
            __syncthreads();
            if (tid < BlockSize / (2 * stride)) {
                const int keep = tid * 2 * stride;
                const int drop = keep + stride;
                float right_value[kRanks];
                int right_id[kRanks];
#pragma unroll
                for (int rank = 0; rank < kRanks; ++rank) {
                    right_value[rank] = list_value[drop][rank];
                    right_id[rank]    = list_id[drop][rank];
                }
                target_logprob_merge_lists(list_value[keep], list_id[keep], right_value, right_id);
            }
        }
    }

    if (tid != 0) { return; }
    output[column] =
        target_logprob_scaled_value(logits, base, target_ids[column], config, inverse_temperature,
                                    lane_tokens, subtract_begin, subtract_end) -
        maximum - logf(sum);

    if constexpr (!ReportRanking) { return; }
    const std::int32_t reported   = top_k < valid_rows ? top_k : valid_rows;
    const float normalizer        = maximum + logf(sum);
    const std::int64_t column_off = static_cast<std::int64_t>(column) * top_k;
    for (std::int32_t rank = 0; rank < top_k; ++rank) {
        const std::int64_t destination = static_cast<std::int64_t>(rank) + column_off;
        if (rank < reported && list_id[0][rank] != INT_MAX) {
            top_ids[destination]      = list_id[0][rank];
            top_logprobs[destination] = list_value[0][rank] - normalizer;
        } else {
            top_ids[destination]      = kNoReportedLogprobRank;
            top_logprobs[destination] = -CUDART_INF_F;
        }
    }
}

} // namespace ninfer::ops
