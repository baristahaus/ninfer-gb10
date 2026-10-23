#pragma once

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

inline constexpr int kSparseMoeExperts = 256;
inline constexpr int kSparseMoeTopK    = 8;

struct SparseMoeRankedValue {
    float value;
    int id;
};

__device__ __forceinline__ bool sparse_moe_ranked_better(const SparseMoeRankedValue& a,
                                                         const SparseMoeRankedValue& b) {
    return a.value > b.value || (a.value == b.value && a.id < b.id);
}

// Merges the descending run of each lane with the run of its xor partner and keeps the better
// half. The eight exchanges of a step do not depend on each other, so the warp reaches the
// warp-wide top-8 in five merge steps instead of eight dependent reduction rounds. Ranking is a
// total order over distinct expert ids, so the selected set and its order are the same as the
// ones any other correct selection produces.
__device__ __forceinline__ void
sparse_moe_merge_ranked_runs(SparseMoeRankedValue (&run)[kSparseMoeTopK]) {
#pragma unroll
    for (int partner = 1; partner < 32; partner <<= 1) {
        SparseMoeRankedValue merged[kSparseMoeTopK];
#pragma unroll
        for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
            const SparseMoeRankedValue mirror = run[kSparseMoeTopK - 1 - rank];
            SparseMoeRankedValue other;
            other.value  = __shfl_xor_sync(kFullWarpMask, mirror.value, partner);
            other.id     = __shfl_xor_sync(kFullWarpMask, mirror.id, partner);
            merged[rank] = sparse_moe_ranked_better(run[rank], other) ? run[rank] : other;
        }
        // The kept half is bitonic; three compare-exchange stages restore descending order.
#pragma unroll
        for (int stride = kSparseMoeTopK / 2; stride > 0; stride >>= 1) {
#pragma unroll
            for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
                if ((rank & stride) != 0) { continue; }
                const int partner_rank = rank | stride;
                if (!sparse_moe_ranked_better(merged[rank], merged[partner_rank])) {
                    const SparseMoeRankedValue swap = merged[rank];
                    merged[rank]                    = merged[partner_rank];
                    merged[partner_rank]            = swap;
                }
            }
        }
#pragma unroll
        for (int rank = 0; rank < kSparseMoeTopK; ++rank) { run[rank] = merged[rank]; }
    }
}

// SP-4 phase 2b: the per-expert residency entry. Bit 31 marks the warm tier; the low 31
// bits are the expert's compacted rank in its bank. A null table selects the untiered
// direct path: the expert reads the device bank at its own rank.
__device__ __forceinline__ void sparse_moe_residency(const unsigned* __restrict__ table,
                                                     int expert, bool& warm, int& rank) {
    if (table == nullptr) {
        warm = false;
        rank = expert;
        return;
    }
    const unsigned entry = table[expert];
    warm = (entry & 0x80000000u) != 0u;
    rank = static_cast<int>(entry & 0x7fffffffu);
}

__device__ __forceinline__ void sparse_moe_select_top8_warp(const float* scores, int* ids,
                                                            float* alpha, float* shared_scale,
                                                            float* selected_logits,
                                                            std::uint32_t* expert_activations) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    SparseMoeRankedValue local[kSparseMoeTopK];
#pragma unroll
    for (int item = 0; item < kSparseMoeTopK; ++item) {
        const int id = lane + item * 32;
        local[item]  = {scores[id], id};
    }
#pragma unroll
    for (int i = 1; i < kSparseMoeTopK; ++i) {
        const SparseMoeRankedValue value = local[i];
        int position                     = i;
        while (position > 0 && sparse_moe_ranked_better(value, local[position - 1])) {
            local[position] = local[position - 1];
            --position;
        }
        local[position] = value;
    }
    sparse_moe_merge_ranked_runs(local);

    if (lane == 0) {
#pragma unroll
        for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
            ids[rank]             = local[rank].id;
            selected_logits[rank] = local[rank].value;
        }
    }
    __syncwarp();

    // SP-4 phase 2b: accumulate one count per selected expert when profiling is armed.
    if (expert_activations != nullptr && lane < kSparseMoeTopK) {
        atomicAdd(expert_activations + ids[lane], 1u);
    }

    float exponential = 0.0f;
    if (lane < kSparseMoeTopK) { exponential = expf(selected_logits[lane] - selected_logits[0]); }
    float denominator = warp_reduce_sum(exponential);
    denominator       = __shfl_sync(kFullWarpMask, denominator, 0);
    if (lane < kSparseMoeTopK) { alpha[lane] = exponential / denominator; }
    if (lane == 0) { *shared_scale = sigmoid(scores[kSparseMoeExperts]); }
}

// The general-N top-k selection: identical epilogue, generalized to any (Experts, TopK) pair
// admitted by the sparse-MoE bank shapes. Per-lane items = Experts/32 feed the same register
// insertion sort; the run is truncated to TopK and merged in five xor-partner rounds, each
// keeping the better TopK of the two sorted runs through a full register-local insertion sort.
// Ranking is a total order over distinct expert ids, so the selected set and its order are the
// same as the ones any other correct selection produces.
template <int Experts, int TopK>
__device__ __forceinline__ void sparse_moe_select_topk_warp(const float* scores, int* ids,
                                                            float* alpha, float* shared_scale,
                                                            float* selected_logits,
                                                            std::uint32_t* expert_activations) {
    constexpr int kPerLane = Experts / 32;
    static_assert(kPerLane >= TopK, "top-k selection needs Experts/32 >= TopK");
    const int lane = static_cast<int>(threadIdx.x) & 31;
    SparseMoeRankedValue local[kPerLane];
#pragma unroll
    for (int item = 0; item < kPerLane; ++item) {
        const int id = lane + item * 32;
        local[item]  = {scores[id], id};
    }
#pragma unroll
    for (int i = 1; i < kPerLane; ++i) {
        const SparseMoeRankedValue value = local[i];
        int position                     = i;
        while (position > 0 && sparse_moe_ranked_better(value, local[position - 1])) {
            local[position] = local[position - 1];
            --position;
        }
        local[position] = value;
    }
    SparseMoeRankedValue run[TopK];
#pragma unroll
    for (int rank = 0; rank < TopK; ++rank) { run[rank] = local[rank]; }
#pragma unroll
    for (int partner = 1; partner < 32; partner <<= 1) {
        SparseMoeRankedValue other[TopK];
#pragma unroll
        for (int rank = 0; rank < TopK; ++rank) {
            other[rank].value = __shfl_xor_sync(kFullWarpMask, run[rank].value, partner);
            other[rank].id    = __shfl_xor_sync(kFullWarpMask, run[rank].id, partner);
        }
        SparseMoeRankedValue merged[2 * TopK];
#pragma unroll
        for (int rank = 0; rank < TopK; ++rank) {
            merged[rank]         = run[rank];
            merged[TopK + rank]  = other[rank];
        }
        // Full register-local insertion sort over the two runs' union; the better TopK stay.
#pragma unroll
        for (int i = 1; i < 2 * TopK; ++i) {
            const SparseMoeRankedValue value = merged[i];
            int position                     = i;
            while (position > 0 && sparse_moe_ranked_better(value, merged[position - 1])) {
                merged[position] = merged[position - 1];
                --position;
            }
            merged[position] = value;
        }
#pragma unroll
        for (int rank = 0; rank < TopK; ++rank) { run[rank] = merged[rank]; }
    }

    if (lane == 0) {
#pragma unroll
        for (int rank = 0; rank < TopK; ++rank) {
            ids[rank]             = run[rank].id;
            selected_logits[rank] = run[rank].value;
        }
    }
    __syncwarp();

    if (expert_activations != nullptr && lane < TopK) {
        atomicAdd(expert_activations + ids[lane], 1u);
    }

    float exponential = 0.0f;
    if (lane < TopK) { exponential = expf(selected_logits[lane] - selected_logits[0]); }
    float denominator = warp_reduce_sum(exponential);
    denominator       = __shfl_sync(kFullWarpMask, denominator, 0);
    if (lane < TopK) { alpha[lane] = exponential / denominator; }
    if (lane == 0) { *shared_scale = sigmoid(scores[Experts]); }
}

} // namespace ninfer::ops::detail
