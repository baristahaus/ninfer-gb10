#include "ninfer/ops/column_topk.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

#include "core/device.h"

namespace ninfer::ops {
namespace {
constexpr int kThreads = 256;
constexpr int kMaxRank = 16;

__device__ __forceinline__ std::uint32_t monotonic_key(float x) {
    const std::uint32_t b = __float_as_uint(x);
    return (b & 0x80000000u) ? ~b : (b ^ 0x80000000u);
}

__device__ __forceinline__ float from_monotonic_key(std::uint32_t u) {
    const std::uint32_t b = (u & 0x80000000u) ? (u ^ 0x80000000u) : ~u;
    return __uint_as_float(b);
}

// Larger keys win; equal scores prefer the lower row index.
__device__ __forceinline__ std::uint64_t packed_key(float x, std::int32_t row) {
    return (static_cast<std::uint64_t>(monotonic_key(x)) << 32) |
           static_cast<std::uint32_t>(0xFFFFFFFFu - static_cast<std::uint32_t>(row));
}

template <int Rank>
__device__ __forceinline__ void insert_sorted(std::uint64_t (&values)[Rank],
                                              std::uint64_t key, int& count) {
    if (count == Rank && key <= values[Rank - 1]) { return; }
    int i = count < Rank ? count : Rank - 1;
    while (i > 0 && values[i - 1] < key) {
        values[i] = values[i - 1];
        --i;
    }
    values[i] = key;
    if (count < Rank) { ++count; }
}

template <int Rank>
__global__ void column_topk_kernel(const __nv_bfloat16* logits, std::int32_t rows,
                                   std::int32_t columns, std::int32_t* ids, float* scores) {
    const std::int32_t column = blockIdx.x;
    __shared__ std::uint64_t per_thread[kThreads * Rank];
    __shared__ std::uint64_t per_lane[32 * Rank];

    std::uint64_t local[Rank];
    int count = 0;
    for (std::int32_t row = threadIdx.x; row < rows; row += kThreads) {
        // The vocabulary producers store logits token-major: one column's V values are contiguous.
        const float value = __bfloat162float(logits[static_cast<std::int64_t>(column) * rows + row]);
        insert_sorted(local, packed_key(value, row), count);
    }
    for (int j = 0; j < Rank; ++j) {
        per_thread[threadIdx.x * Rank + j] = j < count ? local[j] : 0;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        std::uint64_t lane_values[Rank];
        int lane_count = 0;
        for (int entry = threadIdx.x; entry < kThreads * Rank; entry += 32) {
            const std::uint64_t key = per_thread[entry];
            if (key != 0) { insert_sorted(lane_values, key, lane_count); }
        }
        for (int j = 0; j < Rank; ++j) {
            per_lane[threadIdx.x * Rank + j] = j < lane_count ? lane_values[j] : 0;
        }
        __syncwarp();
        if (threadIdx.x == 0) {
            std::uint64_t final_values[Rank];
            int final_count = 0;
            for (int entry = 0; entry < 32 * Rank; ++entry) {
                const std::uint64_t key = per_lane[entry];
                if (key != 0) { insert_sorted(final_values, key, final_count); }
            }
            for (int rank = 0; rank < Rank; ++rank) {
                const std::int64_t index = static_cast<std::int64_t>(rank) * columns + column;
                if (rank < final_count) {
                    const std::uint64_t key = final_values[rank];
                    ids[index] = static_cast<std::int32_t>(
                        0xFFFFFFFFu - static_cast<std::uint32_t>(key & 0xFFFFFFFFu));
                    scores[index] = from_monotonic_key(static_cast<std::uint32_t>(key >> 32));
                } else {
                    ids[index]    = 0;
                    scores[index] = 0.0f;
                }
            }
        }
    }
}
} // namespace

void column_topk(const Tensor& logits, Tensor& ids, Tensor& scores, CudaExecutionView execution) {
    if (logits.device != execution.device() || ids.device != execution.device() ||
        scores.device != execution.device()) {
        throw std::invalid_argument("column_topk: operands must belong to execution");
    }
    if (logits.dtype != DType::BF16 || ids.dtype != DType::I32 || scores.dtype != DType::FP32) {
        throw std::invalid_argument("column_topk: expected BF16 logits, I32 ids, FP32 scores");
    }
    if (!logits.is_contiguous() || !ids.is_contiguous() || !scores.is_contiguous()) {
        throw std::invalid_argument("column_topk: operands must be contiguous");
    }
    const std::int32_t rows    = logits.ne[0];
    const std::int32_t columns = logits.ne[1] * logits.ne[2];
    const std::int32_t rank    = ids.ne[0];
    if (rows < 1 || columns < 1 || rank < 1 || rank > kMaxRank ||
        ids.ne[1] != logits.ne[1] || ids.ne[2] != logits.ne[2] ||
        scores.ne[0] != rank || scores.ne[1] != logits.ne[1] ||
        scores.ne[2] != logits.ne[2]) {
        throw std::invalid_argument("column_topk: invalid shape");
    }
    if (reinterpret_cast<std::uintptr_t>(logits.data) % 16 != 0 ||
        reinterpret_cast<std::uintptr_t>(ids.data) % 16 != 0 ||
        reinterpret_cast<std::uintptr_t>(scores.data) % 16 != 0) {
        throw std::invalid_argument("column_topk: operands require 16-byte alignment");
    }

    const auto* logits_ptr = static_cast<const __nv_bfloat16*>(logits.data);
    auto* ids_ptr          = static_cast<std::int32_t*>(ids.data);
    auto* scores_ptr       = static_cast<float*>(scores.data);
    const dim3 grid(columns);
    switch (rank) {
#define NINFER_COLUMN_TOPK_CASE(R)                                                             \
    case R:                                                                                    \
        column_topk_kernel<R><<<grid, kThreads, 0, execution.stream()>>>(logits_ptr, rows,     \
                                                                        columns, ids_ptr,      \
                                                                        scores_ptr);           \
        break;
        NINFER_COLUMN_TOPK_CASE(1)
        NINFER_COLUMN_TOPK_CASE(2)
        NINFER_COLUMN_TOPK_CASE(4)
        NINFER_COLUMN_TOPK_CASE(8)
        NINFER_COLUMN_TOPK_CASE(16)
#undef NINFER_COLUMN_TOPK_CASE
    default:
        throw std::invalid_argument("column_topk: rank must be one of 1, 2, 4, 8, 16");
    }
    if (cudaGetLastError() != cudaSuccess) {
        throw std::runtime_error("column_topk: kernel launch failed");
    }
}

} // namespace ninfer::ops
