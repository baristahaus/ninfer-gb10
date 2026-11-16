// Implements: include/ninfer/ops/target_logprobs.h
//
// ONE KERNEL FOR BOTH FORMS. The scalar form reports the target token's log-probability; the
// ranking form reports the same value plus the top_k largest valid-row logits as (id,
// log-probability) pairs. They share one kernel with a compile-time ranking branch, because the
// ranking list is built from the same pass that finds the column maximum: a shift by the maximum
// does not change the order, so collecting candidates there costs no extra sweep.
//
// ONE OWNER. This file used to be one of two routes. The ranking form lived in
// ops/wrapper/target_logprobs.cpp with ops/launcher/target_logprobs.cu and
// ops/kernel/target_logprobs.cuh, and the scalar form lived here. The two had drifted: this route
// gained a SPARK-only 1024-thread fast-exp path that the other never had, and the other carried the
// ranking machinery this one lacked. Both are now here, and the legacy route is deleted.
//
// Algorithm assumptions: one independent CTA per column; no global workspace. The ranking lists
// live in shared memory, so reporting them is a compile-time choice and an unranked call allocates
// none. The ranking lists need the 256-thread block for their shared storage to fit, which is why
// the SPARK fast path (1024 threads) is scalar-only.

#include "ops/common/host/operands.h"
#include "ninfer/ops/target_logprobs.h"
#include <climits>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "core/device.h"
#include "core/platform.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/sampling_workspace.h"
#include "ops/kernel/sampling_order.cuh"
#include <cuda_bf16.h>
#include <math_constants.h>

namespace ninfer::ops {
using namespace common;

namespace {

inline constexpr int kTargetLogprobsBlock = 256;

// The reported ranking ceiling is the sampler's own candidate cap: a wider report would describe
// candidates the sampler could never have drawn from.
static_assert(kMaxReportedLogprobRanks == kSamplerCandidateCap);

// Merges two descending ranking lists of exactly kMaxReportedLogprobRanks entries into the first,
// under the ordering rule of ops::sample (higher value wins, lower id breaks exact ties). The
// caller holds both lists in registers, so each shared row is read once and written once.
__device__ __forceinline__ void
target_logprob_merge_lists(float (&left)[kMaxReportedLogprobRanks],
                           int (&left_id)[kMaxReportedLogprobRanks],
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

template <int BlockSize, bool FastExp, bool ReportRanking>
__launch_bounds__(BlockSize) __global__
    void target_logprobs_kernel(const __nv_bfloat16* logits, const std::int32_t* target_ids,
                                float* output, std::int32_t valid_rows, std::int32_t physical_rows,
                                std::int32_t top_k, std::int32_t* top_ids, float* top_logprobs) {
    constexpr int kRanks = kMaxReportedLogprobRanks;
    if constexpr (ReportRanking) {
        static_assert(BlockSize == kTargetLogprobsBlock,
                      "the ranking lists need the 256-thread block so their shared storage fits");
    }
    __shared__ float list_value[ReportRanking ? BlockSize : 1][ReportRanking ? kRanks : 1];
    __shared__ int list_id[ReportRanking ? BlockSize : 1][ReportRanking ? kRanks : 1];

    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base   = static_cast<std::int64_t>(column) * physical_rows;
    const int tid             = static_cast<int>(threadIdx.x);

    if constexpr (ReportRanking) {
#pragma unroll
        for (int rank = 0; rank < kRanks; ++rank) {
            list_value[tid][rank] = -CUDART_INF_F;
            list_id[tid][rank]    = INT_MAX;
        }
    }

    float local_max = -CUDART_INF_F;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        const float value = __bfloat162float(logits[base + row]);
        local_max         = fmaxf(local_max, value);
        if constexpr (ReportRanking) {
            // Shifting by the column maximum does not change the order, so the same pass that finds
            // the maximum can collect the leading candidates without a second sweep.
            sampling_insert_candidate(list_value[tid], list_id[tid], static_cast<int>(top_k), value,
                                      static_cast<int>(row));
        }
    }
    const float maximum = block_max_broadcast<BlockSize>(local_max);

    float local_sum = 0.0f;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        const float value = __bfloat162float(logits[base + row]) - maximum;
        local_sum += FastExp ? __expf(value) : expf(value);
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
    const float normalizer = maximum + logf(sum);
    output[column]         = __bfloat162float(logits[base + target_ids[column]]) - normalizer;

    if constexpr (!ReportRanking) { return; }
    const std::int32_t reported   = top_k < valid_rows ? top_k : valid_rows;
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

template <int Block, bool FastExp, bool ReportRanking>
void launch_columns(const Tensor& logits, const Tensor& target_ids, std::int32_t valid_rows,
                    Tensor& output, std::int32_t top_k, Tensor* top_ids, Tensor* top_logprobs,
                    cudaStream_t stream) {
    const auto columns = static_cast<unsigned int>(logits.ne[1]);
    target_logprobs_kernel<Block, FastExp, ReportRanking><<<columns, Block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<const std::int32_t*>(target_ids.data), static_cast<float*>(output.data),
        valid_rows, logits.ne[0], top_k,
        top_ids != nullptr ? static_cast<std::int32_t*>(top_ids->data) : nullptr,
        top_logprobs != nullptr ? static_cast<float*>(top_logprobs->data) : nullptr);
    CUDA_CHECK(cudaGetLastError());
}

void target_logprobs_launch(const Tensor& logits, const Tensor& target_ids, std::int32_t valid_rows,
                            Tensor& output, Tensor* top_ids, Tensor* top_logprobs,
                            cudaStream_t stream) {
    if (top_ids != nullptr) {
        const auto top_k = static_cast<std::int32_t>(top_ids->ne[0]);
        launch_columns<kTargetLogprobsBlock, false, true>(logits, target_ids, valid_rows, output,
                                                          top_k, top_ids, top_logprobs, stream);
        return;
    }
#if NINFER_TARGET_SPARK
    if (logits.ne[0] == 248320 && valid_rows == 248077) {
        launch_columns<1024, true, false>(logits, target_ids, valid_rows, output, 0, nullptr,
                                          nullptr, stream);
        return;
    }
#endif
    launch_columns<kTargetLogprobsBlock, false, false>(logits, target_ids, valid_rows, output, 0,
                                                       nullptr, nullptr, stream);
}

// Validation for the ranking overload. The scalar overload validates through the shared operand
// helpers below; this one states the same contract in the master-generation shape terms the serving
// adapters use, and adds the ranking tensors.
void require_rank_two(const Tensor& tensor, const char* label) {
    if (tensor.ne[0] <= 0 || tensor.ne[1] <= 0 || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("target_logprobs: ") + label +
                                    " must be rank-2 with positive dimensions");
    }
}

void require_vector(const Tensor& tensor, std::int32_t columns, const char* label) {
    if (tensor.ne[0] != columns || tensor.ne[1] != 1 || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("target_logprobs: ") + label +
                                    " must have shape [columns]");
    }
}

void require_accessible(const Tensor& tensor, std::size_t alignment, const char* label) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("target_logprobs: ") + label +
                                    " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("target_logprobs: ") + label +
                                    " data must be non-null");
    }
    if ((reinterpret_cast<std::uintptr_t>(tensor.data) & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string("target_logprobs: ") + label +
                                    " data is not naturally aligned");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

// Validates an optional ranking request: either both outputs are supplied with a common rank count
// or neither is, so a caller cannot receive ids without probabilities or the reverse.
std::int32_t validate_ranking(const Tensor& logits, Tensor* top_ids, Tensor* top_logprobs) {
    if ((top_ids == nullptr) != (top_logprobs == nullptr)) {
        throw std::invalid_argument("target_logprobs: top_ids and top_logprobs are requested "
                                    "together or not at all");
    }
    if (top_ids == nullptr) { return 0; }
    if (top_ids->dtype != DType::I32) {
        throw std::invalid_argument("target_logprobs: top_ids must be I32");
    }
    if (top_logprobs->dtype != DType::FP32) {
        throw std::invalid_argument("target_logprobs: top_logprobs must be FP32");
    }
    require_rank_two(*top_ids, "top_ids");
    if (top_ids->ne[0] > kMaxReportedLogprobRanks) {
        throw std::invalid_argument("target_logprobs: top_ids requests more than " +
                                    std::to_string(kMaxReportedLogprobRanks) + " ranks");
    }
    if (top_ids->ne[1] != logits.ne[1] || top_ids->ne[2] != 1 || top_ids->ne[3] != 1) {
        throw std::invalid_argument("target_logprobs: top_ids must have shape [ranks,columns]");
    }
    if (top_logprobs->ne[0] != top_ids->ne[0] || top_logprobs->ne[1] != top_ids->ne[1] ||
        top_logprobs->ne[2] != 1 || top_logprobs->ne[3] != 1) {
        throw std::invalid_argument("target_logprobs: top_logprobs must match the top_ids shape");
    }
    require_accessible(*top_ids, alignof(std::int32_t), "top_ids");
    require_accessible(*top_logprobs, alignof(float), "top_logprobs");
    return top_ids->ne[0];
}

} // namespace

void target_logprobs(const Tensor& logits, const Tensor& target_ids, std::int32_t valid_rows,
                     Tensor& output, CudaExecutionView execution) {
    const auto stream        = execution.stream();
    constexpr const char* op = "target_logprobs";
    if (logits.ne[0] <= 0 || logits.ne[1] <= 0 || valid_rows <= 0 || valid_rows > logits.ne[0])
        throw std::invalid_argument("target_logprobs: invalid column or row domain");
    const auto columns = logits.ne[1];
    common::operand::tensor(logits, DType::BF16, {logits.ne[0], columns}, alignof(std::uint16_t),
                            op, "logits", execution.device());
    common::operand::tensor(target_ids, DType::I32, {columns}, alignof(std::int32_t), op,
                            "target_ids", execution.device());
    common::operand::tensor(output, DType::FP32, {columns}, alignof(float), op, "output",
                            execution.device());
    common::operand::disjoint(common::operand::range(output, op),
                              common::operand::range(logits, op), op);
    common::operand::disjoint(common::operand::range(output, op),
                              common::operand::range(target_ids, op), op);

    target_logprobs_launch(logits, target_ids, valid_rows, output, nullptr, nullptr, stream);
}

void target_logprobs(const Tensor& logits, const Tensor& target_ids, std::int32_t valid_rows,
                     Tensor& output, Tensor* top_ids, Tensor* top_logprobs, cudaStream_t stream) {
    if (logits.dtype != DType::BF16) {
        throw std::invalid_argument("target_logprobs: logits must be BF16");
    }
    if (target_ids.dtype != DType::I32) {
        throw std::invalid_argument("target_logprobs: target_ids must be I32");
    }
    if (output.dtype != DType::FP32) {
        throw std::invalid_argument("target_logprobs: output must be FP32");
    }

    require_rank_two(logits, "logits");
    const std::int32_t columns = logits.ne[1];
    require_vector(target_ids, columns, "target_ids");
    require_vector(output, columns, "output");
    if (valid_rows <= 0 || valid_rows > logits.ne[0]) {
        throw std::invalid_argument("target_logprobs: valid_rows must be in [1, physical_rows]");
    }

    require_accessible(logits, alignof(std::uint16_t), "logits");
    require_accessible(target_ids, alignof(std::int32_t), "target_ids");
    require_accessible(output, alignof(float), "output");
    if (overlaps(output, logits) || overlaps(output, target_ids)) {
        throw std::invalid_argument("target_logprobs: output must not overlap either input");
    }

    const std::int32_t ranks = validate_ranking(logits, top_ids, top_logprobs);
    if (ranks > 0) {
        if (overlaps(*top_ids, logits) || overlaps(*top_ids, target_ids) ||
            overlaps(*top_logprobs, logits) || overlaps(*top_logprobs, target_ids) ||
            overlaps(*top_ids, output) || overlaps(*top_logprobs, output) ||
            overlaps(*top_ids, *top_logprobs)) {
            throw std::invalid_argument(
                "target_logprobs: ranking outputs must not overlap any other tensor");
        }
    }

    target_logprobs_launch(logits, target_ids, valid_rows, output, top_ids, top_logprobs, stream);
}

} // namespace ninfer::ops
