// Dense MXFP8 linear dispatch and kernel launches (the bounded A16 fallback).

#include "ops/linear/mxfp8/mxfp8_dispatch.h"

#include "core/device.h"
#include "ops/common/token_slices.h"
#include "ops/linear/mxfp8/mxfp8_a16_kernels.cuh"

#include <cuda_bf16.h>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Mxfp8GemvSchedule =
    Mxfp8A16GemvSchedule<8, 4, 16, 4, 2, 1>;
using Mxfp8MmaSchedule =
    Mxfp8A16MmaSchedule<64, 64, 64, 16, 16, 2, 1, Cache::ca, Cache::ca>;

void validate_mxfp8_weight(const Weight& weight, const char* label) {
    if (weight.qtype != QType::MXFP8 || weight.layout != QuantLayout::DenseBlockScaleK32Mxfp8) {
        throw std::invalid_argument(std::string(label) +
                                    ": weight is not a dense block-scale MXFP8 matrix");
    }
    if (weight.group != 32 || weight.scale_dtype != DType::U8) {
        throw std::invalid_argument(std::string(label) + ": MXFP8 group size must be 32 (UE8M0)");
    }
    if (weight.scales == nullptr || weight.qdata == nullptr) {
        throw std::invalid_argument(std::string(label) + ": MXFP8 planes must be resident");
    }
    if (weight.k % 32 != 0) {
        throw std::invalid_argument(std::string(label) + ": MXFP8 K must be a multiple of 32");
    }
}

} // namespace

std::size_t mxfp8_linear_workspace_capacity_bytes(std::int32_t, std::int32_t, LinearPolicy policy,
                                                  std::int32_t min_tokens, std::int32_t max_tokens) {
    if (!valid_linear_policy(policy)) {
        throw std::invalid_argument("mxfp8 linear: unsupported policy");
    }
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("mxfp8 linear workspace: invalid token interval");
    }
    return 0;
}

void mxfp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    MemoryArena* workspace, cudaStream_t stream) {
    (void)policy;
    (void)workspace;
    validate_mxfp8_weight(weight, "mxfp8 linear");
    const int tokens = x.ne[1];
    if (tokens <= 0) {
        throw std::invalid_argument("mxfp8 linear: T must be positive");
    }
    if (weight.k % 16 != 0) {
        throw std::invalid_argument("mxfp8 linear: K must be a multiple of 16");
    }
    auto* out_data = static_cast<__nv_bfloat16*>(out.data);
    const auto* x_data = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(weight.scales);

    if (tokens == 1 && weight.k % (32 * Mxfp8GemvSchedule::kValuesPerLane) == 0) {
        const int blocks = (weight.n + Mxfp8GemvSchedule::kBlockRows - 1) /
                           Mxfp8GemvSchedule::kBlockRows;
        mxfp8_a16_gemv_kernel<Mxfp8GemvSchedule>
            <<<blocks, Mxfp8GemvSchedule::kThreads, 0, stream>>>(x_data, codes, scales, out_data,
                                                                 weight.n, weight.k);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (weight.k % Mxfp8MmaSchedule::kBlockK != 0) {
        throw std::invalid_argument(
            "mxfp8 linear: K must be a whole number of warp phases (gemv) or K tiles (mma)");
    }
    for_each_token_slice(tokens, Mxfp8MmaSchedule::kBlockTokens,
                         [&](std::int32_t offset, std::int32_t count) {
                             const dim3 grid((weight.n + Mxfp8MmaSchedule::kBlockRows - 1) /
                                                 Mxfp8MmaSchedule::kBlockRows,
                                             (count + Mxfp8MmaSchedule::kBlockTokens - 1) /
                                                 Mxfp8MmaSchedule::kBlockTokens);
                             const auto launch = [&]<bool Full>() {
                                 constexpr auto kernel =
                                     mxfp8_a16_mma_kernel<Mxfp8MmaSchedule, Full>;
                                 const int bytes = Mxfp8MmaSchedule::kSharedBytes;
                                 kernel<<<grid, Mxfp8MmaSchedule::kThreads, bytes, stream>>>(
                                     x_data, codes, scales, out_data, weight.n, weight.k, offset,
                                     count);
                                 CUDA_CHECK(cudaGetLastError());
                             };
                             if (count % Mxfp8MmaSchedule::kBlockTokens == 0) {
                                 launch.template operator()<true>();
                             } else {
                                 launch.template operator()<false>();
                             }
                         });
}

} // namespace ninfer::ops::detail
