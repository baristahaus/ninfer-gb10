// Flash-Next attention and GDN projections in the optional row-scaled FP8 profile. All routes
// keep BF16 activations (weight-only FP8): one token uses the GEMV, compact batches (MTP
// verification, small batches) the K-split Tensor Core kernel, and prefill the A16 GEMM.
// Schedules were measured on 48 cold weight copies per shape (RTX PRO 6000).
#include "ops/linear/fp8/fp8_shapes.h"
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/fp8/fp8_a16_gemm_mma.cuh"
#include "ops/linear/fp8/fp8_a16_ksplit_mma.cuh"
#include "ops/linear/fp8/fp8_gemv.cuh"
#include "ops/linear/fp8/fp8_output.cuh"

namespace ninfer::ops::detail {
namespace {

using Gemv     = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using Prefill  = Fp8A16GemmSchedule<64, 128, 64, 64, 16, 2, 2>;
constexpr int kKSplitMaxTokens = 16;

template <class Geometry>
void launch_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    fp8_gemv_kernel<Geometry, Gemv><<<Geometry::kOutputRows / Gemv::kRowsPerCta, Gemv::kThreads,
                                      0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales),
        Fp8ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows});
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int TileTokens>
void launch_ksplit(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Fp8A16KSplitSchedule<8, TileTokens, 1>;
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    fp8_a16_ksplit_mma_kernel<Geometry, TileTokens, Schedule, Fp8ContiguousOutput, true>
        <<<Geometry::kOutputRows / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales),
            Fp8ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows},
            x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, bool FullTokens>
void launch_prefill_slice(const Tensor& x, const Weight& weight, Tensor& out,
                          cudaStream_t stream) {
    static_assert((Geometry::kOutputRows % Prefill::kBlockRows) == 0);
    static_assert((Geometry::kInputRows % Prefill::kBlockK) == 0);
    const dim3 grid(static_cast<unsigned>(Geometry::kOutputRows / Prefill::kBlockRows),
                    static_cast<unsigned>(div_up(x.ne[1], Prefill::kBlockTokens)), 1U);
    if constexpr (Prefill::kSharedBytes > 48 * 1024) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            fp8_a16_gemm_mma_kernel<Geometry, Prefill, FullTokens>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Prefill::kSharedBytes);
        CUDA_CHECK(attribute);
    }
    fp8_a16_gemm_mma_kernel<Geometry, Prefill, FullTokens>
        <<<grid, Prefill::kThreads, Prefill::kSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales),
            Fp8ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows},
            x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return launch_gemv<Geometry>(x, weight, out, stream);
    if (tokens <= 8) return launch_ksplit<Geometry, 8>(x, weight, out, stream);
    if (tokens <= kKSplitMaxTokens) return launch_ksplit<Geometry, 16>(x, weight, out, stream);
    for_each_token_slice(tokens, Prefill::kBlockTokens, [&](std::int32_t offset, std::int32_t count) {
        const Tensor input = x.slice(1, offset, count);
        Tensor output      = out.slice(1, offset, count);
        if ((count % Prefill::kBlockTokens) == 0) {
            launch_prefill_slice<Geometry, true>(input, weight, output, stream);
        } else {
            launch_prefill_slice<Geometry, false>(input, weight, output, stream);
        }
    });
}

bool uses_a8(std::int32_t, std::int32_t) { return false; }

template <int N, int K>
constexpr Fp8LinearShape flash_next_shape() {
    return {N, K, launch_a16<Fp8Geometry<N, K>>, nullptr, uses_a8};
}
} // namespace

const Fp8LinearShape kFp8N12288K2560 = flash_next_shape<12288, 2560>();
const Fp8LinearShape kFp8N10240K2560 = flash_next_shape<10240, 2560>();
const Fp8LinearShape kFp8N6144K2560  = flash_next_shape<6144, 2560>();
const Fp8LinearShape kFp8N2560K6144  = flash_next_shape<2560, 6144>();
const Fp8LinearShape kFp8N512K2560   = flash_next_shape<512, 2560>();
} // namespace ninfer::ops::detail
