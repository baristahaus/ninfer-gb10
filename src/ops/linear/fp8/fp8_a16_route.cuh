#pragma once

// Complete A16 route for one exact row-scaled FP8 problem: a CUDA-core GEMV at T=1, K-split Tensor
// Core tiles through 48 tokens, and tiled Tensor Core GEMMs beyond. Activations stay BF16 on every
// route; each route decodes the persistent E4M3 codes exactly and applies the BF16 row multiplier
// once to the FP32 dot product. The token crossovers and the large-T tiling follow the measured
// [248320,5120] vocabulary route; the K-split warp count is the largest of that route's choices
// that divides K.

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/fp8/fp8_a16_gemm_mma.cuh"
#include "ops/linear/fp8/fp8_a16_ksplit_mma.cuh"
#include "ops/linear/fp8/fp8_gemv.cuh"
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kFp8A16RouteKSplitMaxTokens = 48;
inline constexpr std::int32_t kFp8A16RouteFirstGemmTokens = 42;

template <int InputRows, int ActiveTokens>
struct Fp8A16RouteKSplitSchedule {
    static constexpr int kTileTokens     = ActiveTokens <= 8    ? 8
                                           : ActiveTokens <= 16 ? 16
                                           : ActiveTokens <= 24 ? 24
                                           : ActiveTokens <= 32 ? 32
                                           : ActiveTokens <= 40 ? 40
                                                                : 48;
    static constexpr int kPreferredWarps = ActiveTokens <= 8 ? 16 : (ActiveTokens <= 24 ? 8 : 4);
    static constexpr int kWarps = (InputRows % (kPreferredWarps * 64)) == 0 ? kPreferredWarps
                                  : (InputRows % (kPreferredWarps / 2 * 64)) == 0
                                      ? kPreferredWarps / 2
                                      : 4;
    static_assert((InputRows % (kWarps * 64)) == 0, "K must be a multiple of 256");
    using Type = Fp8A16KSplitSchedule<kWarps, kTileTokens, kWarps == 16 ? 1 : 2>;
};

template <class Geometry, class GemvSchedule,
          class Tail64Schedule = Fp8A16GemmSchedule<128, 64, 64, 64, 16, 2, 2>>
struct Fp8A16Route {
    using Main128 = Fp8A16GemmSchedule<64, 128, 64, 64, 16, 2, 2>;
    using Tail64  = Tail64Schedule;
    using Tail96  = Fp8A16GemmSchedule<64, 96, 64, 64, 16, 2, 2>;
    static_assert(Tail64::kBlockTokens == 64);

    static void gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
        const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                         Geometry::kOutputRows};
        fp8_gemv_kernel<Geometry, GemvSchedule>
            <<<Geometry::kOutputRows / GemvSchedule::kRowsPerCta, GemvSchedule::kThreads, 0,
               stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                         static_cast<const std::uint8_t*>(weight.qdata),
                         static_cast<const __nv_bfloat16*>(weight.scales), output);
        CUDA_CHECK(cudaGetLastError());
    }

    template <int ActiveTokens>
    static void ksplit_tile(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream) {
        using Schedule =
            typename Fp8A16RouteKSplitSchedule<Geometry::kInputRows, ActiveTokens>::Type;
        const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                         Geometry::kOutputRows};
        fp8_a16_ksplit_mma_kernel<Geometry, ActiveTokens, Schedule, Fp8ContiguousOutput, true>
            <<<Geometry::kOutputRows / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const __nv_bfloat16*>(weight.scales), output, x.ne[1]);
        CUDA_CHECK(cudaGetLastError());
    }

    static void ksplit(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
        const int tokens = x.ne[1];
        if (tokens <= 8) return ksplit_tile<8>(x, weight, out, stream);
        if (tokens <= 16) return ksplit_tile<16>(x, weight, out, stream);
        if (tokens <= 24) return ksplit_tile<24>(x, weight, out, stream);
        if (tokens <= 32) return ksplit_tile<32>(x, weight, out, stream);
        if (tokens <= 40) return ksplit_tile<40>(x, weight, out, stream);
        if (tokens <= 48) return ksplit_tile<48>(x, weight, out, stream);
        throw std::logic_error("fp8 K-split exceeds its token capacity");
    }

    template <class Schedule, bool FullTokens>
    static void gemm_slice(const Tensor& x, const Weight& weight, Tensor& out,
                           cudaStream_t stream) {
        static_assert((Geometry::kOutputRows % Schedule::kBlockRows) == 0);
        static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
        constexpr int row_tiles = Geometry::kOutputRows / Schedule::kBlockRows;
        const int token_tiles   = div_up(x.ne[1], Schedule::kBlockTokens);
        const dim3 grid(static_cast<unsigned>(row_tiles), static_cast<unsigned>(token_tiles), 1U);
        const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                         Geometry::kOutputRows};
        fp8_a16_gemm_mma_kernel<Geometry, Schedule, FullTokens>
            <<<grid, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const __nv_bfloat16*>(weight.scales), output, x.ne[1]);
        CUDA_CHECK(cudaGetLastError());
    }

    template <class Schedule>
    static void gemm(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
        for_each_token_slice(x.ne[1], Schedule::kBlockTokens,
                             [&](std::int32_t offset, std::int32_t count) {
                                 const Tensor input = x.slice(1, offset, count);
                                 Tensor output      = out.slice(1, offset, count);
                                 if ((count % Schedule::kBlockTokens) == 0) {
                                     gemm_slice<Schedule, true>(input, weight, output, stream);
                                 } else {
                                     gemm_slice<Schedule, false>(input, weight, output, stream);
                                 }
                             });
    }

    static void tail(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
        if (x.ne[1] < kFp8A16RouteFirstGemmTokens) {
            ksplit(x, weight, out, stream);
        } else if (x.ne[1] <= Tail64::kBlockTokens) {
            gemm<Tail64>(x, weight, out, stream);
        } else {
            gemm<Tail96>(x, weight, out, stream);
        }
    }

    static void launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
        const std::int32_t tokens = x.ne[1];
        if (tokens == 1) return gemv(x, weight, out, stream);
        if (tokens < kFp8A16RouteFirstGemmTokens) return ksplit(x, weight, out, stream);
        if (tokens <= Tail64::kBlockTokens) return gemm<Tail64>(x, weight, out, stream);
        if (tokens <= Tail96::kBlockTokens) return gemm<Tail96>(x, weight, out, stream);
        if (tokens <= Main128::kBlockTokens) return gemm<Main128>(x, weight, out, stream);

        // Whole 96-token GEMMs beat a 128-token prefix plus a tail in two early packing intervals.
        if ((tokens >= 161 && tokens <= 192) || (tokens >= 257 && tokens <= 288)) {
            return gemm<Tail96>(x, weight, out, stream);
        }
        const std::int32_t remainder = tokens % Main128::kBlockTokens;
        if (remainder == 0 || remainder > Tail96::kBlockTokens) {
            return gemm<Main128>(x, weight, out, stream);
        }
        const std::int32_t prefix = tokens - remainder;
        const Tensor input_prefix = x.slice(1, 0, prefix);
        Tensor output_prefix      = out.slice(1, 0, prefix);
        gemm<Main128>(input_prefix, weight, output_prefix, stream);
        const Tensor input_tail = x.slice(1, prefix, remainder);
        Tensor output_tail      = out.slice(1, prefix, remainder);
        tail(input_tail, weight, output_tail, stream);
    }
};

} // namespace ninfer::ops::detail
