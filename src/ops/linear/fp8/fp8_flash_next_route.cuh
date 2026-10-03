#pragma once

// A16 routes for the Flash-Next dense FP8 projections on the unified FP8 templates: GEMV at T=1,
// sliced-K Tensor Core tiles through 64 tokens and tiled Tensor Core GEMMs beyond. T=2..4 run
// the same sliced-K route as the wider MTP verify batches rather than the CUDA-core SIMT
// kernels: the verify forward (W=3 rows at single-stream serving) must keep one Tensor Core
// per-element profile for the whole round, or the draft/verify argmax agreement drops (measured
// 65.0% -> 63.2-63.6% MTP acceptance on GB10 with the SIMT selection, which carried no speed
// benefit at these widths). Output and epilogue are template parameters, so a fused consumer
// (the HyperConnection down projection's scaled SiLU) runs the same route at every width.
// Schedules follow upstream's measured selections for the nearest shapes; the T=2..4
// divergence above is the one GB10 re-tune. The short-K HyperConnection up (K=320) and
// shared-expert down (K=640) projections take the same route with the sliced-K warp count and
// GEMM K tile that their K admits: two K-warps (K=320 ends in a one-warp tail group) and 64-wide
// GEMM K tiles where K is not a multiple of 128.

#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear/fp8/fp8_operands.h"
#include "ops/linear/fp8/fp8_template_launch.cuh"

#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail::flash_next {

// The preferred sliced-K warp count when K is a whole number of its groups, otherwise the
// largest smaller even count that divides K, otherwise two warps with a partial last group.
template <int K, int Preferred>
constexpr int sliced_k_warps() {
    for (int warps = Preferred; warps >= 2; warps /= 2)
        if (K % (warps * 64) == 0) return warps;
    return 2;
}

// The T <= 16 sliced-K tiles. Long-K shapes (K >= 6144: the [2560,6144] mixer output and the
// [320,10240] HyperConnection down) keep three K groups in flight, which their 15-20 groups per
// tile can fill; shorter K stays at two stages. Stage depth changes no output bit (GB10
// small-T probe, 2026-10-03: hc.down 74.5 -> 78.5% and out 94.1 -> 96.1% of a plain read at
// T = 8; shared gate/up and shared down lose 1-4 points at three stages).
template <int Tokens, int Warps, int K>
using Fp8FlashNextDecodeTile =
    std::conditional_t<(K >= 6144),
                       Fp8A16SlicedKMmaSchedule<Warps, Tokens, 1, Cache::ca, Cache::cg,
                                                Fp8ActivationStage::PaddedZero, 3>,
                       Fp8SlicedInstance<Tokens, Warps, 2>>;

template <class Geometry, class GemvSchedule, class Output, class Epilogue>
void launch_fp8_dense_a16(const Tensor& x, const Weight& weight, Output output, Epilogue epilogue,
                          cudaStream_t stream) {
    constexpr int K = Geometry::kInputRows;
    static_assert(K % 64 == 0, "the Flash-Next FP8 route needs whole 64-column K tiles");
    constexpr int kWideWarps   = sliced_k_warps<K, 8>();
    constexpr int kNarrowWarps = sliced_k_warps<K, 4>();
    constexpr int kGemmK       = K % 128 == 0 ? 128 : 64;
    const auto operands        = fp8_a16_operands(x, weight);
    const int tokens      = x.ne[1];
    if (tokens == 1) {
        return launch_fp8_a16_gemv<Fp8ScheduleInstance<GemvSchedule, K>>(operands, output,
                                                                         epilogue, stream);
    }
    if (tokens <= 8) {
        return launch_fp8_a16_sliced_k_mma<
            Fp8ScheduleInstance<Fp8FlashNextDecodeTile<8, kWideWarps, K>, K>>(operands, output,
                                                                               epilogue, stream);
    }
    if (tokens <= 16) {
        return launch_fp8_a16_sliced_k_mma<
            Fp8ScheduleInstance<Fp8FlashNextDecodeTile<16, kWideWarps, K>, K>>(operands, output,
                                                                                epilogue, stream);
    }
    if (tokens <= 32) {
        return launch_fp8_a16_sliced_k_mma<
            Fp8ScheduleInstance<Fp8SlicedInstance<16, kNarrowWarps, 2>, K>>(operands, output,
                                                                            epilogue, stream);
    }
    if (tokens <= 64) {
        return launch_fp8_a16_sliced_k_mma<
            Fp8ScheduleInstance<Fp8SlicedInstance<32, kNarrowWarps, 1>, K>>(operands, output,
                                                                            epilogue, stream);
    }
    if (tokens <= 128) {
        return launch_fp8_a16_mma<
            Fp8ScheduleInstance<Fp8A16MmaSchedule<64, 64, kGemmK, 32, 16, 2, 2>, K>>(
            operands, output, epilogue, stream);
    }
    launch_fp8_a16_mma<Fp8ScheduleInstance<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, K>>(
        operands, output, epilogue, stream);
}

template <class Geometry, class GemvSchedule>
void launch_fp8_dense_linear(const Tensor& x, const Weight& weight, Tensor& out,
                             cudaStream_t stream) {
    launch_fp8_dense_a16<Geometry, GemvSchedule>(
        x, weight, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
        LinearIdentityEpilogue{}, stream);
}

// The 248320-row output head: GEMV at T=1, sliced-K tiles below 42 tokens (warp counts that
// divide K=2560), then the vocabulary GEMM schedules with their packing intervals and tails, as
// upstream's [248320,5120] route.
template <int K, class GemvSchedule>
struct Fp8FlashNextVocabularyRoute {
    using Main128 = Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>;
    using Tail64  = Fp8A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>;
    using Tail96  = Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>;

    template <int ActiveTokens>
    static void tile(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
        constexpr int kPreferredWarps = ActiveTokens <= 24 ? 8 : 4;
        constexpr int kWarps          = K % (kPreferredWarps * 64) == 0 ? kPreferredWarps : 2;
        using Schedule = Fp8A16SlicedKMmaSchedule<kWarps, ActiveTokens, ActiveTokens <= 8 ? 1 : 2>;
        static_assert(K % Schedule::kBlockK == 0);
        launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, K, ActiveTokens>>(
            fp8_a16_operands(x, weight),
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
            LinearIdentityEpilogue{}, stream);
    }

    static void sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
        const int tokens = x.ne[1];
        if (tokens <= 8) return tile<8>(x, weight, out, stream);
        if (tokens <= 16) return tile<16>(x, weight, out, stream);
        if (tokens <= 24) return tile<24>(x, weight, out, stream);
        if (tokens <= 32) return tile<32>(x, weight, out, stream);
        if (tokens <= 40) return tile<40>(x, weight, out, stream);
        if (tokens <= 48) return tile<48>(x, weight, out, stream);
        throw std::logic_error("fp8 vocabulary sliced-K exceeds its capacity");
    }

    template <class Schedule>
    static void gemm(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
        launch_fp8_a16_mma<Fp8ScheduleInstance<Schedule, K>>(
            fp8_a16_operands(x, weight),
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
            LinearIdentityEpilogue{}, stream);
    }

    static void tail(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
        if (x.ne[1] < 42) return sliced(x, weight, out, stream);
        if (x.ne[1] <= Tail64::kBlockTokens) return gemm<Tail64>(x, weight, out, stream);
        gemm<Tail96>(x, weight, out, stream);
    }

    static void launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
        const int tokens = x.ne[1];
        if (tokens == 1) {
            return launch_fp8_a16_gemv<Fp8ScheduleInstance<GemvSchedule, K>>(
                fp8_a16_operands(x, weight),
                LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
                LinearIdentityEpilogue{}, stream);
        }
        if (tokens < 42) return sliced(x, weight, out, stream);
        if (tokens <= Tail64::kBlockTokens) return gemm<Tail64>(x, weight, out, stream);
        if (tokens <= Tail96::kBlockTokens) return gemm<Tail96>(x, weight, out, stream);
        if (tokens <= Main128::kBlockTokens) return gemm<Main128>(x, weight, out, stream);
        // Two early packing intervals run faster as whole 96-token GEMMs than as a 128-token
        // prefix plus a tail launch.
        if ((tokens >= 161 && tokens <= 192) || (tokens >= 257 && tokens <= 288)) {
            return gemm<Tail96>(x, weight, out, stream);
        }
        const int remainder = tokens % Main128::kBlockTokens;
        if (remainder == 0 || remainder > Tail96::kBlockTokens) {
            return gemm<Main128>(x, weight, out, stream);
        }
        const int prefix          = tokens - remainder;
        const Tensor input_prefix = x.slice(1, 0, prefix);
        Tensor output_prefix      = out.slice(1, 0, prefix);
        gemm<Main128>(input_prefix, weight, output_prefix, stream);
        const Tensor input_tail = x.slice(1, prefix, remainder);
        Tensor output_tail      = out.slice(1, prefix, remainder);
        tail(input_tail, weight, output_tail, stream);
    }
};

} // namespace ninfer::ops::detail::flash_next
