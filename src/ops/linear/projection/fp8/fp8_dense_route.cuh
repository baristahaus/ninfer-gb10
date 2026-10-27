#pragma once

// A16 routes for the Flash-Next dense FP8 projections on the unified FP8 templates: GEMV at
// T=1, sliced-K Tensor Core tiles through 64 tokens and tiled Tensor Core GEMMs beyond. This
// mirrors ops/linear/fp8/fp8_flash_next_route.cuh on the kernels/fp8 headers so the projection
// dispatch can register Flash-Next shapes without mixing the two fp8_a8_plan headers.

#include "ops/linear/kernels/fp8/fp8_geometry.h"
#include "ops/linear/kernels/fp8/fp8_instances.cuh"
#include "ops/linear/kernels/fp8/fp8_launch.h"
#include "ops/linear/kernels/fp8/fp8_operands.h"
#include "ops/linear/kernels/fp8/fp8_schedule.cuh"
#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace fp8_dense {

template <int K, int Preferred>
constexpr int sliced_k_warps() {
    for (int warps = Preferred; warps >= 2; warps /= 2)
        if (K % (warps * 64) == 0) return warps;
    return 2;
}

template <class Geometry, class GemvSchedule>
void launch_dense_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int K = Geometry::kInputRows;
    static_assert(K % 64 == 0, "the Flash-Next FP8 route needs whole 64-column K tiles");
    constexpr int kWideWarps   = sliced_k_warps<K, 8>();
    constexpr int kNarrowWarps = sliced_k_warps<K, 4>();
    constexpr int kGemmK       = K % 128 == 0 ? 128 : 64;
    const auto operands        = fp8_a16_operands(x, weight);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), weight.n};
    const int tokens = x.ne[1];
    if (tokens == 1) {
        // K=320 is a whole number of lane packs but not of warp phases; the gemv kernel's
        // predicated tail phase covers it (master fp8_a16_gemv.cuh semantics).
        return launch_fp8_a16_gemv_tail<Fp8ScheduleInstance<GemvSchedule, K>>(
            operands, output, LinearIdentityEpilogue{}, stream);
    }
    if (tokens <= 8) {
        return launch_fp8_a16_sliced_k_mma<
            Fp8ScheduleInstance<Fp8SlicedInstance<8, kWideWarps, 2>, K>>(
            operands, output, LinearIdentityEpilogue{}, stream);
    }
    if (tokens <= 16) {
        return launch_fp8_a16_sliced_k_mma<
            Fp8ScheduleInstance<Fp8SlicedInstance<16, kWideWarps, 2>, K>>(
            operands, output, LinearIdentityEpilogue{}, stream);
    }
    if (tokens <= 64) {
        return launch_fp8_a16_sliced_k_mma<
            Fp8ScheduleInstance<Fp8SlicedInstance<32, kNarrowWarps, 1>, K>>(
            operands, output, LinearIdentityEpilogue{}, stream);
    }
    if (tokens <= 128) {
        return launch_fp8_a16_mma<
            Fp8ScheduleInstance<Fp8A16MmaSchedule<64, 64, kGemmK, 32, 16, 2, 2>, K>>(
            operands, output, LinearIdentityEpilogue{}, stream);
    }
    launch_fp8_a16_mma<
        Fp8ScheduleInstance<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, K>>(
        operands, output, LinearIdentityEpilogue{}, stream);
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
        using Schedule                = Fp8A16SlicedKMmaSchedule<kWarps, ActiveTokens,
                                                             ActiveTokens <= 8 ? 1 : 2>;
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

    static void tail(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
        if (x.ne[1] < 42) return sliced(x, weight, out, stream);
        if (x.ne[1] <= Tail64::kBlockTokens) return gemm<Tail64>(x, weight, out, stream);
        gemm<Tail96>(x, weight, out, stream);
    }
};

} // namespace fp8_dense
} // namespace ninfer::ops::detail
