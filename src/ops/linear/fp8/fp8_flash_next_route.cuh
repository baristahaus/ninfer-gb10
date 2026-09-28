#pragma once

// A16 routes for the Flash-Next dense FP8 projections on the unified FP8 templates: GEMV at T=1,
// SIMT at the MTP verify widths (T=2..4), sliced-K Tensor Core tiles through 64 tokens and tiled
// Tensor Core GEMMs beyond. Output and epilogue are template parameters, so a fused consumer (the
// HyperConnection down projection's scaled SiLU) runs the same route at every width. Schedules
// follow upstream's measured RTX 5090 selections for the nearest shapes; they are not yet tuned
// on GB10.

#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear/fp8/fp8_operands.h"
#include "ops/linear/fp8/fp8_template_launch.cuh"

#include <stdexcept>

namespace ninfer::ops::detail::flash_next {

using Fp8FlashNextSimtT2 =
    Fp8A16SimtSchedule<8, 2, 16, 2, 1, Fp8SimtActivationAccess::TokenPacked,
                       Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using Fp8FlashNextSimtT4 =
    Fp8A16SimtSchedule<8, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked,
                       Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;

template <class Geometry, class GemvSchedule, class Output, class Epilogue>
void launch_fp8_dense_a16(const Tensor& x, const Weight& weight, Output output, Epilogue epilogue,
                          cudaStream_t stream) {
    constexpr int K       = Geometry::kInputRows;
    const auto operands   = fp8_a16_operands(x, weight);
    const int tokens      = x.ne[1];
    if (tokens == 1) {
        return launch_fp8_a16_gemv<Fp8ScheduleInstance<GemvSchedule, K>>(operands, output,
                                                                         epilogue, stream);
    }
    if (tokens <= 2) {
        return launch_fp8_a16_simt<Fp8ScheduleInstance<Fp8FlashNextSimtT2, K, 2>>(
            operands, output, epilogue, stream);
    }
    if (tokens <= 4) {
        return launch_fp8_a16_simt<Fp8ScheduleInstance<Fp8FlashNextSimtT4, K, 4>>(
            operands, output, epilogue, stream);
    }
    if (tokens <= 8) {
        return launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<8, 8, 2>, K>>(
            operands, output, epilogue, stream);
    }
    if (tokens <= 16) {
        return launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<16, 8, 2>, K>>(
            operands, output, epilogue, stream);
    }
    if (tokens <= 32) {
        return launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<16, 4, 2>, K>>(
            operands, output, epilogue, stream);
    }
    if (tokens <= 64) {
        return launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<32, 4, 1>, K>>(
            operands, output, epilogue, stream);
    }
    if (tokens <= 128) {
        return launch_fp8_a16_mma<
            Fp8ScheduleInstance<Fp8A16MmaSchedule<64, 64, 128, 32, 16, 2, 2>, K>>(
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
