#pragma once

// A16 routes for the Flash-Next dense FP8 projections on the unified FP8 templates: GEMV at T=1,
// sliced-K Tensor Core tiles through 64 tokens and tiled Tensor Core GEMMs beyond. T=2..4 run the
// same sliced-K route as the wider MTP verify batches rather than the CUDA-core SIMT kernels: the
// verify forward (W=3 rows at single-stream serving) must keep one Tensor Core per-element profile
// for the whole round, or the draft/verify argmax agreement drops (measured 65.0% -> 63.2-63.6% MTP
// acceptance on GB10 with the SIMT selection, which carried no speed benefit at these widths).
// Output and epilogue are template parameters, so a fused consumer (the HyperConnection down
// projection's scaled SiLU, or the up projection's gate mix) runs the same route at every width.
//
// MOVED HERE from ops/linear/fp8/fp8_flash_next_route.cuh with the fp8 tree merge. Only the generic
// form came across: the flat route's launch_fp8_dense_linear and Fp8FlashNextVocabularyRoute have
// no consumer, because the projection shapes already use fp8_dense's own copies of them.

#include "ops/linear/kernels/fp8/fp8_instances.cuh"
#include "ops/linear/kernels/fp8/fp8_launch.h"
#include "ops/linear/kernels/fp8/fp8_operands.h"
#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"

namespace ninfer::ops::detail::flash_next {

// The preferred sliced-K warp count when K is a whole number of its groups, otherwise the largest
// smaller even count that divides K, otherwise two warps with a partial last group.
template <int K, int Preferred>
constexpr int sliced_k_warps() {
    for (int warps = Preferred; warps >= 2; warps /= 2)
        if (K % (warps * 64) == 0) return warps;
    return 2;
}

template <class Geometry, class GemvSchedule, class Output, class Epilogue>
void launch_fp8_dense_a16(const Tensor& x, const Weight& weight, Output output, Epilogue epilogue,
                          cudaStream_t stream) {
    constexpr int K = Geometry::kInputRows;
    static_assert(K % 64 == 0, "the Flash-Next FP8 route needs whole 64-column K tiles");
    constexpr int kWideWarps   = sliced_k_warps<K, 8>();
    constexpr int kNarrowWarps = sliced_k_warps<K, 4>();
    constexpr int kGemmK       = K % 128 == 0 ? 128 : 64;
    const auto operands        = fp8_a16_operands(x, weight);
    const int tokens           = x.ne[1];
    if (tokens <= 8) {
        return launch_fp8_a16_sliced_k_mma<
            Fp8ScheduleInstance<Fp8SlicedInstance<8, kWideWarps, 2>, K>>(operands, output, epilogue,
                                                                         stream);
    }
    if (tokens <= 16) {
        return launch_fp8_a16_sliced_k_mma<
            Fp8ScheduleInstance<Fp8SlicedInstance<16, kWideWarps, 2>, K>>(operands, output,
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

} // namespace ninfer::ops::detail::flash_next
