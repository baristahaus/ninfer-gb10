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
        return launch_fp8_a16_gemv<Fp8ScheduleInstance<GemvSchedule, K>>(operands, output,
                                                                         LinearIdentityEpilogue{},
                                                                         stream);
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

} // namespace fp8_dense
} // namespace ninfer::ops::detail
