#include "ops/linear/kernels/fp8/fp8_instances.cuh"
#include "ops/linear/swiglu/row_major_mma_epilogue.cuh"
#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/linear/swiglu/fp8/dispatch.h"

#include "core/device.h"
#include "ops/linear/kernels/fp8/fp8_schedule.cuh"
#include "ops/linear/kernels/fp8/fp8_a16_simt.cuh"
#include "ops/linear/swiglu/paired_output.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// The fused gate_up object's input width, the same for every shape this file serves: 5120.
constexpr int kInputRows = 5120;

// `kIntermediate` is the fused object's output rows over two, and it is a COMPILE-TIME value
// because SwiGluBranchRows tiles by it. Two shapes are served here: the single device's [34816,
// 5120] object (intermediate 17408) and one rank's Rows half at tensor-parallel width 2 ([17408,
// 5120], intermediate 8704). The matrix path below reads weight.n / 2 at run time and needs no
// branch; only this small-T path does.
template <int Intermediate>
void launch_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule =
        Fp8A16SimtSchedule<4, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked,
                           Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
    using Rows = SwiGluBranchRows<Schedule::kRowsPerWarp / 2, Intermediate>;
    launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, kInputRows, 4>>(
        fp8_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), Intermediate},
        SwiGluPairEpilogue{}, stream, Rows{});
}
} // namespace

void launch_fp8_swiglu_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                               cudaStream_t stream) {
    if (weight.n / 2 == 8704) return launch_small_t<8704>(x, weight, out, stream);
    return launch_small_t<17408>(x, weight, out, stream);
}

void launch_fp8_swiglu_matrix(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream) {
    const auto p = fp8_a16_operands(x, weight);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2};
    const auto sliced = [&]<int T, int W, int Stages>() {
        using S = Fp8ScheduleInstance<Fp8SlicedInstance<T, W, Stages>, 5120>;
        launch_fp8_a16_sliced_k_mma<S>(p, output, SwiGluPairEpilogue{}, stream,
                                       SwiGluRowMajorMmaRows<S>{});
    };
    const auto mma = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, 5120>;
        launch_fp8_a16_mma<S>(p, output, SwiGluRowMajorMmaEpilogue{}, stream,
                              SwiGluRowMajorMmaRows<S>{});
    };
    if (x.ne[1] <= 8) return sliced.template operator()<8, 4, 1>();
    if (x.ne[1] <= 16) return sliced.template operator()<16, 4, 1>();
    if (x.ne[1] <= 24) return sliced.template operator()<32, 8, 1>();
    if (x.ne[1] <= 32) return sliced.template operator()<32, 4, 1>();
    if (x.ne[1] <= 64)
        return mma.template operator()<Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>();
    if (x.ne[1] <= 96)
        return mma.template operator()<Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>();
    mma.template operator()<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>();
}
} // namespace ninfer::ops::detail
