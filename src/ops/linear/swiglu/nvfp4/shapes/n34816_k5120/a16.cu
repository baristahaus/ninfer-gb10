#include "core/weight.h"
#include "ops/linear/swiglu/nvfp4/dispatch.h"

#include "core/device.h"
#include "ops/linear/kernels/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"
#include "ops/linear/swiglu/paired_output.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

// Two shapes are served here: the single device's [34816, 5120] fused gate_up (intermediate 17408)
// and one rank's Rows half at tensor-parallel width 2 ([17408, 5120], intermediate 8704). The input
// width is 5120 in both, so only the intermediate — a compile-time tiling value — needs a branch.
template <int Intermediate>
void launch_two_tokens(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Nvfp4A16SimtSchedule<8, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::SharedPhase,
                                          Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                          Nvfp4SimtBlockOrder::RowsContiguous, 1>;
    launch_nvfp4_a16_simt<Nvfp4ScheduleInstance<Schedule, 5120, 2, true>>(
        nvfp4_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), Intermediate},
        SwiGluPairEpilogue{}, stream, SwiGluBranchRows<1>{});
}


} // namespace

template <int Intermediate>
void launch_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const auto p = nvfp4_a16_operands(x, weight);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), Intermediate};
    if (x.ne[1] <= 2)
        launch_two_tokens<Intermediate>(x, weight, out, stream);
    else if (x.ne[1] <= 8)
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, 5120>>(
            p, output, SwiGluPairEpilogue{}, stream, SwiGluBranchRows<8>{});
    else
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, 5120>>(
            p, output, SwiGluPairEpilogue{}, stream, SwiGluBranchRows<8>{});
}

void launch_nvfp4_swiglu_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                 cudaStream_t stream) {
    if (weight.n / 2 == 8704) return launch_small_t<8704>(x, weight, out, stream);
    return launch_small_t<17408>(x, weight, out, stream);
}

void launch_nvfp4_swiglu_decode(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<
        Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 2>,
        5120>;
    launch_nvfp4_a16_gemv<S>(nvfp4_a16_operands(x, w),
                             LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n / 2},
                             SwiGluPairEpilogue{}, stream, SwiGluBranchRows<1>{});
}
} // namespace ninfer::ops::detail
