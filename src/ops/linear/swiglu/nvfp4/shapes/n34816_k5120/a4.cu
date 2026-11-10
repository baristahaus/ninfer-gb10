#include "core/weight.h"
#include "ops/linear/swiglu/nvfp4/dispatch.h"

#include "core/device.h"
#include "ops/linear/kernels/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/swiglu/token_major_mma_epilogue.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_plan.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kInputRows = 5120;
// Column tiles amortize gate/up decode over the complete speculative block.
using M32N128  = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128  = Nvfp4A4MmaSchedule<64, 128, 256, 4, 4, 2, 1>;
using M128N128 = Nvfp4A4MmaSchedule<128, 128, 256, 4, 4, 2, 1>;
using M96N128  = Nvfp4A4MmaSchedule<96, 128, 256, 3, 4, 2, 1>;

// The intermediate is a compile-time tiling value and differs between the single device's
// [34816, 5120] object (17408) and one rank's Rows half ([17408, 5120] -> 8704).
template <int Intermediate, class Schedule>
void launch_gemm(const Weight& weight, Tensor& out, Nvfp4A4Workspace workspace, int tokens,
                 cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Schedule, kInputRows>;
    launch_nvfp4_a4_mma<S>(nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::RowMajor),
                           LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), Intermediate},
                           SwiGluTokenMajorMmaEpilogue{}, stream, SwiGluTokenMajorMmaRows<S>{});
}

template <int Intermediate, class Schedule>
void launch(const Tensor& x, const Weight& weight, Tensor& out, MemoryArena& workspace,
            cudaStream_t stream) {
    auto scope = workspace.scope();
    const Nvfp4A4Workspace scratch =
        allocate_nvfp4_a4_workspace(workspace, x.ne[1], kInputRows);
    launch_nvfp4_a4_quantize(x, weight, scratch, Nvfp4ScaleLayout::RowMajor, stream);
    launch_gemm<Intermediate, Schedule>(weight, out, scratch, x.ne[1], stream);
}

template <int Intermediate>
void launch_a4(const Tensor& x, const Weight& weight, Tensor& out, MemoryArena& workspace,
               cudaStream_t stream) {
    if (x.ne[1] <= 32) {
        launch<Intermediate, M32N128>(x, weight, out, workspace, stream);
    } else if (x.ne[1] <= M64N128::kBlockTokens) {
        launch<Intermediate, M64N128>(x, weight, out, workspace, stream);
    } else if (x.ne[1] <= M96N128::kBlockTokens) {
        launch<Intermediate, M96N128>(x, weight, out, workspace, stream);
    } else {
        launch<Intermediate, M128N128>(x, weight, out, workspace, stream);
    }
}
} // namespace

void launch_nvfp4_swiglu_a4(const Tensor& x, const Weight& weight, Tensor& out,
                            MemoryArena& workspace, cudaStream_t stream) {
    if (weight.n / 2 == 8704) return launch_a4<8704>(x, weight, out, workspace, stream);
    return launch_a4<17408>(x, weight, out, workspace, stream);
}

} // namespace ninfer::ops::detail
