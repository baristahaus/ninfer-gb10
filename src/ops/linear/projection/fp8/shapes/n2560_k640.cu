#include "ops/linear/kernels/fp8/fp8_geometry.h"
#include "ops/linear/kernels/fp8/fp8_schedule.cuh"
#include "ops/linear/projection/fp8/fp8_dense_route.cuh"
#include "ops/linear/projection/fp8/fp8_shapes.h"

// Flash-Next shared-expert down projection [2560,640]. At T=1 each row is one full 512-value
// phase and a predicated 128-value tail; the schedule is a starting point, not a measured
// winner.

namespace ninfer::ops::detail {
namespace {
using Geometry = Fp8Geometry<2560, 640>;
using Gemv     = Fp8A16GemvSchedule<4, 2, 16, 4, Fp8CodeCache::Streaming, 1, 4>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    // K=640 is not a whole number of sliced-K groups, so tokens > 1 run the 64-wide-GEMM-K
    // Tensor Core schedules directly; T=1 uses the predicated-tail GEMV.
    const auto operands = fp8_a16_operands(x, weight);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), weight.n};
    if (x.ne[1] == 1) {
        return launch_fp8_a16_gemv_tail<Fp8ScheduleInstance<Gemv, 640>>(
            operands, output, LinearIdentityEpilogue{}, stream);
    }
    if (x.ne[1] <= 128) {
        return launch_fp8_a16_mma<
            Fp8ScheduleInstance<Fp8A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>, 640>>(
            operands, output, LinearIdentityEpilogue{}, stream);
    }
    launch_fp8_a16_mma<
        Fp8ScheduleInstance<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, 640>>(
        operands, output, LinearIdentityEpilogue{}, stream);
}

bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8ProjN2560K640{2560, 640, launch_a16, nullptr, uses_a8};
} // namespace ninfer::ops::detail
