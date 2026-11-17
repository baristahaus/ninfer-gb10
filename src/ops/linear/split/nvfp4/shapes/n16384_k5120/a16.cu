#include "ops/linear/split/nvfp4/shapes/n16384_k5120/launch.h"
#include "core/weight.h"
#include "core/device.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail::linear_split::nvfp4_n16384_k5120 {
namespace {

} // namespace

void launch_a16(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
                cudaStream_t stream) {
    const int tokens = x.ne[1];
    const auto p      = nvfp4_a16_operands(x, weight);
    const auto output = LinearBf16SegmentedOutput<10240, 6144>{
        static_cast<__nv_bfloat16*>(outputs[0].data), static_cast<__nv_bfloat16*>(outputs[1].data)};
    const auto epilogue = LinearIdentityEpilogue{};
    if (tokens <= 8) {
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 16) {
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 24) {
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 8, 1>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 32) {
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 8, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 64) {
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 96) {
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    {
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
}

} // namespace ninfer::ops::detail::linear_split::nvfp4_n16384_k5120
