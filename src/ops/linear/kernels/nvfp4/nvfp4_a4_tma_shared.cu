#include "ops/linear/kernels/nvfp4/nvfp4_a4_tma_shared.h"

#include "ops/linear/kernels/nvfp4/nvfp4_a4_tma.cuh"

namespace ninfer::ops::detail {

void launch_nvfp4_a4_tma_segmented_6144_1024(const Nvfp4A4Operands& operands,
                                             Nvfp4Segmented6144x1024 output, cudaStream_t stream) {
    if (operands.scale_layout == Nvfp4ScaleLayout::Tiled128)
        launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<128, 4, 1>, 5120>>(
            operands, output, LinearIdentityEpilogue{}, stream);
    else
        launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1>, 5120>>(
            operands, output, LinearIdentityEpilogue{}, stream);
}

void launch_nvfp4_a4_tma_segmented_10240_6144(const Nvfp4A4Operands& operands,
                                              Nvfp4Segmented10240x6144 output,
                                              cudaStream_t stream) {
    launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1>, 5120>>(
        operands, output, LinearIdentityEpilogue{}, stream);
}

} // namespace ninfer::ops::detail
