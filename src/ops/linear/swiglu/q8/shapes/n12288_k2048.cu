#include "ops/common/cuda/memory.cuh"
#include "ops/linear/swiglu/q8/launch.cuh"
#include "ops/linear/kernels/q8/q8_instances.cuh"
#include "ops/linear/kernels/q8/q8_geometry.h"
#include <utility>

namespace ninfer::ops::detail {

LinearSwiGluLaunch select_q8_swiglu_n12288_k2048(int tokens) {
    if (tokens <= 64) return launch_q8_swiglu_mma<Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>>;
    if (tokens <= 80) return launch_q8_swiglu_mma<Q8A16MmaSchedule<32, 80, 64, 32, 16, 2, 3>>;
    if (tokens <= 96) return launch_q8_swiglu_mma<Q8A16MmaSchedule<32, 96, 64, 32, 16, 2, 2>>;
    if (tokens <= 128) return launch_q8_swiglu_mma<Q8A16MmaSchedule<64, 64, 64, 64, 16, 2, 2>>;
    if (tokens <= 192) return launch_q8_swiglu_mma<Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>>;
    if (tokens <= 240) return launch_q8_swiglu_mma<Q8A16MmaSchedule<128, 80, 64, 64, 16, 2, 2>>;
    if (tokens <= 255) return launch_q8_swiglu_mma<Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>>;
    if (tokens <= 256) return launch_q8_swiglu_mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
    if (tokens <= 264) return launch_q8_swiglu_mma<Q8A16MmaSchedule<64, 64, 64, 64, 16, 2, 2>>;
    if (tokens <= 288) return launch_q8_swiglu_mma<Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>;
    if (tokens <= 320) return launch_q8_swiglu_mma<Q8A16MmaSchedule<64, 64, 64, 64, 16, 2, 2>>;
    if (tokens <= 384) return launch_q8_swiglu_mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
    if (tokens <= 448) return launch_q8_swiglu_mma<Q8A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>;
    if (tokens <= 512) return launch_q8_swiglu_mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
    if (tokens <= 560) return launch_q8_swiglu_mma<Q8A16MmaSchedule<128, 80, 64, 64, 16, 2, 2>>;
    return launch_q8_swiglu_mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}
} // namespace ninfer::ops::detail
