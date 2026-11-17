#include "ops/linear/add/dispatch.h"
#include "ops/linear/add/q8/launch.cuh"

#include <array>
#include <utility>

namespace ninfer::ops::detail {
namespace {
LinearAddLaunch select_medium(std::int32_t tokens) {
    if (tokens <= 64)
        return launch_q8_add_grouped<
            Q8A16GroupedSlicedKMmaSchedule<64, 8, 4, 1, 1, 4096, Cache::cg, Cache::cg, false>>;
    if (tokens <= 65)
        return launch_q8_add_grouped<
            Q8A16GroupedSlicedKMmaSchedule<80, 8, 2, 1, 1, 4096, Cache::cg, Cache::cg, false>>;
    if (tokens <= 72)
        return launch_q8_add_grouped<
            Q8A16GroupedSlicedKMmaSchedule<72, 8, 3, 1, 1, 4096, Cache::cg, Cache::cg, false>>;
    if (tokens <= 80)
        return launch_q8_add_grouped<
            Q8A16GroupedSlicedKMmaSchedule<80, 8, 2, 1, 1, 4096, Cache::cg, Cache::cg, false>>;
    if (tokens <= 96)
        return launch_q8_add_grouped<
            Q8A16GroupedSlicedKMmaSchedule<96, 4, 6, 1, 1, 4096, Cache::cg, Cache::cg, false>>;
    if (tokens <= 112)
        return launch_q8_add_grouped<
            Q8A16GroupedSlicedKMmaSchedule<112, 4, 7, 1, 1, 4096, Cache::cg, Cache::cg, false>>;
    if (tokens <= 120)
        return launch_q8_add_grouped<
            Q8A16GroupedSlicedKMmaSchedule<120, 4, 5, 1, 1, 4096, Cache::cg, Cache::cg, false>>;
    if (tokens <= 125)
        return launch_q8_add_grouped<
            Q8A16GroupedSlicedKMmaSchedule<128, 4, 4, 1, 1, 4096, Cache::cg, Cache::cg, false>>;
    return launch_q8_add_grouped<
        Q8A16GroupedSlicedKMmaSchedule<128, 4, 8, 1, 1, 4096, Cache::cg, Cache::cg, false>>;
}
} // namespace

LinearAddLaunch select_q8_linear_add_n2048_k4096(std::int32_t tokens) {
    if (tokens <= 128) return select_medium(tokens);
    if (tokens <= 640) return launch_q8_add_mma<Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>>;
    return launch_q8_add_mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}
} // namespace ninfer::ops::detail
