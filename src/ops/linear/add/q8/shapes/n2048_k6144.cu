#include "ops/linear/add/dispatch.h"
#include "ops/linear/add/q8/launch.cuh"

#include <array>
#include <utility>

namespace ninfer::ops::detail {
namespace {
LinearAddLaunch select_medium(std::int32_t tokens) {
    if (tokens <= 64)
        return launch_q8_add_grouped<
            Q8A16GroupedSlicedKMmaSchedule<64, 8, 4, 1, 1, 6144, Cache::cg, Cache::cg, false>>;
    if (tokens <= 65)
        return launch_q8_add_grouped<
            Q8A16GroupedSlicedKMmaSchedule<80, 8, 2, 1, 1, 6144, Cache::cg, Cache::cg, false>>;
    if (tokens <= 72)
        return launch_q8_add_grouped<
            Q8A16GroupedSlicedKMmaSchedule<72, 8, 3, 1, 1, 6144, Cache::cg, Cache::cg, false>>;
    if (tokens <= 80)
        return launch_q8_add_grouped<
            Q8A16GroupedSlicedKMmaSchedule<80, 8, 2, 1, 1, 6144, Cache::cg, Cache::cg, false>>;
    if (tokens <= 96)
        return launch_q8_add_grouped<
            Q8A16GroupedSlicedKMmaSchedule<96, 4, 6, 1, 1, 6144, Cache::cg, Cache::cg, false>>;
    if (tokens <= 112)
        return launch_q8_add_grouped<
            Q8A16GroupedSlicedKMmaSchedule<112, 4, 7, 1, 1, 6144, Cache::cg, Cache::cg, false>>;
    if (tokens <= 120)
        return launch_q8_add_grouped<
            Q8A16GroupedSlicedKMmaSchedule<120, 4, 5, 1, 1, 6144, Cache::cg, Cache::cg, false>>;
    if (tokens <= 125)
        return launch_q8_add_grouped<
            Q8A16GroupedSlicedKMmaSchedule<128, 4, 4, 1, 1, 6144, Cache::cg, Cache::cg, false>>;
    return launch_q8_add_grouped<
        Q8A16GroupedSlicedKMmaSchedule<128, 4, 8, 1, 1, 6144, Cache::cg, Cache::cg, false>>;
}
} // namespace

LinearAddLaunch select_q8_linear_add_n2048_k6144(std::int32_t tokens) {
    if (tokens <= 128) return select_medium(tokens);
    if (tokens <= 191) return launch_q8_add_mma<Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>>;
    if (tokens <= 192) return launch_q8_add_mma<Q8A16MmaSchedule<32, 96, 64, 32, 16, 2, 2>>;
    if (tokens <= 256) return launch_q8_add_mma<Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>>;
    if (tokens <= 384) return launch_q8_add_mma<Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>>;
    if (tokens <= 399) return launch_q8_add_mma<Q8A16MmaSchedule<32, 96, 64, 32, 16, 2, 2>>;
    if (tokens <= 400) return launch_q8_add_mma<Q8A16MmaSchedule<32, 80, 64, 32, 16, 2, 3>>;
    if (tokens <= 447) return launch_q8_add_mma<Q8A16MmaSchedule<32, 96, 64, 32, 16, 2, 2>>;
    if (tokens <= 448) return launch_q8_add_mma<Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>>;
    if (tokens <= 480) return launch_q8_add_mma<Q8A16MmaSchedule<32, 96, 64, 32, 16, 2, 2>>;
    if (tokens <= 640) return launch_q8_add_mma<Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>>;
    if (tokens <= 672) return launch_q8_add_mma<Q8A16MmaSchedule<48, 96, 64, 48, 16, 2, 2>>;
    if (tokens <= 704) return launch_q8_add_mma<Q8A16MmaSchedule<48, 64, 64, 48, 16, 2, 3>>;
    if (tokens <= 784) return launch_q8_add_mma<Q8A16MmaSchedule<48, 112, 64, 48, 16, 2, 2>>;
    if (tokens <= 896) return launch_q8_add_mma<Q8A16MmaSchedule<48, 128, 64, 48, 16, 2, 2>>;
    if (tokens <= 960) return launch_q8_add_mma<Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>;
    if (tokens <= 1023) return launch_q8_add_mma<Q8A16MmaSchedule<64, 112, 64, 64, 16, 2, 2>>;
    if (tokens <= 1024) return launch_q8_add_mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
    if (tokens <= 1120) return launch_q8_add_mma<Q8A16MmaSchedule<64, 112, 64, 64, 16, 2, 2>>;
    if (tokens <= 1280) return launch_q8_add_mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
    if (tokens <= 1344) return launch_q8_add_mma<Q8A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>;
    if (tokens <= 1408) return launch_q8_add_mma<Q8A16MmaSchedule<48, 128, 64, 48, 16, 2, 2>>;
    if (tokens <= 1680) return launch_q8_add_mma<Q8A16MmaSchedule<128, 80, 64, 64, 16, 2, 2>>;
    if (tokens <= 1791) return launch_q8_add_mma<Q8A16MmaSchedule<48, 128, 64, 48, 16, 2, 2>>;
    if (tokens <= 1792) return launch_q8_add_mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
    if (tokens <= 1919) return launch_q8_add_mma<Q8A16MmaSchedule<48, 128, 64, 48, 16, 2, 2>>;
    if (tokens <= 1920) return launch_q8_add_mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
    if (tokens <= 2016) return launch_q8_add_mma<Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>;
    if (tokens <= 2047) return launch_q8_add_mma<Q8A16MmaSchedule<64, 112, 64, 64, 16, 2, 2>>;
    if (tokens <= 2048) return launch_q8_add_mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
    return launch_q8_add_mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}
} // namespace ninfer::ops::detail
