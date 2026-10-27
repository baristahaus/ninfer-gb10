#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {
Q4Launch select_q4_n147456_k2560(std::int32_t tokens) {
    // Flash-Next optimized proposal head. Follows the measured 131072-row vocabulary selections,
    // with the generic sliced-K instance where those use a K-specific one; not yet timed on its
    // own shape.
    if (tokens <= 1) return launch_q4_a16_gemv_r4_w1_direct;
    if (tokens <= 8) return launch_q4_a16_sliced_r32_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 72) return launch_q4_a16_mma_r64_t72;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q4_a16_mma_r64_t112;
    if (tokens <= 120) return launch_q4_a16_mma_r64_t120;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
