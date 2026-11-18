#include "ops/linear/projection/q5/q5_shapes.h"

namespace ninfer::ops::detail {
Q5Launch select_q5_n17408_k5120(std::int32_t tokens) {
    // The groupwise recipe's MLP up rows (Q5, one parent per role). The ladder mirrors
    // n7168_k5120, the nearest same-K sibling; only the row count differs.
    if (tokens <= 8) return launch_q5_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 24) return launch_q5_a16_sliced_r16_t24_w4_s2;
    if (tokens <= 32) return launch_q5_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 96) return launch_q5_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 128) return launch_q5_a16_mma_r32_t128;
    if (tokens <= 192) return launch_q5_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 384) return launch_q5_a16_mma_r64_t128;
    if (tokens <= 576) return launch_q5_a16_mma_r64_t96_k128_s1_a1;
    return launch_q5_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
