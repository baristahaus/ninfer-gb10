#include "ops/linear/projection/q8/shapes.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/kernels/q8/q8_instance_launch.cuh"

// The fused mlp gate|up parent's Rows-shard half: 34816 rows halved by the Rows shard at
// tensor-parallel width 2, which is the shape the MTP block's dense MLP presents there (the fused
// parent is 34816 because gate and up are ranges of one object). It sits between the role split's
// 8704 and the width-1 parent's 34816, and the parent is the sibling this ladder comes from.
//
// That is a derivation rather than a copy: every band is chosen from (n / kBlockRows) *
// ceil(T / BlockTokens) against the tuned wave, and this shape presents 1088 row tiles -- 6.4 waves at
// T=1 -- so no band needs a different CTA shape to fill the machine, and K is the parent's 5120, so
// each CTA walks the same ten 512-wide K tiles. What the parent's ladder contributes beyond the token
// bands is its larger-row-block MMA fallbacks (r128_t64 above 64 tokens), which suit a shape twice
// the role split's width better than that sibling's r32_t64.

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N17408K5120;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
using C4 =
    Q8A16SlicedKMmaSchedule<8, 4, 1, 2, Access::Direct, Cache::ca, Cache::cg, Stage::PaddedZero>;
using C8 =
    Q8A16SlicedKMmaSchedule<8, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C16 =
    Q8A16SlicedKMmaSchedule<16, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C24 =
    Q8A16SlicedKMmaSchedule<24, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C32 = Q8A16SlicedKMmaSchedule<32, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg,
                                    Stage::RuntimeActive>;
using C40 = Q8A16SlicedKMmaSchedule<40, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg,
                                    Stage::RuntimeActive>;
using C56 =
    Q8A16SlicedKMmaSchedule<56, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
} // namespace

Q8Launch select_q8_n17408_k5120(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_sliced<Geometry, 4, C4>;
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 16, C16>;
    if (tokens <= 24) return launch_q8_a16_sliced<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_a16_sliced<Geometry, 32, C32>;
    if (tokens <= 40) return launch_q8_a16_sliced<Geometry, 40, C40>;
    if (tokens <= 48) return launch_q8_a16_mma_r64x16_t48_k128_a1;
    if (tokens <= 56) return launch_q8_a16_sliced<Geometry, 56, C56>;
    if (tokens <= 64) return launch_q8_a16_mma_r128_t64;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
