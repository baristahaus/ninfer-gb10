#include "ops/linear/projection/q8/shapes.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/kernels/q8/q8_instance_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// the MTP block's mlp down projection at width 2: the 5120x17408 Columns shard, which halves K.
//
// The ladder is N5120K6144's, and that is a derivation rather than a copy: every band in it is
// chosen from (n / kBlockRows) * ceil(T / BlockTokens) against the tuned wave, and that expression
// mentions neither K nor the row count, only the fixed 16-row CTA tile this schedule family uses
// (kBlockRows = 16). This shape has 320 row tiles, so T=1 already presents 320 CTAs
// -- 1.9 waves -- and no band needs a different CTA shape to fill the machine; K only sets
// how many 17-wide K iterations each CTA runs (17 here). The MTP block's hot
// band is T<=4 (draft_tokens 3 gives a four-token verify block), which the first band covers.
using Geometry = Q8N5120K8704;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
using C4 =
    Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C8 =
    Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C16 =
    Q8A16SlicedKMmaSchedule<16, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>;
using C24 =
    Q8A16SlicedKMmaSchedule<24, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>;
using C32 = Q8A16SlicedKMmaSchedule<32, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg,
                                    Stage::RuntimeActive>;
using C40 =
    Q8A16SlicedKMmaSchedule<40, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>;
using C48 =
    Q8A16SlicedKMmaSchedule<48, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>;
using C56 =
    Q8A16SlicedKMmaSchedule<56, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>;
using C64 =
    Q8A16SlicedKMmaSchedule<64, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::PaddedZero>;
} // namespace

Q8Launch select_q8_n5120_k8704(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_sliced<Geometry, 4, C4>;
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 16, C16>;
    if (tokens <= 24) return launch_q8_a16_sliced<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_a16_sliced<Geometry, 32, C32>;
    if (tokens <= 40) return launch_q8_a16_sliced<Geometry, 40, C40>;
    if (tokens <= 48) return launch_q8_a16_sliced<Geometry, 48, C48>;
    if (tokens <= 56) return launch_q8_a16_sliced<Geometry, 56, C56>;
    if (tokens <= 64) return launch_q8_a16_sliced<Geometry, 64, C64>;
    if (tokens <= 128) return launch_q8_a16_mma_r32_t128;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
