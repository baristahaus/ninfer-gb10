#include "ops/linear/projection/fp8/fp8_shapes.h"
#include "ops/linear/projection/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// The 27B's GDN z PARENT at tensor-parallel width 1: the z half of the fused 16384-row GDN input
// projection (48 value heads of width 128 = 6144 rows). Width 1 presents it as its own parent beside
// the 10240-row qkv parent (see n10240_k5120.cu); width 2 takes the four-role split and never
// presents either. This shape exists for the width-1 reference arm.
//
// The schedule set is the 5120x5120 one, and that is a derivation rather than a copy: 6144 rows is
// 1.2x 5120, so at every tile size this shape has at least the row-tile count its sibling has
// (192/96/48/24 against 160/80/40/20 at BlockRows 32/64/128/256) and every band below reaches at
// least the same CTA count. K is 5120, which the A8 activation quantizer already specializes.
using Geometry  = Fp8Geometry<6144, 5120>;
using Gemv      = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
// 6144/32 = 192 row tiles: one full wave from T=1 through T=32 without a split.
using T32R32    = Fp8A8MmaSchedule<32, 32, 128, 1, 2, 3, 3, Cache::cg, Cache::cg,
                                   Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
// 6144/128 = 48 row tiles over 32-token columns: 96 CTAs at T<=64 and 144 through T=96.
using T32R128   = Fp8A8MmaSchedule<32, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                   Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
// 48 row tiles of 128 rows over 64-token columns: 96 CTAs through T=128.
using Tma64x128 = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>;
// 24 row tiles of 256 rows: 72 CTAs at T=192 and through T=288 with the 96-token tile. Below a wave,
// which is why the large-tile path hands over to the split-K schedules above 288.
using Tma64x256 = Fp8A8TmaMmaSchedule<64, 256, 128, 2, 4, 2, 1>;
using Tma96x256 = Fp8A8TmaMmaSchedule<96, 256, 128, 3, 4, 2, 1>;
// 24 row tiles of 256 rows over 128-token columns: 48 CTAs at T=289..384, a thin tail that splits
// into a full wave; 72 at T=512, 96 at T=640, 168 at T=1024.
using Bulk      = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    // 6144/16 = 384 CTAs at the 16-row sliced-K tile: the hot interval is covered without a split.
    if (tokens <= 16)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 8, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 24)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 32)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 1>>(x, weight, out,
                                                                              stream);
    if (tokens <= 64)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(
            x, weight, out, stream);
    if (tokens <= 128)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>>(
            x, weight, out, stream);
    fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, out,
                                                                               stream);
}

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    if (x.ne[1] <= 32)
        return launch_fp8_a8<Geometry, T32R32>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 96)
        return launch_fp8_a8<Geometry, T32R128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 128)
        return launch_fp8_a8_tma<Geometry, Tma64x128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 192)
        return launch_fp8_a8_tma<Geometry, Tma64x256>(x, weight, out, scratch, stream);
    // Three 96-token tiles keep the 24 row tiles at 72 CTAs, the widest tile the shape reaches
    // before the split-K path pays for its reduction.
    if (x.ne[1] <= 288)
        return launch_fp8_a8_tma<Geometry, Tma96x256>(x, weight, out, scratch, stream);
    launch_fp8_a8_tma<Geometry, Bulk>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    // The boundary is the dispatch's: Bulk, the only split-K schedule in this set, runs from 289
    // tokens.
    return max_tokens > 288 ? Bulk::kPartialBytes : 0;
}

} // namespace

const Fp8LinearShape kFp8N6144K5120{6144,      5120,    launch_a16,
                                    launch_a8, uses_a8, partial_capacity_bytes};
} // namespace ninfer::ops::detail
