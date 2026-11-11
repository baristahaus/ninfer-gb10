#include "ops/linear/projection/fp8/fp8_shapes.h"
#include "ops/linear/projection/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// The 27B's attention key and value ROLES at tensor-parallel width 1: 4 KV heads of width 256, so
// 1024 rows of a 5120-wide activation. The artifact binds them as the key and value parts of the one
// 14336-row attention input projection, and width 1 presents them as their own weights (the packed
// form still hands the model one weight per role), which is why the reference arm needed this shape.
// The query and gate roles are 6144 rows, served by the GDN z parent's shape N6144K5120, because the
// FP8 table is keyed by (n, k) and not by role. Width 2 presents the halved 512-row N512K5120 and
// never reaches this one.
//
// This is the narrowest shape in the table after N512K5120, and its schedule set is derived from
// that: 1024 rows is too few to fill a 170-CTA wave at any tile wider than 32 rows, so the A8 path
// stays on the 32-row tile and lets the TOKEN dimension supply the grid — at BlockRows 32 the shape
// has 32 row tiles, so one tile covers 17..288 tokens as 32..288 CTAs, where BlockRows 128 would
// give 8 row tiles and 24 CTAs over the same range. Above 288 the split-K path takes over, because
// the row dimension can no longer grow. K is 5120, which the A8 activation quantizer specializes.
using Geometry = Fp8Geometry<1024, 5120>;
using Gemv     = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
// 32 row tiles of 32 rows: 32 CTAs at T<=32, 96 through T=96, 288 at T=288.
using T32R32    = Fp8A8MmaSchedule<32, 32, 128, 1, 2, 3, 3, Cache::cg, Cache::cg,
                                   Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
// 8 row tiles of 128 rows over 128-token columns: 24 CTAs at T=289..384, a thin tail that splits into
// 96; 32 at T=385..512 splits into 128; 64 at T=513..768; 128 at T=1024.
using MidBulk   = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>, 170, 4, 8>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    // 1024/16 = 64 CTAs at the 16-row sliced-K tile; eight K-warps each, so the narrow shape still
    // reaches three warps per SM across the hot interval instead of one and a half.
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
    // One band, because for this shape the row tile cannot get any wider without emptying the grid:
    // 32 row tiles of 32 rows is the only tiling that reaches a full wave (288 CTAs at 288 tokens)
    // before the split-K path becomes the better trade.
    if (x.ne[1] <= 288)
        return launch_fp8_a8<Geometry, T32R32>(x, weight, out, scratch, stream);
    launch_fp8_a8_tma<Geometry, MidBulk>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    // The boundary is the dispatch's: MidBulk, the only split-K schedule in this set, runs from 289
    // tokens.
    return max_tokens > 288 ? MidBulk::kPartialBytes : 0;
}

} // namespace

const Fp8LinearShape kFp8N1024K5120{1024,      5120,    launch_a16,
                                    launch_a8, uses_a8, partial_capacity_bytes};
} // namespace ninfer::ops::detail
