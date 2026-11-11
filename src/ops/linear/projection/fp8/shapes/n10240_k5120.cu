#include "ops/linear/projection/fp8/fp8_shapes.h"
#include "ops/linear/projection/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// The 27B's GDN qkv PARENT at tensor-parallel width 1. The artifact binds the whole GDN input
// projection as one FP8 object of 16384 rows — query 2048 (16 key heads of width 128), key 2048,
// value 6144 (48 value heads) and z 6144 — and width 1 presents it as the two contiguous parents the
// model's paired form consumes: qkv = 2048 + 2048 + 6144 = 10240 rows, and z = 6144 rows (see
// n6144_k5120.cu). Width 2 never reaches either: it takes the four-role split and presents
// 1024/1024/3072/3072 instead, so these two shapes exist for the width-1 reference arm.
//
// The schedule set is derived from this shape's own row-tile counts, not copied from a parent: at
// BlockRows 32/64/128/256 the shape has 320/160/80/40 row tiles, so every band below reaches at
// least one 170-CTA wave in the hot interval, and the split-K bands start where the large-tile paths
// stop filling it. K is 5120, which the A8 activation quantizer already specializes.
using Geometry  = Fp8Geometry<10240, 5120>;
using Gemv      = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
// 10240/32 = 320 row tiles: one full wave from T=1 through T=32 without any split.
using T32R32    = Fp8A8MmaSchedule<32, 32, 128, 1, 2, 3, 3, Cache::cg, Cache::cg,
                                   Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
// 10240/128 = 80 row tiles over 32-token columns: 160 CTAs at T<=64, 240 through T=96.
using T32R128   = Fp8A8MmaSchedule<32, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                   Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
// 80 row tiles of 128 rows over 64-token columns: 160 CTAs through T=128.
using Tma64x128 = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>;
// 40 row tiles of 256 rows: 120 CTAs at T=192, where the wider tile keeps the K loop long enough to
// pay for itself.
using Tma64x256 = Fp8A8TmaMmaSchedule<64, 256, 128, 2, 4, 2, 1>;
// 80 row tiles of 128 rows over 128-token columns: 160 CTAs at T=256 and 240 at T=384, both at or
// over one wave, so the narrow-tile path needs no split.
using MidBulk   = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, 170, 4, 8>;
// 40 row tiles of 256 rows: 160 CTAs at T=512 and 320 at T=1024; below a wave the tail splits.
using Bulk      = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    // 10240/16 = 640 CTAs at the 16-row sliced-K tile, so the whole hot interval is covered without
    // a split.
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
    if (x.ne[1] <= 384)
        return launch_fp8_a8_tma<Geometry, MidBulk>(x, weight, out, scratch, stream);
    launch_fp8_a8_tma<Geometry, Bulk>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    // The boundary is the dispatch's, not a convenience: MidBulk is a split-K schedule and runs from
    // 193 tokens, Bulk from 385. A capacity below the selected schedule's partial refuses the launch
    // it was planned for.
    if (max_tokens > 384) return Bulk::kPartialBytes;
    return max_tokens > 192 ? MidBulk::kPartialBytes : 0;
}

} // namespace

const Fp8LinearShape kFp8N10240K5120{10240,     5120,    launch_a16,
                                     launch_a8, uses_a8, partial_capacity_bytes};
} // namespace ninfer::ops::detail
