#include "ops/linear/projection/fp8/fp8_shapes.h"
#include "ops/linear/projection/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// The 27B's two output projections at tensor-parallel width 2: attention/output and gdn/output are
// Columns-sharded over 6144, so each rank presents 5120x3072 where width 1 presents 5120x6144.
// The schedule set is the 5120x6144 one, and the reason it transfers is that every tiling decision
// below is a function of (n, T) only: the CTA count of a TMA tile is
// (n / BlockRows) * ceil(T / BlockTokens), which does not mention K, and the split-K decision is
// taken from that count against the tuned wave. Halving K only shortens each CTA's K loop, from 48
// tiles to 24 at BlockK 128, and 24 still leaves a 4-way tail split six full K tiles per part, so
// the cap that gates split-K (kMaxParts <= K / BlockK) is not reached. What does NOT transfer is
// the stale wave arithmetic in the sibling files' comments; the counts here are recomputed.
using Geometry  = Fp8Geometry<5120, 3072>;
using Gemv      = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
// 5120/64 = 80 row tiles: 80 CTAs at T<=32 and 160 at T<=64, a full wave either side.
using Tma32x64  = Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>;
// 5120/128 = 40 row tiles: 40 * ceil(T/64) CTAs, 120 at T=192.
using Tma64x128 = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>;
// 40 row tiles of 128 rows over 128-token columns: 80 CTAs at T=256, 240 at T=768. Below the 170
// wave the tail is split rather than left as a thin wave.
using MidBulk   = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, 170, 4, 8>;
// 20 row tiles of 256 rows: 160 CTAs at T=1024 and 320 at T=2048, where the wider tile wins.
using Bulk      = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 16)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 8, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 32)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 4, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 64)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 1>>(x, weight, out,
                                                                              stream);
    if (tokens <= 128)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 64, 128, 32, 16, 2, 2>>(
            x, weight, out, stream);
    fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, out,
                                                                               stream);
}

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    if (x.ne[1] <= 64)
        return launch_fp8_a8_tma<Geometry, Tma32x64>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 128)
        return launch_fp8_a8<Geometry, Fp8A8T64R64K128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 192)
        return launch_fp8_a8_tma<Geometry, Tma64x128>(x, weight, out, scratch, stream);
    // The narrower row tile fills the GPU before the large-tile path reaches a full wave.
    if (x.ne[1] <= 768)
        return launch_fp8_a8_tma<Geometry, MidBulk>(x, weight, out, scratch, stream);
    launch_fp8_a8_tma<Geometry, Bulk>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    if (max_tokens > 768) return Bulk::kPartialBytes;
    return max_tokens > 192 ? MidBulk::kPartialBytes : 0;
}

} // namespace

const Fp8LinearShape kFp8N5120K3072{5120,      3072,    launch_a16,
                                    launch_a8, uses_a8, partial_capacity_bytes};
} // namespace ninfer::ops::detail
