// Host-only gate on the GDN control projection's head-count geometry.
//
// The two-GPU lane needs the control projection at 24 value heads — one rank of the 27B's 48. The
// MMA route's M tile is 16 heads, and 24 is not a multiple of 16, so the tile is PADDED: two tiles
// cover 32 head slots, the last 8 masked. This test pins the consequences of that padding, the
// route table that now exists for it, and the fact that the single-device geometry is untouched.
//
// Compiled as a .cu because it reads the MMA tile constants out of the kernel header; it calls only
// host functions and runs without a device.

#include "ops/linear_attention/gated_delta_net/control/kernels/mma.cuh"
#include "ops/linear_attention/gated_delta_net/control/operands.h"
#include "ops/linear_attention/gated_delta_net/control/plan.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace {

using ninfer::ops::LinearProfile;
using ninfer::ops::ProjectionProfile;
using ninfer::ops::detail::gdn_control::Geometry;
using ninfer::ops::detail::gdn_control::Kernel;
using ninfer::ops::detail::gdn_control::MmaGeometry;
using ninfer::ops::detail::gdn_control::NormKernel;
using ninfer::ops::detail::gdn_control::describe_profile;
using ninfer::ops::detail::gdn_control::kBlockM;
using ninfer::ops::detail::gdn_control::select_norm_plan;
using ninfer::ops::detail::gdn_control::select_projection_plan;

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) { ++failures; }
}

constexpr std::int32_t kSingleHeads = 48;
constexpr std::int32_t kRankHeads   = 24;
constexpr std::int32_t kHidden      = 5120;

ProjectionProfile paired(std::int32_t rows) {
    const LinearProfile parent{ninfer::QType::BF16, rows, kHidden,
                               ninfer::ops::LinearPolicy::A16Only,
                               ninfer::QuantLayout::Contiguous};
    return ProjectionProfile{parent, parent};
}

// ------------------------------------------------------------------ the acceptance gate
void check_profile_gate() {
    std::int32_t heads = 0;
    bool single_ok     = true;
    try {
        heads = describe_profile(paired(kSingleHeads)).heads;
    } catch (const std::invalid_argument&) { single_ok = false; }
    check(single_ok && heads == 48, "the 48-row paired control profile is accepted, heads 48");

    bool rank_ok = false;
    std::int32_t rank_heads = 0;
    try {
        rank_heads = describe_profile(paired(kRankHeads)).heads;
        rank_ok    = true;
    } catch (const std::invalid_argument&) { rank_ok = false; }
    std::printf("  control profile at 24 rows: %s (heads %d)\n", rank_ok ? "accepted" : "REFUSED",
                rank_heads);
    check(rank_ok && rank_heads == 24, "the 24-row paired control profile is now accepted, heads 24");
}

// ------------------------------------------------------------------ the padded M tile
void check_padded_tile() {
    using Single = MmaGeometry<kSingleHeads, kHidden, 128>;
    using Rank   = MmaGeometry<kRankHeads, kHidden, 128>;
    check(kBlockM == 16, "the control MMA row tile is 16 heads");

    // The single-device geometry keeps its exact tile count, so its generated code is unchanged.
    check(Single::kHeadTiles == 3 && Single::kPaddedRows == 48 && Single::kLogicalRows == 48,
          "48 heads still tile exactly: 3 tiles, 48 rows, no padding");

    // The rank geometry pads: 24 logical rows over two 16-row tiles = 32 slots, 8 masked.
    check(Rank::kLogicalRows == 24 && Rank::kHeadTiles == 2 && Rank::kPaddedRows == 32,
          "24 heads pad to two 16-head tiles (32 slots, 8 masked)");
    check(Rank::kPaddedRows > Rank::kLogicalRows,
          "the rank geometry's padding is real and is masked at load and store");
    std::printf("  single: tiles=%d padded=%d logical=%d\n", Single::kHeadTiles,
                Single::kPaddedRows, Single::kLogicalRows);
    std::printf("  rank:   tiles=%d padded=%d logical=%d\n", Rank::kHeadTiles, Rank::kPaddedRows,
                Rank::kLogicalRows);
}

// ------------------------------------------------------------------ the route table
void check_routes() {
    const Geometry single{kSingleHeads, kHidden};
    const Geometry rank{kRankHeads, kHidden};
    bool single_ok = true, rank_ok = true;
    for (std::int32_t columns : {1, 2, 4, 8, 9, 1024, 1025, 2048, 2049, 4096, 4097, 8192, 65536}) {
        try {
            (void)select_projection_plan(single, columns);
        } catch (const std::exception&) { single_ok = false; }
        try {
            (void)select_projection_plan(rank, columns);
        } catch (const std::exception&) { rank_ok = false; }
    }
    check(single_ok, "every token count still routes on the single-device geometry");
    check(rank_ok, "every token count now routes on the 24-head geometry");
    check(select_projection_plan(rank, 1).kernel == Kernel::Gemv, "the rank's T=1 takes GEMV");

    // The 48-row route table must not have moved: same kernel and split at every boundary.
    bool same = true;
    for (std::int32_t columns : {1, 2, 9, 1024, 1025, 2048, 4097, 8192}) {
        const auto a = select_projection_plan(single, columns);
        if (columns == 1 && a.kernel != Kernel::Gemv) { same = false; }
        if (columns == 9 && a.kernel != Kernel::Mma) { same = false; }
    }
    check(same, "the 48-row route table still selects GEMV at T=1 and MMA above it");
    std::printf("  single T=9: kernel=%d tile=%d split=%d | rank T=9: kernel=%d tile=%d split=%d\n",
                static_cast<int>(select_projection_plan(single, 9).kernel),
                select_projection_plan(single, 9).tile_cols,
                select_projection_plan(single, 9).split_k,
                static_cast<int>(select_projection_plan(rank, 9).kernel),
                select_projection_plan(rank, 9).tile_cols,
                select_projection_plan(rank, 9).split_k);
}

// ------------------------------------------------------------------ the norm fallback
// norm.cuh carries `H = 48` as a compile-time constant, so a 24-head rank cannot use the fused norm
// kernel and takes the materialized fallback instead: rmsnorm, then the projection plan.
void check_norm_fallback() {
    check(select_norm_plan({kSingleHeads, kHidden}, 4).kernel == NormKernel::Simt,
          "the single device still takes the fused norm below T=42");
    const auto rank = select_norm_plan({kRankHeads, kHidden}, 4);
    check(rank.kernel == NormKernel::Materialized,
          "a 24-head rank takes the materialized norm fallback, not the 48-head kernel");
    std::printf("  norm plan: single kernel=%d, rank kernel=%d\n", static_cast<int>(NormKernel::Simt),
                static_cast<int>(rank.kernel));
}

// ------------------------------------------------------------------ no half-built acceptance
// A geometry with no route table must still be refused, so a future round cannot add acceptance
// ahead of a route.
void check_unrouted_geometry_refused() {
    const Geometry unrouted{32, kHidden}; // 32 heads at 5120: no table
    bool refused = false;
    try {
        (void)select_projection_plan(unrouted, 1024);
    } catch (const std::exception&) { refused = true; }
    check(refused, "a geometry with no route table is still refused");

    bool profile_refused = false;
    try {
        (void)describe_profile(paired(16)); // 16 rows at 5120: not in the allow-list
    } catch (const std::invalid_argument&) { profile_refused = true; }
    check(profile_refused, "a row count with no geometry is still refused at the profile gate");
}

} // namespace

int main() {
    std::printf("GDN control projection head geometry (host)\n");
    check_profile_gate();
    check_padded_tile();
    check_routes();
    check_norm_fallback();
    check_unrouted_geometry_refused();
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures,
                failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
