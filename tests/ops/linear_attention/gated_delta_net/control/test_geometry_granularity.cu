// Host-only gate on the GDN control projection's head-count granularity.
//
// WHY THIS EXISTS: the two-GPU lane needs the control projection to serve 24 value heads (the 27B's
// 48 split by rows). Whether that is "add an instance", "use the merged row mode", or "pad the M
// tile" depends on one number — the MMA tile's M granularity — and on a straddle that the tile
// count alone does not reveal. This test pins both, so the next round starts from the measured
// constraint rather than from arithmetic that looks right.
//
// Compiled as a .cu because it reads the MMA tile constant out of the kernel header; it calls only
// host functions and runs without a device.

#include "ops/linear_attention/gated_delta_net/control/kernels/mma.cuh"
#include "ops/linear_attention/gated_delta_net/control/operands.h"
#include "ops/linear_attention/gated_delta_net/control/plan.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace {

using ninfer::ops::detail::gdn_control::Geometry;
using ninfer::ops::detail::gdn_control::Kernel;
using ninfer::ops::detail::gdn_control::NormKernel;
using ninfer::ops::detail::gdn_control::describe_profile;
using ninfer::ops::detail::gdn_control::kBlockM;
using ninfer::ops::detail::gdn_control::select_norm_plan;
using ninfer::ops::detail::gdn_control::select_projection_plan;
using ninfer::ops::LinearProfile;
using ninfer::ops::ProjectionProfile;

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

    bool refused = false;
    try {
        (void)describe_profile(paired(kRankHeads));
    } catch (const std::invalid_argument&) { refused = true; }
    std::printf("  control profile at 24 rows: %s\n", refused ? "REFUSED" : "accepted");
    check(refused, "a tp2 rank's 24-row control profile is still refused");
}

// ------------------------------------------------------------------ the M-tile straddle
// This is the finding, and the tile COUNT is not enough to see it. MmaGeometry asserts
// `Heads % kBlockM == 0` for BOTH row modes, because a 16-head tile boundary has to line up with
// the a/b boundary at kHeads. At 24 heads it does not: tile 1 starts at head 16, and the kernel
// picks `is_b` from head0 alone, so it reads a rows 16..31 when a has only 24 rows.
void check_tile_straddle() {
    check(kBlockM == 16, "the control MMA row tile is 16 heads");
    check(kSingleHeads % kBlockM == 0, "48 value heads divide the 16-head tile");

    // Merged: kHeadTiles = 2*kHeads/kBlockM. The count divides (48/16 = 3) — which is the
    // arithmetic that makes Merged look available — but the a/b boundary at head 24 falls inside
    // tile 1, so the assertion still has to reject the geometry.
    check((2 * kRankHeads) % kBlockM == 0, "2*24 rows DO divide into 16-row tiles");
    std::printf("  merged tile count 2*24/16 = %d (divides), but head 24 lands inside tile %d\n",
                (2 * kRankHeads) / kBlockM, kRankHeads / kBlockM);

    // The a/b boundary must be a tile boundary: that is what Heads % kBlockM == 0 guarantees and
    // why it is asserted for both row modes.
    check(kRankHeads % kBlockM != 0,
          "24 value heads put the a/b boundary inside a 16-head tile — Merged is rejected too");
    const int straddling_tile = kRankHeads / kBlockM;
    const int tile_first_head = straddling_tile * kBlockM;
    std::printf("  tile %d covers heads %d..%d; a holds only %d rows\n", straddling_tile,
                tile_first_head, tile_first_head + kBlockM - 1, kRankHeads);
    check(tile_first_head + kBlockM > kRankHeads,
          "that tile reads past the end of the rank's a block");
}

// ------------------------------------------------------------------ the 48-row path is intact
void check_single_device_routes() {
    const Geometry single{48, kHidden};
    bool all_routed = true;
    for (std::int32_t columns : {1, 2, 8, 9, 1024, 2048, 4096, 8192, 65536}) {
        try {
            (void)select_projection_plan(single, columns);
        } catch (const std::exception&) { all_routed = false; }
    }
    check(all_routed, "every token count still routes on the single-device geometry");
    check(select_projection_plan(single, 1).kernel == Kernel::Gemv, "T=1 still takes the GEMV route");
    const auto norm = select_norm_plan(single, 4);
    check(norm.kernel == NormKernel::Simt, "T<=42 on the single device still takes the fused norm");
    std::printf("  single-device T=4: projection kernel=%d, norm kernel=%d\n",
                static_cast<int>(select_projection_plan(single, 4).kernel),
                static_cast<int>(norm.kernel));
}

// ------------------------------------------------------------------ no half-built acceptance
void check_rank_geometry_has_no_route() {
    const Geometry rank{24, kHidden};
    bool refused = false;
    try {
        (void)select_projection_plan(rank, 1024);
    } catch (const std::exception&) { refused = true; }
    check(refused, "the 24-head geometry has no projection route yet, and says so");
}

} // namespace

int main() {
    std::printf("GDN control projection head granularity (host)\n");
    check_profile_gate();
    check_tile_straddle();
    check_single_device_routes();
    check_rank_geometry_has_no_route();
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures,
                failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
