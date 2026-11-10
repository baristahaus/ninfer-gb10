// Host-only gate on the GDN control projection's head-count granularity.
//
// WHY THIS EXISTS: the two-GPU lane needs the control projection to serve 24 value heads (the 27B's
// 48 split by rows). Whether that is "add an instance" or "restructure the kernel" depends on a
// single number — the MMA tile's M granularity — and on how many of the five routes are actually
// parameterised on the head count. This test pins both, so the next round starts from the measured
// constraint rather than from a guess.
//
// It is compiled as a .cu because it reads kBlockM out of the MMA kernel header; it calls only host
// functions and runs without a device.

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
using ninfer::ops::ProjectionProfile;
using ninfer::ops::LinearProfile;

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) { ++failures; }
}

// The 27B's control projections: one row per value head. Single device 48 rows, tp2 rank 24.
constexpr std::int32_t kSingleRows = 48;
constexpr std::int32_t kRankRows   = 24;
constexpr std::int32_t kHidden     = 5120;

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
        heads = describe_profile(paired(kSingleRows)).heads;
    } catch (const std::invalid_argument&) { single_ok = false; }
    check(single_ok && heads == 48, "the 48-row paired control profile is accepted, heads 48");

    bool rank_refused = false;
    try {
        (void)describe_profile(paired(kRankRows));
    } catch (const std::invalid_argument&) { rank_refused = true; }
    std::printf("  control profile at 24 rows: %s\n",
                rank_refused ? "REFUSED" : "accepted");
    check(rank_refused, "a tp2 rank's 24-row control profile is still refused");
}

// ------------------------------------------------------------------ the MMA M granularity
// This is the finding: the MMA route's row tile is 16 heads and the kernel asserts the head count
// divides it. 48 does; 24 does not. The kernel also computes kHeadTiles by integer division, so
// without the assertion a 24-head geometry would silently cover only 16 of the 24 heads.
void check_mma_tile_granularity() {
    check(kBlockM == 16, "the control MMA row tile is 16 heads");
    check(kSingleRows % kBlockM == 0, "48 value heads divide the 16-head tile");
    std::printf("  24 %% %d = %d — does not divide\n", kBlockM, kRankRows % kBlockM);
    check(kRankRows % kBlockM != 0,
          "24 value heads do NOT divide the tile, so MmaGeometry<24,...> cannot instantiate");
    // The kernel tiles (Paired ? Heads : 2*Heads) rows. A MERGED (fused a|b) weight would tile 48
    // rows and divide cleanly — but the 27B stores a_projection and b_projection as separate
    // parents, so the Paired mode is the one that has to work, and it cannot at 24.
    check((2 * kRankRows) % kBlockM == 0,
          "a merged 2*24-row weight WOULD divide the tile — the blocker is the paired packing");
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

    const auto at_one = select_projection_plan(single, 1);
    check(at_one.kernel == Kernel::Gemv, "T=1 still takes the GEMV route");
    // The GEMV kernel is templated on Heads, so it is the one route a 24-head geometry could take
    // today — but only at T=1.
    const auto small = select_projection_plan(single, 4);
    std::printf("  T=4 single-device route: kernel=%d tile=%d split=%d\n",
                static_cast<int>(small.kernel), small.tile_cols, small.split_k);
}

void check_norm_route_is_single_device_only() {
    const Geometry single{48, kHidden};
    const auto norm = select_norm_plan(single, 4);
    check(norm.kernel == NormKernel::Simt, "T<=42 on the single device still takes the fused norm");
    std::printf("  fused norm plan: kernel=%d tile=%d threads=%d\n", static_cast<int>(norm.kernel),
                norm.tile_cols, norm.threads);
}

// ------------------------------------------------------------------ no half-built acceptance
// There is deliberately no route table for 24 heads yet, so the plan selector refuses rather than
// returning a plan that would fail in a launcher. If a future round adds acceptance without a
// working route, this is the test that catches it.
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
    check_mma_tile_granularity();
    check_single_device_routes();
    check_norm_route_is_single_device_only();
    check_rank_geometry_has_no_route();
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures,
                failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
