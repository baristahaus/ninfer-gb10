// The 24-head control projection: one rank of the 27B's [48, 5120] a/b pair at tensor-parallel
// width 2. Same 5120 hidden width and the same 3:1 split of the output rows into g and beta as
// n96_k5120.cu, at half the heads.
//
// It serves the routes whose kernels are already parameterised on the head count — the MMA geometry
// pads its 16-head M tile so 24 heads fit in two tiles. The decode regime T=1..1024 rides the
// split-8 MMA instance alone: its grid masks partial token tiles, so a row's bits do not depend on
// T. The two routes that still carry a literal 48 (small_split.cuh's `kHeads` and norm.cuh's `H`)
// are deliberately NOT selected here — the plan sends every token count to the MMA route, and
// `select_norm_plan` sends this geometry to the materialized fallback, which normalizes with
// `ops::rmsnorm` and then calls back into this projection. That keeps this file free of a copy of
// either kernel; lifting those two constants is a tuning follow-up, not a correctness prerequisite.

#include "ops/linear_attention/gated_delta_net/control/kernels/mma_launch.cuh"

#include <array>
#include <limits>

namespace ninfer::ops::detail::gdn_control {
namespace {
constexpr int kLast = std::numeric_limits<std::int32_t>::max();

constexpr Plan mma(int tile, int split) { return {Kernel::Mma, tile, split}; }

// Every token count goes to the MMA route: the small-split route still carries a literal
// 48 in its kernels, and the MMA geometry's padded 16-head M tile serves 24 heads without it.
constexpr std::array kN48K5120{
    Route{1, 1024, mma(128, 8)},
    Route{1025, 2048, mma(128, 4)},
    Route{2049, 4096, mma(128, 2)},
    Route{4097, kLast, mma(128, 1)},
};
static_assert(closed(kN48K5120));
} // namespace

std::span<const Route> routes_n48_k5120() { return kN48K5120; }

void launch_n48_k5120(const Inputs& in, GdnControlOutputs out, Plan plan, Scratch scratch,
                      CudaExecutionView e) {
    // Padded: 24 value heads occupy two 16-head M tiles, the second half-masked.
    using Geometry = MmaGeometry<24, 5120, 128>;
    if (plan.kernel == Kernel::Mma) {
        bool launched = false;
        switch (plan.split_k) {
        case 8:
            launched = launch_mma<Geometry, 8, 8>(in, out, scratch, e);
            break;
        case 4:
            launched = launch_mma<Geometry, 4, 16>(in, out, scratch, e);
            break;
        case 2:
            launched = launch_mma<Geometry, 2, 16>(in, out, scratch, e);
            break;
        case 1:
            launched = launch_mma<Geometry, 1, 8>(in, out, {}, e);
            break;
        default:
            throw std::logic_error("gdn_control: invalid 48x5120 plan");
        }
        if (!launched) (void)launch_mma<Geometry, 1, 8>(in, out, {}, e);
        return;
    }
    throw std::logic_error("gdn_control: invalid 48x5120 plan");
}
} // namespace ninfer::ops::detail::gdn_control
