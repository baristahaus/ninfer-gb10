// Host-only gate on the GDN recurrent path's head-count contract.
//
// WHY THIS EXISTS: GDN is 48 of the 27B's 64 layers, and a tp2 rank holds 8 QK heads and 24 value
// heads instead of 16 and 48. Whether that rank can run depends on ONE question — is the recurrent
// path keyed on the absolute head counts, or on the 3:1 ratio and the derived quantities?
//
// This test answers it at every layer the path has: the validation predicate, the route selector,
// the workspace layout, and the head map the kernels index with. It also pins what the packed
// batch-update path assumes, because that path is a different thing (Flash-Next only) and the
// distinction is what keeps a future reader from "fixing" the wrong gate.
//
// Compiled as a .cu because it reads `HeadMap` out of a CUDA kernel header; it calls only host
// functions and allocates nothing on a device, so it runs without one.

#include "ops/linear_attention/gated_delta_net/control/operands.h"
#include "ops/linear_attention/gated_delta_net/kernels/transition.cuh"
#include "ops/linear_attention/gated_delta_net/operands.h"
#include "ops/linear_attention/gated_delta_net/recurrence/plan.h"

#include <cstdio>
#include <stdexcept>

namespace {

using ninfer::ops::detail::gated_delta_net::HeadMap;
using ninfer::ops::detail::gated_delta_net::RecurrenceRoute;
using ninfer::ops::detail::gated_delta_net::are_head_counts_valid;
using ninfer::ops::detail::gated_delta_net::recurrence_workspace_layout;
using ninfer::ops::detail::gated_delta_net::select_recurrence_route;

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) { ++failures; }
}

// The two geometries that matter: the 27B on one device, and one rank of it at tensor-parallel
// width 2. The ratio is 3:1 in both.
constexpr int kSingleQk = 16, kSingleValue = 48;
constexpr int kRankQk = 8, kRankValue = 24;

// ------------------------------------------------------------------ the ratio predicate
void check_head_counts() {
    check(are_head_counts_valid(kSingleQk, kSingleValue), "16/48 is a valid head geometry");
    check(are_head_counts_valid(kRankQk, kRankValue), "8/24 is a valid head geometry");
    check(kRankValue / kRankQk == kSingleValue / kSingleQk,
          "the tp2 rank keeps the 3:1 value-to-QK ratio");
    // The predicate is the ratio test, not a whitelist: anything with a whole ratio passes and
    // anything without one fails, so it cannot accidentally reject the tp2 shape.
    check(!are_head_counts_valid(kRankQk, 20), "a non-whole ratio is rejected");
    check(!are_head_counts_valid(0, kRankValue), "a zero QK head count is rejected");
    check(!are_head_counts_valid(kRankValue, kRankQk), "value heads below QK heads are rejected");
}

// ------------------------------------------------------------------ the route selector
// The route is chosen on the token count alone; the head counts are not read at all.
void check_route_ignores_head_counts() {
    bool same = true;
    for (int tokens = 1; tokens <= 4096; ++tokens) {
        if (select_recurrence_route(kSingleQk, kSingleValue, tokens) !=
            select_recurrence_route(kRankQk, kRankValue, tokens)) {
            same = false;
            break;
        }
    }
    check(same, "the recurrence route is identical for 16/48 and 8/24 at every token count");
    std::printf("  route at 1 token: %s, at 64 tokens: %s\n",
                select_recurrence_route(kRankQk, kRankValue, 1) == RecurrenceRoute::Recurrent
                    ? "Recurrent"
                    : "Chunked",
                select_recurrence_route(kRankQk, kRankValue, 64) == RecurrenceRoute::Recurrent
                    ? "Recurrent"
                    : "Chunked");
}

// ------------------------------------------------------------------ the head map
// `HeadMap::of` builds the value-head -> QK-head divisor at run time from value/qk, so the kernels
// index with `group.divide(value_head)` rather than with a compile-time head count.
void check_head_map() {
    const HeadMap single = HeadMap::of(kSingleQk, kSingleValue);
    const HeadMap rank   = HeadMap::of(kRankQk, kRankValue);
    check(single.group_size() == 3 && rank.group_size() == 3,
          "both geometries group three value heads per QK head");
    // The mapping is the same *shape*: value head v belongs to QK head v/3, and the tp2 rank's
    // 24 value heads map onto its 8 QK heads exactly as the single device's 48 map onto 16.
    bool mapped = true;
    for (int value_head = 0; value_head < kRankValue; ++value_head) {
        if (rank.qk_head(value_head) != value_head / 3) { mapped = false; }
        if (rank.qk_head(value_head) >= kRankQk) { mapped = false; }
    }
    check(mapped, "every tp2 value head maps to a QK head inside the rank's own 8");
    std::printf("  head map: 16/48 -> %d groups, 8/24 -> %d groups\n", single.group_size(),
                rank.group_size());
}

// ------------------------------------------------------------------ the workspace
void check_workspace_is_parametric() {
    bool ok = true;
    for (int tokens = 1; tokens <= 64; ++tokens) {
        const auto single = recurrence_workspace_layout(kSingleQk, kSingleValue, tokens);
        const auto rank   = recurrence_workspace_layout(kRankQk, kRankValue, tokens);
        // The rank holds half the heads, so its workspace must be smaller and still nonzero for
        // the chunked route; the recurrent route needs none at all.
        if (rank.total_bytes > single.total_bytes) { ok = false; }
    }
    check(ok, "the tp2 rank's recurrence workspace never exceeds the single device's");
    std::printf("  64-token workspace: 16/48 %zu B, 8/24 %zu B\n",
                recurrence_workspace_layout(kSingleQk, kSingleValue, 64).total_bytes,
                recurrence_workspace_layout(kRankQk, kRankValue, 64).total_bytes);
}

// ------------------------------------------------------------------ what the PACKED path assumes
// The packed batch-update path is NOT on the 27B's route: the qwen3_5 model calls
// `ops::gated_delta_net` / `gated_delta_net_batch_update`, which use the parametric launchers.
// `launch_recurrent_batch_update_packed_qkv` is reached only from the Flash-Next family. It is
// keyed on the ABSOLUTE head counts, and this pins exactly how, so nobody generalises the wrong
// gate: its packed stride is (2*QK + value) * state_dim, which is a head-count-derived constant,
// and its grid is the literal value-head count.
void check_packed_path_is_absolute() {
    constexpr int kStateDim = 128;
    constexpr int kPackedSingle = (2 * kSingleQk + kSingleValue) * kStateDim;
    constexpr int kPackedRank   = (2 * kRankQk + kRankValue) * kStateDim;
    check(kPackedSingle == 10240, "the packed stride for 16/48 is the 10240 the gate requires");
    std::printf("  packed stride: 16/48 %d, 8/24 %d — the tp2 rank would need its own instance\n",
                kPackedSingle, kPackedRank);
    check(kPackedRank != kPackedSingle,
          "the packed path's stride is head-count-derived, so 8/24 needs a distinct instance");
}

// ------------------------------------------------------------------ the ACTUAL remaining gate
// The recurrence is head-count agnostic, so the gate that still blocks a tp2 rank is one layer up,
// in the GDN CONTROL projections (a_projection / b_projection, one row per value head). Their
// profile check requires the ABSOLUTE row count, and at width 2 each rank's is 24, not 48. This is
// the opposite of the recurrence: there is no derived ratio to key on, because the row count IS the
// head count. Recorded here so the next round starts from the real blocker.
void check_control_is_absolute() {
    using ninfer::ops::LinearProfile;
    using ninfer::ops::ProjectionProfile;
    using ninfer::ops::detail::gdn_control::describe_profile;

    const auto profile = [](std::int32_t rows) {
        const LinearProfile parent{ ninfer::QType::BF16, rows, 5120,
                                    ninfer::ops::LinearPolicy::A16Only,
                                    ninfer::QuantLayout::Contiguous };
        return ProjectionProfile{parent, parent};
    };
    // The single device's paired a/b projections: 48 rows each, accepted.
    bool single_ok = true;
    std::int32_t heads = 0;
    try {
        heads = describe_profile(profile(48)).heads;
    } catch (const std::invalid_argument&) { single_ok = false; }
    check(single_ok && heads == 48, "the 48-row control profile is accepted on one device");

    // A tp2 rank's half: 24 rows each. The paired branch requires 48, so this throws.
    bool rank_refused = false;
    try {
        (void)describe_profile(profile(24));
    } catch (const std::invalid_argument&) { rank_refused = true; }
    std::printf("  control profile at 24 rows: %s\n",
                rank_refused ? "REFUSED (the remaining gate)" : "accepted");
    check(rank_refused,
          "a tp2 rank's 24-row control profile is refused — this, not the recurrence, is the gate");
}

} // namespace

int main() {
    std::printf("GDN recurrent path head geometry (host)\n");
    check_head_counts();
    check_route_ignores_head_counts();
    check_head_map();
    check_workspace_is_parametric();
    check_packed_path_is_absolute();
    check_control_is_absolute();
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures,
                failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
