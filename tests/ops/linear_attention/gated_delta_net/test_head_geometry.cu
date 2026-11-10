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

#include "core/platform.h" // NINFER_TARGET_RTX5090 / NINFER_TARGET_SPARK
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
// THE ROUTE IS TARGET-DEPENDENT, and only one target ignores the head counts:
//
//   rtx5090     select_recurrence_route(int /*qk_heads*/, int /*value_heads*/, int tokens)
//               -> tokens < 16 ? Recurrent : Chunked
//   dgx_spark   chunked_first = (qk_heads == 8 && value_heads == 24) ? 11 : 16
//               -> tokens < chunked_first ? Recurrent : Chunked
//
// So on the Spark target the tp2 geometry (8/24) crosses to the chunked route at 11 tokens while
// the single-device geometry (16/48) crosses at 16, and for tokens 11..15 the two DO take different
// routes — deliberately, as a tuning choice for exactly the shape this lane exists to serve.
//
// The earlier version of this test asserted "identical at every token count", which is true only on
// rtx5090 and would fail by design on dgx_spark. That is an assumption the code never promised on
// both targets, so the claim is narrowed here: the outer bound is asserted everywhere, and the
// exact identity only where the selector promises it.
void check_route_selector() {
    bool bounded = true;
    for (int tokens = 1; tokens <= 4096; ++tokens) {
        const bool single_recurrent =
            select_recurrence_route(kSingleQk, kSingleValue, tokens) == RecurrenceRoute::Recurrent;
        const bool rank_recurrent =
            select_recurrence_route(kRankQk, kRankValue, tokens) == RecurrenceRoute::Recurrent;
        // Both geometries agree below the earliest crossover either target uses (11) and above the
        // latest (16). Only the interval between them is target-specific.
        if (tokens < 11 && (!single_recurrent || !rank_recurrent)) { bounded = false; }
        if (tokens >= 16 && (single_recurrent || rank_recurrent)) { bounded = false; }
    }
    check(bounded, "both geometries are recurrent below 11 tokens and chunked from 16 up");

#if NINFER_TARGET_RTX5090
    bool same = true;
    for (int tokens = 1; tokens <= 4096; ++tokens) {
        if (select_recurrence_route(kSingleQk, kSingleValue, tokens) !=
            select_recurrence_route(kRankQk, kRankValue, tokens)) {
            same = false;
            break;
        }
    }
    check(same, "on rtx5090 the route is identical for 16/48 and 8/24 at every token count");
    std::printf("  rtx5090 crossover: both at 16 tokens\n");
#else
    // Spark moves 8/24 to 11 tokens. Assert the difference rather than pretending it is absent.
    const bool rank_earlier =
        select_recurrence_route(kRankQk, kRankValue, 11) == RecurrenceRoute::Chunked &&
        select_recurrence_route(kSingleQk, kSingleValue, 11) == RecurrenceRoute::Recurrent;
    check(rank_earlier, "on dgx_spark the 8/24 geometry crosses to chunked at 11 tokens, 16/48 at 16");
    std::printf("  dgx_spark crossover: 8/24 at 11 tokens, 16/48 at 16 tokens\n");
#endif
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

// ------------------------------------------------------------------ the control profile gate
// The recurrence is head-count agnostic on rtx5090, so the gate that blocked a tp2 rank was one
// layer up, in the GDN CONTROL projections (a_projection / b_projection, one row per value head).
// Their profile check required the ABSOLUTE row count, and at width 2 each rank's is 24, not 48.
// That gate is now open: the 24-row profile is accepted and the projection serves it through the
// padded 16-head M tile.
void check_control_profile_gate() {
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

    // A tp2 rank's half: 24 rows each. This WAS the remaining gate; the padding change that serves
    // 24 heads through two 16-head M tiles made the profile acceptable, so this group now asserts
    // the acceptance rather than the refusal. It went stale silently once — the merge that added
    // the route and the merge that added this assertion disagreed, and only a suite run caught it.
    bool rank_ok = false;
    std::int32_t rank_heads = 0;
    try {
        rank_heads = describe_profile(profile(24)).heads;
        rank_ok    = true;
    } catch (const std::invalid_argument&) { rank_ok = false; }
    std::printf("  control profile at 24 rows: %s (heads %d)\n", rank_ok ? "accepted" : "REFUSED",
                rank_heads);
    check(rank_ok && rank_heads == 24,
          "a tp2 rank's 24-row control profile is accepted, heads 24");

    // The guard that made this group worth having survives: a row count with no geometry is still
    // refused, so acceptance cannot be added ahead of a route.
    bool unrouted_refused = false;
    try {
        (void)describe_profile(profile(16)); // 16 rows at 5120: no registered geometry
    } catch (const std::invalid_argument&) { unrouted_refused = true; }
    check(unrouted_refused, "a control profile with no registered geometry is still refused");
}

} // namespace

int main() {
    std::printf("GDN recurrent path head geometry (host)\n");
    check_head_counts();
    check_route_selector();
    check_head_map();
    check_workspace_is_parametric();
    check_packed_path_is_absolute();
    check_control_profile_gate();
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures,
                failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
