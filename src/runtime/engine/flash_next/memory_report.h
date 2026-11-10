#pragma once

// The Flash-Next route has no executor rank workers, so its startup record and snapshot would
// describe no allocation at all. These map the Program's MemorySummary onto the rank-scoped report
// types the generic route fills from its rank workers, so `server_start.resources` and
// `snapshot.ranks` describe the resident device on both routes.
//
// The mapping is deliberately literal: each field comes from the summary field that names the same
// quantity. A quantity the summary does not describe stays zero rather than being estimated, so the
// record never invents an allocation.

#include "ninfer/observability/resources.h"
#include "ninfer/types.h"

namespace ninfer::flashnext_runtime {

[[nodiscard]] inline ninfer::RankMemoryUsage rank_memory_usage(
    const MemorySummary& summary) noexcept {
    ninfer::RankMemoryUsage usage;
    // Resident weights, the state-slot arena and the unified workspace are the three device arenas
    // the Flash-Next startup prints.
    usage.allocations.device_weights_bytes    = summary.weights.capacity_bytes;
    usage.allocations.persistent_device_bytes = summary.sequence.capacity_bytes;
    usage.allocations.workspace_device_bytes  = summary.workspace.capacity_bytes;
    // The Host context tier is pinned Host KV; this route has no separate host weight bank.
    usage.allocations.host_context_bytes = summary.host_kv_capacity_bytes;
    usage.workspace.allocation_cursor_bytes = summary.workspace.used_bytes;
    usage.workspace.required_peak_bytes     = summary.workspace_logical_peak_bytes;
    usage.workspace.measurement_peak_bytes  = summary.workspace.peak_used_bytes;
    usage.kv_payload_bytes                  = summary.kv_payload_bytes;
    return usage;
}

[[nodiscard]] inline ninfer::RankMemoryBudget rank_memory_budget(
    const MemorySummary& summary) noexcept {
    ninfer::RankMemoryBudget budget;
    budget.runtime_reservation_bytes         = summary.runtime_reservation_bytes;
    budget.minimum_runtime_reservation_bytes = summary.minimum_runtime_reservation_bytes;
    budget.bytes_per_page_group              = summary.kv_capacity_increment_bytes;
    budget.available_after_weights_bytes     = summary.available_after_weights_bytes;
    budget.available_after_startup_bytes     = summary.available_after_startup_bytes;
    budget.headroom_bytes                    = summary.kv_capacity_headroom_bytes;
    budget.planned_slack_bytes               = summary.planned_slack_bytes;
    return budget;
}

} // namespace ninfer::flashnext_runtime
