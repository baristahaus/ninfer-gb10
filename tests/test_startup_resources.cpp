// server_start.resources: the Flash-Next route has no rank workers, so the startup record used to
// carry `ranks: []` and an all-zero `total_allocations`. These cases pin the mapping from the
// Program's MemorySummary and the record's shape, which no existing check covered: a structurally
// present, always-zero field is invisible to every assertion on the fields that do get populated.
#include "product/statistics.h"
#include "runtime/engine/flash_next/memory_report.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

namespace {

using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << "\n";
    return 1;
}

ninfer::MemorySummary populated_summary() {
    ninfer::MemorySummary summary;
    summary.device                    = 0;
    summary.kv_capacity               = 4096;
    summary.kv_capacity_page_groups   = 64;
    summary.weights.capacity_bytes    = 70ULL << 30U;
    summary.weights.used_bytes        = 70ULL << 30U;
    summary.sequence.capacity_bytes   = 512ULL << 20U;
    summary.workspace.capacity_bytes  = 1ULL << 30U;
    summary.workspace.used_bytes      = 128ULL << 20U;
    summary.workspace.peak_used_bytes = 256ULL << 20U;
    summary.workspace_logical_peak_bytes = 300ULL << 20U;
    summary.kv_payload_bytes             = 2ULL << 30U;
    summary.host_kv_capacity_bytes       = 1ULL << 30U;
    summary.host_state_capacity_slots    = 4;
    summary.runtime_reservation_bytes    = 3ULL << 30U;
    summary.minimum_runtime_reservation_bytes = 2ULL << 30U;
    summary.kv_capacity_increment_bytes       = 8ULL << 20U;
    summary.available_after_weights_bytes     = 20ULL << 30U;
    summary.available_after_startup_bytes     = 16ULL << 30U;
    summary.kv_capacity_headroom_bytes        = 12ULL << 30U;
    summary.planned_slack_bytes               = 1ULL << 30U;
    return summary;
}

} // namespace

int main() {
    int failures = 0;

    // The mapping reports the arenas the summary names, and invents nothing it does not name.
    {
        const auto usage = ninfer::flashnext_runtime::rank_memory_usage(populated_summary());
        failures += check(usage.allocations.device_weights_bytes == (70ULL << 30U),
                          "the resident weights were not mapped");
        failures += check(usage.allocations.persistent_device_bytes == (512ULL << 20U),
                          "the state-slot arena was not mapped");
        failures += check(usage.allocations.workspace_device_bytes == (1ULL << 30U),
                          "the workspace arena was not mapped");
        failures += check(usage.allocations.host_context_bytes == (1ULL << 30U),
                          "the Host KV tier was not mapped");
        failures += check(usage.kv_payload_bytes == (2ULL << 30U), "the KV payload was not mapped");
        failures += check(usage.allocations.device_bytes() != 0, "the device allocation is zero");
        const auto empty = ninfer::flashnext_runtime::rank_memory_usage({});
        failures += check(empty.allocations.device_bytes() == 0,
                          "an empty summary produced a nonzero allocation");
    }
    {
        const auto budget = ninfer::flashnext_runtime::rank_memory_budget(populated_summary());
        failures += check(budget.runtime_reservation_bytes == (3ULL << 30U) &&
                              budget.available_after_startup_bytes == (16ULL << 30U) &&
                              budget.headroom_bytes == (12ULL << 30U) &&
                              budget.bytes_per_page_group == (8ULL << 20U),
                          "the startup budget was not mapped");
    }

    // The record itself: one described rank and a non-zero allocation total.
    {
        ninfer::EngineDescription description;
        description.architecture = "Qwen3_8FlashNextForCausalLM";
        description.ranks.push_back({0, 0, 0, "", true});
        description.kv_capacity = 4096;
        ninfer::StartupReport startup;
        startup.ranks.push_back(ninfer::flashnext_runtime::rank_memory_budget(populated_summary()));
        ninfer::EngineSnapshot snapshot;
        ninfer::RankSnapshot rank;
        rank.memory = ninfer::flashnext_runtime::rank_memory_usage(populated_summary());
        snapshot.ranks.push_back(std::move(rank));

        const Json resources = ninfer::product::startup_resources_json(description, startup, snapshot);
        failures += check(resources.at("ranks").size() == 1,
                          "the startup record describes no rank");
        failures += check(resources.at("total_allocations").at("device_weights_bytes") ==
                              (70ULL << 30U),
                          "the startup record reports no device weights");
        failures += check(resources.at("total_allocations").at("device_bytes") != 0,
                          "the startup record reports a zero device allocation");
        failures += check(resources.at("ranks").at(0).at("budget").at("headroom_bytes") ==
                              (12ULL << 30U),
                          "the startup record lost the rank budget");

        // The shape the Flash-Next route produced before the fix: no ranks at all. The record must
        // still not claim an allocation, and this is the case the fix makes unreachable.
        ninfer::EngineDescription empty_description;
        ninfer::StartupReport empty_startup;
        ninfer::EngineSnapshot empty_snapshot;
        const Json empty =
            ninfer::product::startup_resources_json(empty_description, empty_startup, empty_snapshot);
        failures += check(empty.at("ranks").empty() &&
                              empty.at("total_allocations").at("device_bytes") == 0,
                          "an undescribed route invented an allocation");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
