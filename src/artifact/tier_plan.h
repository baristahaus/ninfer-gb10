#pragma once

#include "artifact/schema.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::artifact {

// Expert-tiering placement plan (strategy SP-4). A tier plan lives in the container
// directory's open metadata object under the "tier_plan" member and declares, per routed
// expert bank, which experts are planned for the Device (hot), pinned-host (warm), and
// file-backed (cold) tiers, plus an optional per-expert activation profile for hot-set
// seeding. Version 1 is declarative: the loader validates the plan and reports it; the
// materialization honors it only when the tiered execution path consumes it.
struct TierPlanBank {
    std::string object;
    std::uint32_t experts         = 0;
    std::uint32_t rows_per_expert = 0;
    std::vector<std::uint32_t> hot;
    std::vector<std::uint32_t> warm;
    std::vector<std::uint32_t> cold;
    // Per-expert activation counts from the conversion-time calibration; empty when absent.
    std::vector<std::uint64_t> activation_profile;
};

struct TierPlan {
    std::uint32_t version = 1;
    std::vector<TierPlanBank> banks;
};

// Parses directory.metadata's "tier_plan" member. Returns nullopt when the member is absent;
// throws ArtifactError when it is present but malformed. The tier arrays must disjointly and
// completely cover [0, experts); an activation profile, when present, must have exactly
// `experts` non-negative counts.
[[nodiscard]] std::optional<TierPlan> parse_tier_plan(const Json& metadata);

} // namespace ninfer::artifact
