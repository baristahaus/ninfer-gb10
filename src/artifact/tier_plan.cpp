#include "artifact/tier_plan.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::artifact {

namespace {

[[nodiscard]] const Json& require_member(const Json& value, std::string_view key,
                                         std::string_view label) {
    if (!value.is_object() || !value.contains(key)) {
        throw ArtifactError(std::string(label) + ": missing " + std::string(key));
    }
    return value.at(key);
}

void require_u32(const Json& value, std::string_view key, std::string_view label,
                 std::uint32_t& out) {
    const Json& member = require_member(value, key, label);
    if (!member.is_number_unsigned()) {
        throw ArtifactError(std::string(label) + ": " + std::string(key) + " must be unsigned");
    }
    const auto raw = member.get<std::uint64_t>();
    if (raw == 0 || raw > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(label) + ": " + std::string(key) + " out of range");
    }
    out = static_cast<std::uint32_t>(raw);
}

// Parses an expert-id array into `out`, rejecting duplicates, negatives, and out-of-range ids.
void parse_expert_ids(const Json& value, std::string_view key, std::string_view label,
                      std::uint32_t experts, std::vector<std::uint32_t>& out) {
    if (!value.is_array()) {
        throw ArtifactError(std::string(label) + ": " + std::string(key) + " must be an array");
    }
    out.reserve(value.size());
    for (const auto& entry : value) {
        if (!entry.is_number_unsigned()) {
            throw ArtifactError(std::string(label) + ": " + std::string(key) +
                                " entries must be unsigned expert ids");
        }
        const auto id = entry.get<std::uint64_t>();
        if (id >= experts) {
            throw ArtifactError(std::string(label) + ": " + std::string(key) +
                                " expert id out of range");
        }
        out.push_back(static_cast<std::uint32_t>(id));
    }
    std::vector<bool> seen(experts, false);
    for (const std::uint32_t id : out) {
        if (seen[id]) {
            throw ArtifactError(std::string(label) + ": " + std::string(key) + " declares expert " +
                                std::to_string(id) + " twice");
        }
        seen[id] = true;
    }
}

} // namespace

std::optional<TierPlan> parse_tier_plan(const Json& metadata) {
    if (!metadata.is_object() || !metadata.contains("tier_plan")) { return std::nullopt; }
    const Json& root = metadata.at("tier_plan");
    if (!root.is_object()) { throw ArtifactError("tier_plan must be an object"); }
    const Json& version = require_member(root, "version", "tier_plan");
    if (!version.is_number_unsigned() || version.get<std::uint64_t>() != 1) {
        throw ArtifactError("tier_plan: unsupported version");
    }
    const Json& banks = require_member(root, "banks", "tier_plan");
    if (!banks.is_array() || banks.empty()) {
        throw ArtifactError("tier_plan: banks must be a nonempty array");
    }
    TierPlan plan;
    plan.version = 1;
    plan.banks.reserve(banks.size());
    for (const Json& bank : banks) {
        if (!bank.is_object()) { throw ArtifactError("tier_plan bank must be an object"); }
        TierPlanBank out;
        require_member(bank, "object", "tier_plan bank").get_to(out.object);
        if (out.object.empty()) {
            throw ArtifactError("tier_plan bank: object id must be nonempty");
        }
        require_u32(bank, "experts", "tier_plan bank", out.experts);
        require_u32(bank, "rows_per_expert", "tier_plan bank", out.rows_per_expert);
        parse_expert_ids(require_member(bank, "hot", "tier_plan bank"), "hot", out.object,
                         out.experts, out.hot);
        parse_expert_ids(require_member(bank, "warm", "tier_plan bank"), "warm", out.object,
                         out.experts, out.warm);
        parse_expert_ids(require_member(bank, "cold", "tier_plan bank"), "cold", out.object,
                         out.experts, out.cold);
        // Every expert is planned exactly once across the three tiers.
        if (out.hot.size() + out.warm.size() + out.cold.size() != out.experts) {
            throw ArtifactError("tier_plan bank " + out.object +
                                ": tiers do not cover every expert exactly once");
        }
        const auto profile = bank.find("activation_profile");
        if (profile != bank.end() && !profile->is_null()) {
            if (!profile->is_array() || profile->size() != out.experts) {
                throw ArtifactError("tier_plan bank " + out.object +
                                    ": activation_profile must have one count per expert");
            }
            out.activation_profile.reserve(out.experts);
            for (const auto& count : *profile) {
                if (!count.is_number_unsigned()) {
                    throw ArtifactError("tier_plan bank " + out.object +
                                        ": activation_profile entries must be unsigned counts");
                }
                out.activation_profile.push_back(count.get<std::uint64_t>());
            }
        }
        plan.banks.push_back(std::move(out));
    }
    return plan;
}

} // namespace ninfer::artifact
