// Host-only contract test for the Flash-Next 125B's shard table: every parameter must split
// cleanly at width 2, and the table must report honestly which do not at width 3. This is the
// review's head-count finding made executable.

#include "artifact/shard.h"
#include "models/qwen3_8_flash_next_125b_a6b/impl/shard_rules.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using ninfer::artifact::ShardAxis;
using ninfer::artifact::split_evenly;
using ninfer::models::flash_next_125b_a6b::parameter_shards;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

bool splits_at(const ShardAxis axis, const std::int64_t extent, const int width) {
    try {
        const auto placement = split_evenly(axis, extent, width);
        return placement.ranges.size() == static_cast<std::size_t>(width);
    } catch (const std::invalid_argument&) {
        return false;
    }
}

int test_width_two_is_complete() {
    int failures = 0;
    for (const auto& shard : parameter_shards()) {
        failures += check(splits_at(shard.axis, shard.extent, 2),
                          std::string(shard.suffix) + " must split at width 2");
    }
    failures += check(!parameter_shards().empty(), "the shard table is not empty");
    return failures;
}

int test_width_three_is_partial() {
    // The rows that must refuse width 3. Everything else in the table may split; these are the
    // extents the review measured as hostile.
    struct Expected {
        const char* suffix;
        bool splits;
    };
    const Expected expected[] = {
        {"self_attn.k_proj.weight", false},        // 2 KV heads
        {"self_attn.v_proj.weight", false},        // 2 KV heads
        {"self_attn.o_proj.weight", true},         // 6144 = 24 x 256 divides by 3
        {"linear_attn.A_log", true},               // 48 divides by 3
        {"linear_attn.in_proj_qkv.weight", false}, // 10240 does not divide by 3
        {"linear_attn.in_proj_z.weight", true},    // 6144 divides by 3
        {"linear_attn.out_proj.weight", true},     // 6144 divides by 3
    };
    int failures = 0;
    for (const auto& row : expected) {
        bool found = false;
        for (const auto& shard : parameter_shards()) {
            if (shard.suffix != row.suffix) { continue; }
            found = true;
            const bool splits = splits_at(shard.axis, shard.extent, 3);
            failures += check(splits == row.splits,
                              std::string(row.suffix) + " at width 3 must " +
                                  (row.splits ? "split" : "refuse"));
        }
        failures += check(found, std::string(row.suffix) + " is in the shard table");
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_width_two_is_complete();
    failures += test_width_three_is_partial();
    if (failures == 0) {
        std::cout << "ok (" << parameter_shards().size() << " parameters)\n";
    }
    return failures == 0 ? 0 : 1;
}
