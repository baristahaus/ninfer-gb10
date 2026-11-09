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
using ninfer::models::flash_next_125b_a6b::ranges_for;
using ninfer::models::flash_next_125b_a6b::total_extent;

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
        for (std::uint8_t index = 0; index < shard.segment_count; ++index) {
            failures += check(splits_at(shard.axis, shard.segments[index], 2),
                              std::string(shard.suffix) + " segment " + std::to_string(index) +
                                  " must split at width 2");
        }
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
            bool splits = true;
            for (std::uint8_t index = 0; index < shard.segment_count; ++index) {
                splits = splits && splits_at(shard.axis, shard.segments[index], 3);
            }
            failures += check(splits == row.splits,
                              std::string(row.suffix) + " at width 3 must " +
                                  (row.splits ? "split" : "refuse"));
        }
        failures += check(found, std::string(row.suffix) + " is in the shard table");
    }
    return failures;
}

int test_qkv_segmentation() {
    int failures = 0;
    // GDN's QKV projection is three segments, because a contiguous half of the 10240 rows would
    // give rank 0 all of Q and K and rank 1 none of either.
    const auto& shards  = parameter_shards();
    const auto* qkv     = &shards.front();
    bool found_qkv      = false;
    for (const auto& shard : shards) {
        if (shard.suffix == "linear_attn.in_proj_qkv.weight") { qkv = &shard; found_qkv = true; }
    }
    failures += check(found_qkv, "the QKV projection is in the table");
    if (!found_qkv) { return failures; }
    failures += check(qkv->segment_count == 3, "the QKV projection has three segments");
    failures += check(total_extent(*qkv) == 10240, "the QKV segments cover 10240 rows");

    // At width 2 each rank owns half of each segment: 1024 + 1024 + 3072.
    const auto rank0 = ranges_for(*qkv, 2, 0);
    const auto rank1 = ranges_for(*qkv, 2, 1);
    failures += check(rank0.size() == 3 && rank1.size() == 3, "each rank owns three blocks");
    failures += check(rank0[0] == ninfer::artifact::ShardRange{0, 1024} &&
                          rank0[1] == ninfer::artifact::ShardRange{2048, 1024} &&
                          rank0[2] == ninfer::artifact::ShardRange{4096, 3072},
                      "rank 0 takes the first half of Q, K and V");
    failures += check(rank1[0] == ninfer::artifact::ShardRange{1024, 1024} &&
                          rank1[1] == ninfer::artifact::ShardRange{3072, 1024} &&
                          rank1[2] == ninfer::artifact::ShardRange{7168, 3072},
                      "rank 1 takes the second half of Q, K and V");
    // A single-segment parameter still yields one contiguous block.
    for (const auto& shard : shards) {
        if (shard.suffix != "self_attn.k_proj.weight") { continue; }
        const auto r0 = ranges_for(shard, 2, 0);
        failures += check(r0.size() == 1 && r0[0] == ninfer::artifact::ShardRange{0, 256},
                          "the KV projection is one block per rank");
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_width_two_is_complete();
    failures += test_width_three_is_partial();
    failures += test_qkv_segmentation();
    if (failures == 0) {
        std::cout << "ok (" << parameter_shards().size() << " parameters)\n";
    }
    return failures == 0 ? 0 : 1;
}
