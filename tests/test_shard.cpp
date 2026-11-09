// Host-only contract tests for the sharding arithmetic: the per-rank ranges a logical parameter is
// split into, and the rejections that keep the ranks' geometries equal.

#include "artifact/shard.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::artifact::ShardAxis;
using ninfer::artifact::ShardPlacement;
using ninfer::artifact::ShardRange;
using ninfer::artifact::split_evenly;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

bool throws_invalid(const ShardAxis axis, const std::int64_t extent, const int width) {
    try {
        (void)split_evenly(axis, extent, width);
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

int test_even_splits() {
    int failures = 0;

    // The Qwen3.8 27B's query heads at width 2 and 3: 24 divides both ways.
    const ShardPlacement q_two = split_evenly(ShardAxis::Rows, 24, 2);
    failures += check(q_two.ranges == std::vector<ShardRange>{{0, 12}, {12, 12}},
                      "24 rows over 2 ranks is 12 each");
    const ShardPlacement q_three = split_evenly(ShardAxis::Rows, 24, 3);
    failures += check(q_three.ranges == std::vector<ShardRange>{{0, 8}, {8, 8}, {16, 8}},
                      "24 rows over 3 ranks is 8 each");

    // A column split of the hidden width.
    const ShardPlacement columns = split_evenly(ShardAxis::Columns, 5120, 2);
    failures += check(columns.ranges == std::vector<ShardRange>{{0, 2560}, {2560, 2560}},
                      "5120 columns over 2 ranks is 2560 each");

    // Width 1 is the ordinary single-device engine: one whole range.
    const ShardPlacement single = split_evenly(ShardAxis::Rows, 248320, 1);
    failures += check(single.ranges == std::vector<ShardRange>{{0, 248320}},
                      "one rank owns the whole parameter");

    // Replicated: every rank holds all of it.
    const ShardPlacement replicated = split_evenly(ShardAxis::Replicated, 2560, 3);
    failures += check(replicated.ranges == std::vector<ShardRange>{{0, 2560}, {0, 2560}, {0, 2560}},
                      "replicated parameters are whole on every rank");

    // PrimaryOnly: rank 0 whole, every other rank empty.
    const ShardPlacement primary = split_evenly(ShardAxis::PrimaryOnly, 4096, 3);
    failures += check(primary.ranges ==
                          std::vector<ShardRange>{{0, 4096}, {0, 0}, {0, 0}},
                      "a primary-only parameter is whole on rank 0 and empty elsewhere");
    failures += check(primary.ranges[1].empty() && primary.ranges[2].empty(),
                      "the empty ranks report empty ranges");

    // A zero extent is a legal empty parameter.
    const ShardPlacement empty = split_evenly(ShardAxis::Rows, 0, 2);
    failures += check(empty.ranges == std::vector<ShardRange>{{0, 0}, {0, 0}},
                      "a zero extent splits into zero extents");
    return failures;
}

int test_rejections() {
    int failures = 0;
    // The extents that actually matter: the models' KV heads (4 and 2), the GDN key heads (16),
    // the vocabulary (248320 = 2^9*5*97) and the dense intermediate (17408 = 2^10*17).
    failures += check(throws_invalid(ShardAxis::Rows, 4, 3), "4 KV heads do not split over 3 ranks");
    failures += check(throws_invalid(ShardAxis::Rows, 2, 3), "2 KV heads do not split over 3 ranks");
    failures += check(throws_invalid(ShardAxis::Rows, 16, 3), "16 GDN key heads do not split over 3");
    failures += check(throws_invalid(ShardAxis::Rows, 248320, 3), "the vocabulary does not split over 3");
    failures += check(throws_invalid(ShardAxis::Columns, 17408, 3), "17408 columns do not split over 3");
    failures += check(throws_invalid(ShardAxis::Rows, 8, 0), "a zero-width group is rejected");
    failures += check(throws_invalid(ShardAxis::Rows, 8, -2), "a negative width is rejected");
    failures += check(throws_invalid(ShardAxis::Rows, -8, 2), "a negative extent is rejected");

    // Replicated and PrimaryOnly accept any extent: they are not splits.
    bool replicated_ok = true;
    try {
        (void)split_evenly(ShardAxis::Replicated, 7, 3);
    } catch (...) {
        replicated_ok = false;
    }
    failures += check(replicated_ok, "replicated placement does not need a divisible extent");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_even_splits();
    failures += test_rejections();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
