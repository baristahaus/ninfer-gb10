// Host-only contract tests for the sharding arithmetic: the per-rank ranges a logical parameter is
// split into, and the rejections that keep the ranks' geometries equal.

#include "artifact/shard.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::artifact::Binding;
using ninfer::artifact::ObjectHandle;
using ninfer::artifact::Part;
using ninfer::artifact::ShardAxis;
using ninfer::artifact::shard_binding;
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

Binding single_part_binding(std::size_t object, std::uint64_t elements) {
    Binding binding;
    binding.elements = elements;
    binding.parts.push_back(Part{ObjectHandle{object}, 0, elements});
    return binding;
}

int test_shard_binding() {
    int failures = 0;

    // One stored object, split in half: each rank gets half of the same object.
    const Binding whole = single_part_binding(7, 24);
    const auto halves   = split_evenly(ShardAxis::Rows, 24, 2);
    const Binding rank0 = shard_binding(whole, halves, 0);
    const Binding rank1 = shard_binding(whole, halves, 1);
    failures += check(rank0.parts.size() == 1 && rank0.parts[0].begin == 0 && rank0.parts[0].end == 12,
                      "rank 0 owns the first half of the object");
    failures += check(rank1.parts.size() == 1 && rank1.parts[0].begin == 12 && rank1.parts[0].end == 24,
                      "rank 1 owns the second half of the object");
    failures += check(rank0.parts[0].object == ObjectHandle{7}, "the object is unchanged");
    failures += check(rank0.elements == 12 && rank1.elements == 12, "each rank's element count halves");
    failures += check(!rank0.whole_object && !rank1.whole_object,
                      "a half is not the whole object");

    // Two stored parts: a rank's range can span both.
    Binding two_parts;
    two_parts.elements = 24;
    two_parts.parts.push_back(Part{ObjectHandle{1}, 0, 12});
    two_parts.parts.push_back(Part{ObjectHandle{2}, 0, 12});
    // A range covering logical [8, 16) crosses the part boundary at 12.
    const auto thirds = split_evenly(ShardAxis::Rows, 24, 3); // {0,8},{8,8},{16,8}
    const Binding middle = shard_binding(two_parts, thirds, 1);
    failures += check(middle.parts.size() == 2, "a range spanning two parts yields two slices");
    failures += check(middle.parts[0].object == ObjectHandle{1} && middle.parts[0].begin == 8 &&
                          middle.parts[0].end == 12,
                      "the first slice is the tail of the first part");
    failures += check(middle.parts[1].object == ObjectHandle{2} && middle.parts[1].begin == 0 &&
                          middle.parts[1].end == 4,
                      "the second slice is the head of the second part");
    failures += check(middle.elements == 8, "the rank's element count is its range");

    // Replicated: every rank gets the whole binding, flagged whole.
    const auto replicated = split_evenly(ShardAxis::Replicated, 24, 2);
    const Binding all      = shard_binding(whole, replicated, 1);
    failures += check(all.parts.size() == 1 && all.parts[0].begin == 0 && all.parts[0].end == 24,
                      "a replicated parameter is whole on every rank");
    failures += check(all.whole_object, "a replicated parameter keeps the whole-object flag");

    // PrimaryOnly: rank 1 holds nothing.
    const auto primary = split_evenly(ShardAxis::PrimaryOnly, 24, 2);
    const Binding none = shard_binding(whole, primary, 1);
    failures += check(none.parts.empty() && none.elements == 0,
                      "a rank that owns none of a primary-only parameter gets an empty binding");
    const Binding first = shard_binding(whole, primary, 0);
    failures += check(first.whole_object && first.elements == 24, "rank 0 owns the whole parameter");

    // Rejections.
    bool threw = false;
    try {
        (void)shard_binding(whole, halves, 2);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    failures += check(threw, "a rank outside the placement is rejected");

    threw = false;
    try {
        (void)shard_binding(whole, split_evenly(ShardAxis::Rows, 48, 2), 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    failures += check(threw, "a range outside the binding is rejected");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_even_splits();
    failures += test_rejections();
    failures += test_shard_binding();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
