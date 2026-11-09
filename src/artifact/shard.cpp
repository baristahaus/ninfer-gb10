#include "artifact/shard.h"

#include <stdexcept>
#include <string>

namespace ninfer::artifact {
namespace {

void require_valid(ShardAxis axis, std::int64_t extent, int width) {
    if (width < 1) {
        throw std::invalid_argument("sharding: an execution group needs at least one rank");
    }
    if (extent < 0) {
        throw std::invalid_argument("sharding: a negative extent cannot be placed");
    }
    const bool splits = axis == ShardAxis::Rows || axis == ShardAxis::Columns;
    if (splits && extent % width != 0) {
        throw std::invalid_argument(
            "sharding: extent " + std::to_string(extent) + " does not divide into " +
            std::to_string(width) +
            " equal per-rank blocks; an uneven split would give the ranks different geometries");
    }
}

} // namespace

ShardPlacement split_evenly(ShardAxis axis, std::int64_t extent, int width) {
    require_valid(axis, extent, width);
    ShardPlacement placement;
    placement.axis = axis;
    placement.ranges.reserve(static_cast<std::size_t>(width));

    switch (axis) {
    case ShardAxis::Replicated:
        for (int rank = 0; rank < width; ++rank) {
            placement.ranges.push_back(ShardRange{0, extent});
        }
        break;
    case ShardAxis::PrimaryOnly:
        placement.ranges.push_back(ShardRange{0, extent});
        for (int rank = 1; rank < width; ++rank) {
            placement.ranges.push_back(ShardRange{0, 0});
        }
        break;
    case ShardAxis::Rows:
    case ShardAxis::Columns: {
        const std::int64_t per_rank = extent / width;
        for (int rank = 0; rank < width; ++rank) {
            placement.ranges.push_back(
                ShardRange{static_cast<std::int64_t>(rank) * per_rank, per_rank});
        }
        break;
    }
    }
    return placement;
}


Binding shard_binding(const Binding& binding, const ShardPlacement& placement, int rank) {
    if (rank < 0 || static_cast<std::size_t>(rank) >= placement.ranges.size()) {
        throw std::invalid_argument("sharding: rank outside the placement");
    }
    const ShardRange range = placement.ranges[static_cast<std::size_t>(rank)];
    if (range.begin < 0 || range.extent < 0 ||
        static_cast<std::uint64_t>(range.begin) + static_cast<std::uint64_t>(range.extent) >
            binding.elements) {
        throw std::invalid_argument("sharding: a rank's range lies outside the binding");
    }

    Binding result;
    result.elements = static_cast<std::uint64_t>(range.extent);
    result.whole_object =
        range.begin == 0 && static_cast<std::uint64_t>(range.extent) == binding.elements;
    if (range.extent == 0) { return result; }

    // Walk the parts, tracking each one's logical offset, and cut the rank's range against them.
    const std::uint64_t wanted_begin = static_cast<std::uint64_t>(range.begin);
    const std::uint64_t wanted_end   = wanted_begin + static_cast<std::uint64_t>(range.extent);
    std::uint64_t logical_begin      = 0;
    for (const Part& part : binding.parts) {
        const std::uint64_t part_elements = part.end - part.begin;
        const std::uint64_t logical_end   = logical_begin + part_elements;
        const std::uint64_t overlap_begin = wanted_begin > logical_begin ? wanted_begin : logical_begin;
        const std::uint64_t overlap_end   = wanted_end < logical_end ? wanted_end : logical_end;
        if (overlap_begin < overlap_end) {
            Part sliced;
            sliced.object = part.object;
            sliced.begin  = part.begin + (overlap_begin - logical_begin);
            sliced.end    = part.begin + (overlap_end - logical_begin);
            result.parts.push_back(std::move(sliced));
        }
        logical_begin = logical_end;
        if (logical_begin >= wanted_end) { break; }
    }
    if (result.parts.empty() && range.extent != 0) {
        throw std::invalid_argument("sharding: the binding does not cover the rank's range");
    }
    return result;
}

} // namespace ninfer::artifact
