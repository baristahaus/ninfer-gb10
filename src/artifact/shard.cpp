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

} // namespace ninfer::artifact
