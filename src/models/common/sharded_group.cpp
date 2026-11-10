// Implements: src/models/common/sharded_group.h
#include "models/common/sharded_group.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models {
namespace {

const ExecutionContext& require_two_ranks(const ExecutionContext& execution) {
    if (execution.width() != 2) {
        throw std::invalid_argument(
            "sharded group: a two-Program collective needs an execution group of exactly two ranks");
    }
    return execution;
}

std::size_t partial_bytes(std::int32_t rows, std::int32_t columns) {
    if (rows <= 0 || columns <= 0) {
        throw std::invalid_argument("sharded group: a partial must have positive extents");
    }
    const std::uint64_t cells =
        static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(columns);
    if (cells > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
        throw std::overflow_error("sharded group: the partial size overflows size_t");
    }
    return static_cast<std::size_t>(cells) * sizeof(std::uint16_t);
}

} // namespace

int sharded_rank_of(std::span<const int> device_indices, int device_index) noexcept {
    for (std::size_t rank = 0; rank < device_indices.size(); ++rank) {
        if (device_indices[rank] == device_index) { return static_cast<int>(rank); }
    }
    return -1;
}

ShardedGroupOwner::ShardedGroupOwner(const ExecutionContext& execution, std::int32_t rows,
                                     std::int32_t columns)
    : execution_(execution),
      mailbox_(require_two_ranks(execution), partial_bytes(rows, columns)) {
    // One buffer per rank, on that rank's own device: the combine reads and writes both, so they
    // must be device-resident where their rank's exchange kernel runs.
    const std::size_t bytes = partial_bytes(rows, columns);
    for (int rank = 0; rank < 2; ++rank) {
        const Device device = execution_.device(rank).device();
        const CudaDeviceScope scope(device);
        buffers_[rank]  = DeviceBuffer(device, bytes);
        partials_[rank] = Tensor{buffers_[rank].span(), DType::BF16, {rows, columns}};
    }
    group_.execution = &execution_;
    group_.mailbox   = &mailbox_;
    group_.partials  = partials_;
}

ShardedGroupOwner::~ShardedGroupOwner() = default;

Device ShardedGroupOwner::partial_device(int rank) const {
    if (rank < 0 || rank >= 2) {
        throw std::out_of_range("sharded group: rank outside [0,2)");
    }
    return partials_[static_cast<std::size_t>(rank)].device;
}

} // namespace ninfer::models
