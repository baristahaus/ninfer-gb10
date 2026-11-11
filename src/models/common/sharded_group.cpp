// Implements: src/models/common/sharded_group.h
#include "models/common/sharded_group.h"

#include <cstdio>
#include <cstdlib>
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

// The mailbox slot is sized for the widest payload the group exchanges. That is normally one
// column-parallel partial, but a logits gather exchanges whole columns of the full vocabulary, and
// the gather cannot run at all unless the slot carries one: the slot's extent comes from the KV
// capacity, which has nothing to do with the vocabulary. One column is the floor, so a lane whose
// partial slot is smaller than a logits column still gathers (in as many exchanges as the payload
// needs) instead of refusing.
std::size_t exchange_slot_bytes(std::int32_t rows, std::int32_t columns, std::int32_t output_rows,
                                std::int32_t logits_columns) {
    const std::size_t partial = partial_bytes(rows, columns);
    if (output_rows <= 0 || logits_columns <= 0) { return partial; }
    const std::size_t column = static_cast<std::size_t>(output_rows) * sizeof(std::uint16_t);
    return column > partial ? column : partial;
}

} // namespace

int sharded_rank_of(std::span<const int> device_indices, int device_index) noexcept {
    for (std::size_t rank = 0; rank < device_indices.size(); ++rank) {
        if (device_indices[rank] == device_index) { return static_cast<int>(rank); }
    }
    return -1;
}

ShardedGroupOwner::ShardedGroupOwner(const ExecutionContext& execution, std::int32_t rows,
                                     std::int32_t columns, std::int32_t output_rows,
                                     std::int32_t logits_columns)
    : execution_(execution),
      mailbox_(require_two_ranks(execution),
               exchange_slot_bytes(rows, columns, output_rows, logits_columns)) {
    // One buffer per rank, on that rank's own device: the combine reads and writes both, so they
    // must be device-resident where their rank's exchange kernel runs.
    const std::size_t bytes = partial_bytes(rows, columns);
    for (int rank = 0; rank < 2; ++rank) {
        const Device device = execution_.device(rank).device();
        const CudaDeviceScope scope(device);
        buffers_[rank]  = DeviceBuffer(device, bytes);
        partials_[rank] = Tensor{buffers_[rank].span(), DType::BF16, {rows, columns}};
    }
    // The logits pair is optional, but half of it is not: an extent given without the other is a
    // caller that meant to gather and mistyped, and building nothing would turn that into a gather
    // that finds no buffer later. Refuse here, where the extents are still visible.
    if ((output_rows > 0) != (logits_columns > 0)) {
        throw std::invalid_argument(
            "sharded group: the logits pair needs both extents; got output_rows=" +
            std::to_string(output_rows) + " logits_columns=" + std::to_string(logits_columns));
    }
    if (output_rows > 0) {
        const std::size_t logits_bytes = partial_bytes(output_rows, logits_columns);
        for (int rank = 0; rank < 2; ++rank) {
            const Device device = execution_.device(rank).device();
            const CudaDeviceScope scope(device);
            logits_buffers_[rank] = DeviceBuffer(device, logits_bytes);
            logits_[rank] =
                Tensor{logits_buffers_[rank].span(), DType::BF16, {output_rows, logits_columns}};
        }
    }
    // DIAGNOSTIC (NINFER_MAILBOX_TRACE=1): the group's device buffers, per rank, and an assertion
    // that no two of them are the same allocation. The audit that preceded this checked the partial
    // pair by SHAPE; if a rank's reduce lands in an allocation that is also the logits pair, the
    // sum is not a sum and the magnitude is whatever lives there.
    if (std::getenv("NINFER_MAILBOX_TRACE") != nullptr) {
        for (int rank = 0; rank < 2; ++rank) {
            std::fprintf(stderr,
                         "[group] rank=%d dev=%d partials=%p logits=%p mailbox_slab=%p\n", rank,
                         execution_.device(rank).device().index(),
                         static_cast<const void*>(partials_[rank].data),
                         static_cast<const void*>(logits_[rank].data),
                         static_cast<const void*>(mailbox_.payload_base(rank)));
        }
        const auto* p0 = static_cast<const std::byte*>(partials_[0].data);
        const auto* p1 = static_cast<const std::byte*>(partials_[1].data);
        const auto* l0 = static_cast<const std::byte*>(logits_[0].data);
        const auto* l1 = static_cast<const std::byte*>(logits_[1].data);
        if (p0 == p1 || p0 == l0 || p0 == l1 || p1 == l0 || p1 == l1 || l0 == l1) {
            throw std::logic_error(
                "sharded group: two of the group's buffers are the same allocation");
        }
    }
    group_.execution = &execution_;
    group_.mailbox   = &mailbox_;
    group_.partials  = partials_;
    group_.logits    = logits_;
}

ShardedGroupOwner::~ShardedGroupOwner() = default;

Device ShardedGroupOwner::partial_device(int rank) const {
    if (rank < 0 || rank >= 2) {
        throw std::out_of_range("sharded group: rank outside [0,2)");
    }
    return partials_[static_cast<std::size_t>(rank)].device;
}

} // namespace ninfer::models
