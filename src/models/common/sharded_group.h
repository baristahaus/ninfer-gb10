#pragma once

// The group-scoped resources of a two-GPU (two-Program) lane: ONE mailbox for the whole group and
// ONE partial buffer per rank, shared by both Programs.
//
// WHY ONE OWNER. A mailbox owns a pinned host slab and per-device arrival/epoch words; two of them
// would never rendezvous, because a rank's exchange kernel would wait for a peer that never writes
// to the same slab. The partial pair has the same shape of requirement: the combine reads and
// writes BOTH buffers, so both Programs must pass the same pair. Both therefore belong to the group,
// not to a Program.
//
// LIFETIME, which is the part that bites. The owner is constructed once, after the execution group
// exists and before the first Program, and it must outlive every Program and every graph capture
// that baked in the mailbox slab and the partial addresses. In the engine that means `BuiltModel`
// owns it: declared after `execution` (the devices) and before `instance`/`peers` (the Programs), so
// it is destroyed after the Programs and before the devices. A Program borrowing it must not
// outlive it — the borrow is a raw pointer by design, because the owner is the only thing that can
// outlive the Programs.
//
// WIDTH 1 constructs nothing: the owner exists only when the group has more than one rank, so the
// single-device lane pays no pinned memory and no device words.

#include "core/buffer.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/peer_mailbox.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::models {

// What a Program borrows. Pointers and views only; the owner holds the storage, and it outlives
// every borrower. Pointer members rather than references so this is default-constructible and
// assignable, which is what lets the owner fill it in after allocating the buffers.
struct ShardedGroup {
    const ExecutionContext* execution = nullptr;
    ops::PeerMailbox* mailbox         = nullptr;
    // Full-width BF16 [rows, columns], one buffer per rank's device, in rank order.
    std::array<Tensor, 2> partials{};
};

// The rank of `device_index` in a group whose members are `device_indices`, or -1 when it is not a
// member. Pure — no CUDA, no allocation — so the rule is testable without a device. It is the one
// place that decides which stream a rank's exchange lands on.
[[nodiscard]] int sharded_rank_of(std::span<const int> device_indices, int device_index) noexcept;

// Owns the group's mailbox and its two partial buffers. Constructed once for the whole group.
class ShardedGroupOwner {
public:
    // `rows` x `columns` is one full-width BF16 partial (for the 27B: hidden_size x the plan's token
    // capacity). Throws std::invalid_argument for a group that is not exactly two ranks and
    // propagates the mailbox's allocation errors.
    ShardedGroupOwner(const ExecutionContext& execution, std::int32_t rows, std::int32_t columns);
    ~ShardedGroupOwner();

    ShardedGroupOwner(const ShardedGroupOwner&)            = delete;
    ShardedGroupOwner& operator=(const ShardedGroupOwner&) = delete;
    ShardedGroupOwner(ShardedGroupOwner&&)                 = delete;
    ShardedGroupOwner& operator=(ShardedGroupOwner&&)      = delete;

    [[nodiscard]] const ShardedGroup& group() const noexcept { return group_; }

    [[nodiscard]] const ExecutionContext& execution() const noexcept { return execution_; }

    // The buffers' own devices, in rank order, so a caller can check the pair without reaching into
    // the views.
    [[nodiscard]] Device partial_device(int rank) const;

private:
    const ExecutionContext& execution_;
    ops::PeerMailbox mailbox_;
    std::array<DeviceBuffer, 2> buffers_;
    std::array<Tensor, 2> partials_;
    // Declared last: it references the three members above, and the owner is constructed in
    // declaration order.
    ShardedGroup group_;
};

} // namespace ninfer::models
