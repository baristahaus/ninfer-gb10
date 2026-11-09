#pragma once

// Two-device all-reduce for the tensor-parallel lane.
//
// Ported from capy-flash `include/ninfer/ops/allreduce.h`, reduced to the transport this tree
// carries: the pinned-host mailbox. capy's staged path (its `PeerEvents` / `HostExchange` /
// `PeerLanes`) is deliberately not ported. Measurement on the target box shows that PCIe P2P —
// which is what the staged path exists to exploit — collapses from 26.9 GB/s to 2.2 GB/s once the
// engine is resident, while VRAM occupancy, BAR1 usage, pinned host memory and a foreign process
// holding the VRAM all leave it flat. The mailbox is therefore the transport that actually serves,
// and it is the one ported and tested here. See `review/95-NEXT-STEP.md` §7.2.

#include "core/device.h"
#include "ninfer/ops/peer_mailbox.h"

#include <cstddef>
#include <span>

namespace ninfer::ops {

// True when both devices report peer access to each other. With it a cross-device copy is direct;
// without it the driver stages the copy through host memory. Recorded because it decides whether a
// future staged transport could earn its place, not because this transport depends on it.
[[nodiscard]] bool peer_access_available(const ExecutionContext& ec) noexcept;

// Enables peer access in both directions when the driver allows it and reports whether it is on
// afterwards. Enabling twice is not an error.
bool enable_peer_access(const ExecutionContext& ec);

// Sums one BF16 payload across the two ranks in place, through `mailbox`:
//
//   buffer[r][i] = RNE_bf16(fl32(buffer[0][i]) + fl32(buffer[1][i]))   for both ranks r
//
// so after the call both ranks hold the identical sum. `data[r]` is rank r's device-resident
// payload and `bytes` is the same for both ranks: a multiple of 16, at most
// `mailbox.max_bytes()`. Both ranks call this together, having taken one slot from
// `PeerMailbox::take_capture_slot(bytes)`, and each rank's half is enqueued on that rank's own
// stream.
//
// The addition is commutative, so the two ranks compute the same bits from the same two partials;
// the local combine is float(mine) + float(peer) in FP32 with one round-to-nearest-even on store.
// Consecutive calls need no host synchronization between them.
// `mailbox` is mutable: taking a slot advances its round robin.
void allreduce_sum(const ExecutionContext& ec, PeerMailbox& mailbox, std::span<void* const> data,
                   std::size_t bytes);

} // namespace ninfer::ops
