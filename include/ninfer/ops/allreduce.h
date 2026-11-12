// SPDX-FileCopyrightText: ValerioDolci/capy-flash
// SPDX-License-Identifier: Apache-2.0
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
// payload and `bytes` is the same for both ranks: a multiple of 16, at most `mailbox.max_bytes()`.
//
// TWO CALLING FORMS, and they are NOT interchangeable:
//
//   * `allreduce_sum` — the SINGLE-OWNER form. ONE caller takes one capture slot and enqueues BOTH
//     ranks' exchange kernels, on both ranks' streams. This is the form for a Program that owns both
//     devices (the Flash-Next lane, whose ExecutionContext is shared inside one Program).
//     Calling it from each rank's own Program takes two slots and enqueues four exchanges per
//     reduce, which desynchronises the slot round robin.
//   * `allreduce_sum_rank` — the PER-RANK form. Each rank's Program calls it once, for its own
//     rank. It takes its own slot and enqueues only that rank's exchange, so the caller's own
//     stream order covers the dependency between its projection and its exchange. Use this whenever
//     the two ranks are two Programs on two threads.
//
// The earlier wording here said "both ranks call this together". That was true of the single-owner
// model and is false of the two-Program model; it is exactly the kind of sentence that reads as a
// contract and is not one. Pick the form by who owns the two streams, not by how many ranks take
// part in the arithmetic.
//
// Both forms produce the same bits: the addition is commutative, so the two ranks compute the same
// result from the same two partials; the local combine is float(mine) + float(peer) in FP32 with one
// round-to-nearest-even on store. Consecutive calls need no host synchronization between them.
// `mailbox` is mutable: taking a slot advances its round robin.
void allreduce_sum(const ExecutionContext& ec, PeerMailbox& mailbox, std::span<void* const> data,
                   std::size_t bytes);

// One rank's half of the sum above: enqueues rank `rank`'s exchange on `ec.device(rank)`'s stream
// only, in the caller's `slot`. `data` still carries both payloads, because the combine writes the
// full result into both — the peer's pointer is the caller's business, and both ranks must pass the
// same pair. The caller must be the owner of `rank`'s stream and must enqueue this after whatever
// produced `data[rank]` on that stream; that stream order is what makes the exchange read a
// complete partial.
//
// THE SLOT IS THE CALLER'S. The two ranks must exchange in the same slot, so the slot cannot come
// from a shared counter that each rank advances (they would advance it twice per reduce and land on
// different slots). A caller with a deterministic sequence — both ranks calling the same reduces in
// the same order — derives it with `PeerMailbox::slot_for(index, bytes)` from its own index.
// Throws std::invalid_argument for a width other than 2, a rank outside [0, 2), a slot outside the
// mailbox, a payload set that is not one per rank, a null payload, or bytes that are not a nonzero
// multiple of 16 within `mailbox.max_bytes()`.
void allreduce_sum_rank(const ExecutionContext& ec, PeerMailbox& mailbox, int rank, int slot,
                        std::span<void* const> data, std::size_t bytes,
                        const char* site = "?");

} // namespace ninfer::ops
