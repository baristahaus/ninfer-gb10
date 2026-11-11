// SPDX-FileCopyrightText: ValerioDolci/capy-flash
// SPDX-License-Identifier: Apache-2.0
// Implements: include/ninfer/ops/allreduce.h
//
// The mailbox transport only. See the header for why capy's staged path is not ported.

#include "ninfer/ops/allreduce.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

// Runs `fn` with `rank`'s device current, restoring the previous device afterwards.
template <class Function>
auto with_device(int device, Function&& fn) -> decltype(fn()) {
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess) {
        throw std::runtime_error("allreduce: cudaGetDevice failed");
    }
    if (cudaSetDevice(device) != cudaSuccess) {
        (void)cudaSetDevice(previous);
        throw std::runtime_error("allreduce: cudaSetDevice failed");
    }
    try {
        auto result = fn();
        (void)cudaSetDevice(previous);
        return result;
    } catch (...) {
        (void)cudaSetDevice(previous);
        throw;
    }
}

} // namespace

bool peer_access_available(const ExecutionContext& ec) noexcept {
    if (ec.width() != 2) { return false; }
    const int first  = ec.device(0).device().index();
    const int second = ec.device(1).device().index();
    int forward      = 0;
    int backward     = 0;
    if (cudaDeviceCanAccessPeer(&forward, first, second) != cudaSuccess) { return false; }
    if (cudaDeviceCanAccessPeer(&backward, second, first) != cudaSuccess) { return false; }
    return forward != 0 && backward != 0;
}

bool enable_peer_access(const ExecutionContext& ec) {
    if (!peer_access_available(ec)) { return false; }
    const int first  = ec.device(0).device().index();
    const int second = ec.device(1).device().index();
    // Already-enabled is the success case on a second call.
    const auto enable = [](int device, int peer) {
        return with_device(device, [peer] {
            const cudaError_t status = cudaDeviceEnablePeerAccess(peer, 0);
            return status == cudaSuccess || status == cudaErrorPeerAccessAlreadyEnabled;
        });
    };
    return enable(first, second) && enable(second, first);
}

namespace {

// The validation both forms share. `name` is the caller's name for its own error messages.
void validate_reduce(const ExecutionContext& ec, const PeerMailbox& mailbox,
                     std::span<void* const> data, std::size_t bytes, const char* name) {
    if (ec.width() != 2) {
        throw std::invalid_argument(std::string(name) +
                                    ": requires an ExecutionContext of width 2");
    }
    if (data.size() != 2) {
        throw std::invalid_argument(std::string(name) + ": requires one payload per rank");
    }
    if (data[0] == nullptr || data[1] == nullptr) {
        throw std::invalid_argument(std::string(name) + ": a payload must not be null");
    }
    if (bytes == 0 || bytes % 16 != 0) {
        throw std::invalid_argument(std::string(name) + ": bytes must be whole 16-byte vectors");
    }
    if (bytes > mailbox.max_bytes()) {
        throw std::invalid_argument(std::string(name) +
                                    ": payload exceeds the mailbox's largest slot");
    }
}

// One rank's exchange, on that rank's own stream, in a slot the caller has already taken. This is
// the single primitive both public forms are built from.
void enqueue_exchange(const ExecutionContext& ec, PeerMailbox& mailbox, int rank, int slot,
                      std::span<void* const> data, std::size_t bytes) {
    const int device = ec.device(rank).device().index();
    (void)with_device(device, [&] {
        mailbox.enqueue_exchange_sum(rank, slot, data[rank], bytes, ec.device(rank).stream());
        return 0;
    });
}

} // namespace

void allreduce_sum(const ExecutionContext& ec, PeerMailbox& mailbox,
                   std::span<void* const> data, std::size_t bytes) {
    validate_reduce(ec, mailbox, data, bytes, "allreduce_sum");
    // The single-owner form: one slot, both ranks' exchanges. Unchanged behaviour.
    const int slot = mailbox.take_capture_slot(bytes);
    for (int rank = 0; rank < 2; ++rank) {
        enqueue_exchange(ec, mailbox, rank, slot, data, bytes);
    }
}

void allreduce_sum_rank(const ExecutionContext& ec, PeerMailbox& mailbox, int rank, int slot,
                        std::span<void* const> data, std::size_t bytes) {
    validate_reduce(ec, mailbox, data, bytes, "allreduce_sum_rank");
    if (rank < 0 || rank >= 2) {
        throw std::invalid_argument("allreduce_sum_rank: rank outside [0,2)");
    }
    if (slot < 0) {
        throw std::invalid_argument("allreduce_sum_rank: slot outside the mailbox");
    }
    // The per-rank form: this rank's own exchange, in the slot the caller chose so that both ranks
    // agree on it. The caller owns `rank`'s stream, so enqueueing this after its projection is what
    // orders the two. `enqueue_exchange_sum` bounds-checks the slot.
    enqueue_exchange(ec, mailbox, rank, slot, data, bytes);
}

} // namespace ninfer::ops
