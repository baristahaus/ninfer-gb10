// Implements: include/ninfer/ops/allreduce.h
//
// The mailbox transport only. See the header for why capy's staged path is not ported.

#include "ninfer/ops/allreduce.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <stdexcept>

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

void allreduce_sum(const ExecutionContext& ec, PeerMailbox& mailbox,
                   std::span<void* const> data, std::size_t bytes) {
    if (ec.width() != 2) {
        throw std::invalid_argument("allreduce_sum: requires an ExecutionContext of width 2");
    }
    if (data.size() != 2) {
        throw std::invalid_argument("allreduce_sum: requires one payload per rank");
    }
    if (data[0] == nullptr || data[1] == nullptr) {
        throw std::invalid_argument("allreduce_sum: a payload must not be null");
    }
    if (bytes == 0 || bytes % 16 != 0) {
        throw std::invalid_argument("allreduce_sum: bytes must be whole 16-byte vectors");
    }
    if (bytes > mailbox.max_bytes()) {
        throw std::invalid_argument("allreduce_sum: payload exceeds the mailbox's largest slot");
    }

    const int slot = mailbox.take_capture_slot(bytes);
    for (int rank = 0; rank < 2; ++rank) {
        const int device = ec.device(rank).device().index();
        (void)with_device(device, [&] {
            mailbox.enqueue_exchange_sum(rank, slot, data[rank], bytes,
                                         ec.device(rank).stream());
            return 0;
        });
    }
}

} // namespace ninfer::ops
