// The pinned-host mailbox transport, exercised end to end between two devices: each rank publishes
// a BF16 payload into its own host slot, the two exchange kernels rendezvous on the mailbox flags,
// and each rank's payload becomes the elementwise sum. This is the first test that moves bytes
// through the transport rather than only compiling it.
//
// The arithmetic is checked exactly. The combine is float(mine) + float(peer) in FP32 with one
// round-to-nearest-even on store, and 1.0 + 2.0 = 3.0 is exact in both BF16 and FP32, so any
// disagreement is a transport defect and not a rounding question.
//
// Skipped when the machine has one device.

#include "core/device.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/peer_mailbox.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <span>
#include <vector>
#include <cstring>

namespace {

using ninfer::CudaDeviceScope;
using ninfer::Device;
using ninfer::ExecutionContext;
using ninfer::ops::PeerMailbox;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int device_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) { return 0; }
    return count;
}

// 640 vectors of 16 bytes: the single-token decode activation size the transport is built for.
constexpr std::size_t kBytes = 10240;
constexpr std::uint16_t kBf16One   = 0x3F80; // 1.0
constexpr std::uint16_t kBf16Two   = 0x4000; // 2.0
constexpr std::uint16_t kBf16Three = 0x4040; // 3.0

int test_exchange_sum() {
    ExecutionContext context(std::vector<int>{0, 1});
    PeerMailbox mailbox(context, kBytes);

    int failures = 0;
    failures += check(mailbox.slot_bytes() >= kBytes, "the mailbox carries a 10 KiB payload");
    failures += check(mailbox.max_bytes() >= kBytes, "max_bytes covers the payload");
    failures += check(!mailbox.hang_reported(), "a fresh mailbox reports no hang");

    void* payload[2] = {nullptr, nullptr};
    const std::uint16_t fill[2] = {kBf16One, kBf16Two};
    std::vector<std::uint16_t> host(kBytes / sizeof(std::uint16_t));
    for (int rank = 0; rank < 2; ++rank) {
        const CudaDeviceScope scope(Device::cuda(rank));
        std::fill(host.begin(), host.end(), fill[rank]);
        cudaError_t status = cudaMalloc(&payload[rank], kBytes);
        failures += check(status == cudaSuccess, "payload allocation on rank " + std::to_string(rank));
        if (status != cudaSuccess) { return failures; }
        status = cudaMemcpy(payload[rank], host.data(), kBytes, cudaMemcpyHostToDevice);
        failures += check(status == cudaSuccess, "payload fill on rank " + std::to_string(rank));
    }

    // One slot for both ranks, then both halves enqueued; the kernels meet on the mailbox flags.
    const int slot = mailbox.take_capture_slot(kBytes);
    for (int rank = 0; rank < 2; ++rank) {
        const CudaDeviceScope scope(Device::cuda(rank));
        mailbox.enqueue_exchange_sum(rank, slot, payload[rank], kBytes,
                                     context.device(rank).stream());
    }
    for (int rank = 0; rank < 2; ++rank) {
        const CudaDeviceScope scope(Device::cuda(rank));
        failures += check(cudaDeviceSynchronize() == cudaSuccess,
                          "rank " + std::to_string(rank) + " synchronized");
    }

    failures += check(!mailbox.hang_reported(),
                      "no poller gave up: both ranks met on the mailbox");

    // Every element of both payloads must be 1.0 + 2.0.
    for (int rank = 0; rank < 2; ++rank) {
        const CudaDeviceScope scope(Device::cuda(rank));
        std::fill(host.begin(), host.end(), 0);
        const cudaError_t status =
            cudaMemcpy(host.data(), payload[rank], kBytes, cudaMemcpyDeviceToHost);
        failures += check(status == cudaSuccess, "read back rank " + std::to_string(rank));
        if (status != cudaSuccess) { continue; }
        std::size_t wrong = 0;
        for (const std::uint16_t value : host) {
            if (value != kBf16Three) { ++wrong; }
        }
        failures += check(wrong == 0, "rank " + std::to_string(rank) + " holds the elementwise sum (" +
                                          std::to_string(wrong) + " of " +
                                          std::to_string(host.size()) + " elements differ)");
    }

    for (int rank = 0; rank < 2; ++rank) {
        const CudaDeviceScope scope(Device::cuda(rank));
        (void)cudaFree(payload[rank]);
    }
    return failures;
}

// The all-reduce seam: same arithmetic as the raw exchange, through the function a Program calls.
int test_allreduce_sum() {
    ExecutionContext context(std::vector<int>{0, 1});
    PeerMailbox mailbox(context, kBytes);
    int failures = 0;

    // Reported rather than asserted: peer access depends on the driver and the board, and this
    // transport does not need it.
    std::cout << "peer access available: " << (ninfer::ops::peer_access_available(context) ? "yes" : "no")
              << '\n';

    void* payload[2] = {nullptr, nullptr};
    const std::uint16_t fill[2] = {kBf16One, kBf16Two};
    std::vector<std::uint16_t> host(kBytes / sizeof(std::uint16_t));
    for (int rank = 0; rank < 2; ++rank) {
        const CudaDeviceScope scope(Device::cuda(rank));
        std::fill(host.begin(), host.end(), fill[rank]);
        if (cudaMalloc(&payload[rank], kBytes) != cudaSuccess) {
            return check(false, "allreduce payload allocation failed");
        }
        (void)cudaMemcpy(payload[rank], host.data(), kBytes, cudaMemcpyHostToDevice);
    }

    ninfer::ops::allreduce_sum(context, mailbox, std::span<void* const>(payload, 2), kBytes);
    for (int rank = 0; rank < 2; ++rank) {
        const CudaDeviceScope scope(Device::cuda(rank));
        failures += check(cudaDeviceSynchronize() == cudaSuccess,
                          "allreduce rank " + std::to_string(rank) + " synchronized");
    }
    failures += check(!mailbox.hang_reported(), "allreduce: no poller gave up");

    for (int rank = 0; rank < 2; ++rank) {
        const CudaDeviceScope scope(Device::cuda(rank));
        std::fill(host.begin(), host.end(), 0);
        (void)cudaMemcpy(host.data(), payload[rank], kBytes, cudaMemcpyDeviceToHost);
        std::size_t wrong = 0;
        for (const std::uint16_t value : host) { if (value != kBf16Three) { ++wrong; } }
        failures += check(wrong == 0, "allreduce rank " + std::to_string(rank) +
                                          " holds the elementwise sum");
        (void)cudaFree(payload[rank]);
    }
    return failures;
}

// The per-rank form: each rank's Program calls it once for its own rank, in a slot both ranks
// derive the same way. This is the form the two-Program lane uses, and it must produce the same
// bytes as the single-owner form.
int test_allreduce_sum_rank() {
    ExecutionContext context(std::vector<int>{0, 1});
    PeerMailbox mailbox(context, kBytes);
    int failures = 0;

    void* payload[2] = {nullptr, nullptr};
    const std::uint16_t fill[2] = {kBf16One, kBf16Two};
    std::vector<std::uint16_t> host(kBytes / sizeof(std::uint16_t));
    for (int rank = 0; rank < 2; ++rank) {
        const CudaDeviceScope scope(Device::cuda(rank));
        std::fill(host.begin(), host.end(), fill[rank]);
        if (cudaMalloc(&payload[rank], kBytes) != cudaSuccess) {
            return check(false, "per-rank allreduce payload allocation failed");
        }
        (void)cudaMemcpy(payload[rank], host.data(), kBytes, cudaMemcpyHostToDevice);
    }

    // Both ranks derive the same slot from the same index; neither advances a shared counter.
    const int slot = mailbox.slot_for(0, kBytes);
    ninfer::ops::allreduce_sum_rank(context, mailbox, 0, slot,
                                    std::span<void* const>(payload, 2), kBytes);
    ninfer::ops::allreduce_sum_rank(context, mailbox, 1, slot,
                                    std::span<void* const>(payload, 2), kBytes);
    for (int rank = 0; rank < 2; ++rank) {
        const CudaDeviceScope scope(Device::cuda(rank));
        failures += check(cudaDeviceSynchronize() == cudaSuccess,
                          "per-rank allreduce rank " + std::to_string(rank) + " synchronized");
    }
    failures += check(!mailbox.hang_reported(), "per-rank allreduce: no poller gave up");

    for (int rank = 0; rank < 2; ++rank) {
        const CudaDeviceScope scope(Device::cuda(rank));
        std::fill(host.begin(), host.end(), 0);
        (void)cudaMemcpy(host.data(), payload[rank], kBytes, cudaMemcpyDeviceToHost);
        std::size_t wrong = 0;
        for (const std::uint16_t value : host) { if (value != kBf16Three) { ++wrong; } }
        failures += check(wrong == 0, "per-rank allreduce rank " + std::to_string(rank) +
                                          " holds the elementwise sum");
        (void)cudaFree(payload[rank]);
    }
    return failures;
}


// ---- stress: the lane's reduce pattern, with self-describing payloads ---------------------------
//
// One 27B round issues 4096 column-parallel reduces back to back with NO synchronization between
// them, alternating two slots, each carrying a different payload. The three tests above cannot see a
// race in that pattern: each synchronizes after a single exchange and carries a constant fill, so a
// stale or half-received payload still produces the expected sum. This test reproduces the pattern
// and makes every received vector self-describing, so a mismatch NAMES the exchange the payload
// actually came from instead of only failing a comparison.
//
// Tag: value = (exchange % 32) * 2 + (vector index & 1) + 1, plus 100 for rank 1. Every value is a
// small integer, exact in BF16, and exact through the FP32 combine and its one round-to-nearest
// store, so the expected sum 2 * tag + 100 is exact too. Consecutive exchanges differ by 2, so a
// payload that is one exchange stale is identifiable from the value alone.
constexpr std::size_t kStressBytes   = 60 * 1024; // the width-2 decode projection reduce payload
constexpr int kStressExchanges       = 1024;      // raise for a longer window; 4096 is one round
constexpr int kStressRankOffset      = 100;

int stress_tag(int exchange, int vector_index) {
    return (exchange % 32) * 2 + (vector_index & 1) + 1;
}

std::uint16_t bf16_bits_of(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

float bf16_value_of(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float value              = 0.0f;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

int test_exchange_stress() {
    ExecutionContext context(std::vector<int>{0, 1});
    PeerMailbox mailbox(context, kStressBytes);
    int failures = 0;
    const int vectors = static_cast<int>(kStressBytes / sizeof(uint4));

    std::vector<void*> payload[2];
    std::vector<std::uint16_t> host(kStressBytes / sizeof(std::uint16_t));
    for (int rank = 0; rank < 2; ++rank) {
        const CudaDeviceScope scope(Device::cuda(rank));
        payload[rank].resize(kStressExchanges);
        for (int k = 0; k < kStressExchanges; ++k) {
            if (cudaMalloc(&payload[rank][k], kStressBytes) != cudaSuccess) {
                return check(false, "stress payload allocation failed");
            }
            for (int i = 0; i < vectors * 8; ++i) {
                const int tag = stress_tag(k, i / 8) + rank * kStressRankOffset;
                host[static_cast<std::size_t>(i)] = bf16_bits_of(static_cast<float>(tag));
            }
            (void)cudaMemcpy(payload[rank][k], host.data(), kStressBytes, cudaMemcpyHostToDevice);
        }
    }

    // No synchronization between exchanges: this is the ordering the lane relies on, and the point of
    // the test is that the transport must not need one.
    for (int k = 0; k < kStressExchanges; ++k) {
        const int slot = mailbox.slot_for(static_cast<std::size_t>(k), kStressBytes);
        for (int rank = 0; rank < 2; ++rank) {
            const CudaDeviceScope scope(Device::cuda(rank));
            mailbox.enqueue_exchange_sum(rank, slot, payload[rank][k], kStressBytes,
                                         context.device(rank).stream());
        }
    }
    for (int rank = 0; rank < 2; ++rank) {
        const CudaDeviceScope scope(Device::cuda(rank));
        failures += check(cudaDeviceSynchronize() == cudaSuccess,
                          "stress rank " + std::to_string(rank) + " synchronized");
    }
    failures += check(!mailbox.hang_reported(), "stress: no poller gave up");

    // Verify every vector of every exchange, and diagnose the first few mismatches by asking which
    // exchange the value actually came from.
    std::size_t mismatches = 0;
    int reported           = 0;
    for (int rank = 0; rank < 2; ++rank) {
        const CudaDeviceScope scope(Device::cuda(rank));
        for (int k = 0; k < kStressExchanges; ++k) {
            std::fill(host.begin(), host.end(), 0);
            if (cudaMemcpy(host.data(), payload[rank][k], kStressBytes,
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                failures += check(false, "stress read back failed");
                return failures;
            }
            for (int i = 0; i < vectors * 8; ++i) {
                const int vector_index = i / 8;
                const int expected =
                    2 * stress_tag(k, vector_index) + kStressRankOffset;
                const float actual_value = bf16_value_of(host[static_cast<std::size_t>(i)]);
                if (static_cast<int>(actual_value) == expected) { continue; }
                ++mismatches;
                if (reported >= 8) { continue; }
                ++reported;
                // Which exchange could this value have come from? A stale payload is the common
                // failure and it is worth naming rather than leaving as a number.
                std::string origin = "unrecognised";
                for (int candidate = k - 4; candidate <= k + 4; ++candidate) {
                    if (candidate < 0 || candidate == k) { continue; }
                    const int tag = stress_tag(candidate, vector_index);
                    if (static_cast<int>(actual_value) == 2 * tag + kStressRankOffset) {
                        origin = "exchange " + std::to_string(candidate) + " (stale by " +
                                 std::to_string(k - candidate) + ")";
                        break;
                    }
                    if (static_cast<int>(actual_value) == tag + rank * kStressRankOffset) {
                        origin = "the rank's OWN unpublished payload from exchange " +
                                 std::to_string(candidate);
                        break;
                    }
                }
                std::cerr << "  stress mismatch: rank " << rank << " exchange " << k << " vector "
                          << vector_index << " element " << i << " expected " << expected
                          << " got " << actual_value << " -> " << origin << '\n';
            }
        }
    }
    failures += check(mismatches == 0,
                      "the mailbox carried every vector of every exchange (" +
                          std::to_string(mismatches) + " mismatched of " +
                          std::to_string(std::size_t{2} * kStressExchanges * kStressBytes / 2) +
                          " elements)");

    for (int rank = 0; rank < 2; ++rank) {
        const CudaDeviceScope scope(Device::cuda(rank));
        for (void* buffer : payload[rank]) { (void)cudaFree(buffer); }
    }
    return failures;
}

} // namespace

int main() {
    const int devices = device_count();
    if (devices < 2) {
        // This test exercises only the two-device transport, so with fewer devices it skips
        // entirely: exit 77 so ctest reports a skip rather than a pass.
        std::cerr << "SKIP: two CUDA devices are required for the mailbox transport; fewer are present. Set CUDA_VISIBLE_DEVICES to two devices, for example 0,1\n";
        return 77;
    }
    int failures = 0;
    try {
        failures += test_exchange_sum();
        failures += test_allreduce_sum();
        failures += test_allreduce_sum_rank();
        failures += test_exchange_stress();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: threw: " << e.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
