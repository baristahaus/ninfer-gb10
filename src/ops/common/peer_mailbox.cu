// SPDX-FileCopyrightText: ValerioDolci/capy-flash
// SPDX-License-Identifier: Apache-2.0
// Implements: include/ninfer/ops/peer_mailbox.h
//
// Ported from capy-flash src/ops/common/peer_mailbox.cu; see the header for the adaptation note.
//
// The pinned host slab layout of each slot set, in one allocation per set (the wide set has no
// hang word):
//
//   [ payload rank0 slot0 .. slotN-1 ][ payload rank1 slot0 .. slotN-1 ]
//   [ flags rank0 slot0 lane0..L-1 .. slotN-1 ][ flags rank1 ... ][ hang word ]
//
// Payload slots are 256-byte aligned (the vectorized exchange reads and writes 16-byte units;
// the headroom also keeps a slot's stores on distinct cache lines). Every release flag has a
// 64-byte line of its own (kPeerFlagStride): L per slot and rank, one per warp lane of the
// pipelined kernel at the widest payload; the original kernel uses each slot's first. Flag words
// sit after the payload so a payload overflow from a mis-sized slot cannot reach them without
// being loudly out of contract, and the hang word is last, on its own line.
//
// Each rank's arrival counters and epochs live in that rank's device memory: only that rank's
// kernels touch them, and the epoch is read at the start of every exchange, where a pinned host
// read would add a PCIe round trip.

#include "ninfer/ops/peer_mailbox.h"

#include "ops/kernel/peer_exchange.cuh" // detail::peer_exchange_sum_kernel

#include "core/collective_audit.h"
#include "core/device.h" // CUDA_CHECK

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <ctime>
#include <cstdio>
#include <mutex>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <atomic>
#include <vector>
#include <string>

namespace ninfer::runtime::detail {
// Defined in the executor (src/runtime/executor/local_executor.h): 1 while a rank's host thread is
// inside program().progress(). Declared here so a device-side give-up can report whether the
// starved rank's host is still enqueueing -- blocked in a launch -- or has returned to its caller.
extern std::atomic<int> g_progress_active[2];
} // namespace ninfer::runtime::detail

namespace ninfer::ops {
namespace { void record_launch_time(const std::timespec&, const std::timespec&); }
namespace { void note_mailbox_shape(int, int, int, int); }


namespace {

constexpr std::size_t kSlotAlignment = 256;

std::size_t aligned(std::size_t bytes) {
    return (bytes + kSlotAlignment - 1) & ~(kSlotAlignment - 1);
}

// Current-device save/restore for the phases that need a rank's device current.
class ScopedDevice {
public:
    ScopedDevice() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    ~ScopedDevice() { (void)cudaSetDevice(previous_); }

    ScopedDevice(const ScopedDevice&)            = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

    static void set(int device) { CUDA_CHECK(cudaSetDevice(device)); }

private:
    int previous_ = 0;
};

void require_two_devices(const ExecutionContext& ec, const char* message) {
    if (ec.width() != 2 || ec.device(0).device() == ec.device(1).device()) {
        throw std::invalid_argument(message);
    }
}

void throw_on_error(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("PeerMailbox: ") + what + " failed: " +
                                 cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
    }
}

} // namespace

const char* peer_exchange_kernel_name(PeerExchangeKernel kernel) noexcept {
    return kernel == PeerExchangeKernel::Legacy ? "legacy" : "pipelined";
}

PeerMailbox::PeerMailbox(const ExecutionContext& ec, std::size_t slot_bytes, int slots,
                         PeerExchangeKernel kernel, std::size_t wide_slot_bytes)
    : slots_(slots), kernel_(kernel), trace_(std::getenv("NINFER_MAILBOX_TRACE") != nullptr) {
    require_two_devices(ec, "PeerMailbox: requires an ExecutionContext with two distinct devices");
    if (slots < 2) { throw std::invalid_argument("PeerMailbox: requires at least two slots"); }
    if (slot_bytes == 0) { throw std::invalid_argument("PeerMailbox: slot bytes must be nonzero"); }

    const int pair[2] = {ec.device(0).device().index(), ec.device(1).device().index()};
    devices_[0]       = pair[0];
    devices_[1]       = pair[1];

    const std::size_t set_bytes[2] = {
        aligned(slot_bytes),
        aligned(wide_slot_bytes) > aligned(slot_bytes) ? aligned(wide_slot_bytes) : 0};
    // Everything allocates into the sets first and the destructor's release() frees whatever was
    // allocated: a throwing constructor does not run the destructor, so it releases itself.
    const ScopedDevice scope;
    try {
        for (int index = 0; index < 2; ++index) {
            if (set_bytes[index] == 0) { continue; }
            SlotSet& set   = sets_[index];
            set.slot_bytes = set_bytes[index];
            set.lanes      = detail::peer_pipelined_warps(set.slot_bytes / sizeof(detail::PeerVec),
                                                          detail::kPeerPipelinedVecsPerLane);
            const std::size_t payload_bytes = static_cast<std::size_t>(slots_) * set.slot_bytes * 2;
            const std::size_t flag_words    = static_cast<std::size_t>(slots_) *
                                           static_cast<std::size_t>(set.lanes) *
                                           detail::kPeerFlagStride;
            // Both ranks' flag lines, then (ordinary set) the hang word's line.
            const std::size_t words_bytes =
                (2 * flag_words + (index == 0 ? detail::kPeerFlagStride : 0)) *
                sizeof(std::uint32_t);
            const std::size_t legacy_words    = static_cast<std::size_t>(slots_) * 2;
            const std::size_t pipelined_words = static_cast<std::size_t>(slots_) * set.lanes;
            const std::size_t device_words_bytes =
                (legacy_words > pipelined_words ? legacy_words : pipelined_words) *
                sizeof(std::uint32_t);
            throw_on_error(cudaHostAlloc(&set.slab, payload_bytes + words_bytes,
                                         cudaHostAllocMapped | cudaHostAllocPortable),
                           "cudaHostAlloc");
            std::memset(set.slab, 0, payload_bytes + words_bytes);
            auto* base     = static_cast<std::uint8_t*>(set.slab);
            set.payload[0] = base;
            set.payload[1] = base + static_cast<std::size_t>(slots_) * set.slot_bytes;
            auto* flags    = reinterpret_cast<std::uint32_t*>(base + payload_bytes);
            set.flags[0]   = flags;
            set.flags[1]   = flags + flag_words;
            if (index == 0) { hang_ = flags + 2 * flag_words; }
            // KNOWN LIMITATION, MEASURED: this instrument cannot be armed everywhere. On hephaestus
            // (RTX 5070 Ti, driver of 2026-11) cudaHostRegister(mapped, bytes, cudaHostRegisterMapped)
            // rejects a file-backed mmap with cudaErrorInvalidValue: invalid argument, at both the
            // 32-byte size the four words need and a page-rounded 4096-byte size, so construction
            // throws before any kernel runs and the four words stay zero. A zero status word on such
            // a host means THE INSTRUMENT WAS NEVER ARMED, not that the kernel never reached the
            // spin; check that the file exists and that the entry counter moved before reading its
            // silence as evidence. The earlier reading (entry=129, target=65, last_seen=0) cannot
            // have come from an environment where this constructor throws, so its provenance is
            // unknown and it is not established fact.
            if (index == 0 && std::getenv("NINFER_STATUS_FILE") != nullptr) {
                // THE FILE-BACKED STATUS WORD. Four words per rank -- entry counter, published
                // target, last peer value seen, exit counter -- in a MAP_SHARED file that is
                // page-locked so the kernel can write it and a SECOND PROCESS can read it while the
                // stream is hung. Pinned host memory alone is invisible to another process, and the
                // in-process dump path only runs when a host mailbox call follows a hang, which is
                // exactly what does not happen when the round hangs with no give-up.
                const char* path = std::getenv("NINFER_STATUS_FILE");
                const int fd     = ::open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
                if (fd < 0) {
                    throw std::runtime_error(std::string("mailbox: cannot open the status file ") +
                                             path);
                }
                const std::size_t bytes = 2 * 4 * sizeof(std::uint32_t);
                if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
                    ::close(fd);
                    throw std::runtime_error("mailbox: cannot size the status file");
                }
                void* mapped = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                ::close(fd);
                if (mapped == MAP_FAILED) {
                    throw std::runtime_error("mailbox: cannot map the status file");
                }
                throw_on_error(cudaHostRegister(mapped, bytes, cudaHostRegisterMapped),
                               "cudaHostRegister(status)");
                std::memset(mapped, 0, bytes);
                status_               = static_cast<std::uint32_t*>(mapped);
                status_mapping_       = mapped;
                status_mapping_bytes_ = bytes;
            }
            for (int rank = 0; rank < 2; ++rank) {
                ScopedDevice::set(pair[rank]);
                void* words = nullptr;
                throw_on_error(cudaMalloc(&words, device_words_bytes), "cudaMalloc");
                set.device_words[rank] = static_cast<std::uint32_t*>(words);
                // The legacy default stream does not order the Program's non-blocking streams, so
                // the zeroing is retired here rather than left for the first captured exchange to
                // race.
                throw_on_error(cudaMemsetAsync(words, 0, device_words_bytes, nullptr),
                               "cudaMemsetAsync");
                throw_on_error(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize");
            }
        }
        for (int rank = 0; rank < 2; ++rank) {
            ScopedDevice::set(pair[rank]);
            // Loads the exchange kernel on this device now: a module cannot be loaded inside the
            // capture that first launches it.
            cudaFuncAttributes attributes{};
            constexpr int kVecsPerLane = detail::kPeerPipelinedVecsPerLane;
            throw_on_error(
                kernel_ == PeerExchangeKernel::Legacy
                    ? cudaFuncGetAttributes(&attributes, detail::peer_exchange_sum_kernel)
                    : cudaFuncGetAttributes(&attributes,
                                            detail::peer_exchange_pipelined_kernel<kVecsPerLane>),
                "cudaFuncGetAttributes");
            if (kernel_ == PeerExchangeKernel::Pipelined) {
                throw_on_error(
                    cudaFuncGetAttributes(&attributes,
                                          detail::peer_exchange_pipelined_f32_kernel<kVecsPerLane>),
                    "cudaFuncGetAttributes");
            }
        }
    } catch (...) {
        release();
        throw;
    }
}

void PeerMailbox::release() noexcept {
    for (SlotSet& set : sets_) {
        for (int rank = 0; rank < 2; ++rank) {
            if (set.device_words[rank] == nullptr) { continue; }
            (void)cudaSetDevice(devices_[rank]);
            (void)cudaFree(set.device_words[rank]);
            set.device_words[rank] = nullptr;
        }
        if (set.slab != nullptr) { (void)cudaFreeHost(set.slab); }
        set.slab = nullptr;
    }
}

PeerMailbox::~PeerMailbox() {
    const ScopedDevice scope;
    // The status file's mapping is released after the devices have retired their launches, so the
    // kernel cannot write into an unmapped range.
    if (status_mapping_ != nullptr) {
        (void)cudaHostUnregister(status_mapping_);
        (void)::munmap(status_mapping_, status_mapping_bytes_);
        status_mapping_       = nullptr;
        status_mapping_bytes_ = 0;
        status_               = nullptr;
    }
    // The owner destroys every graph executable first, but a launch may still be in flight: the
    // pinned slabs and the counters are released only once both devices retired it.
    for (int rank = 0; rank < 2; ++rank) {
        if (cudaSetDevice(devices_[rank]) != cudaSuccess) { continue; }
        const cudaError_t status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during cudaDeviceSynchronize: %s: %s\n",
                         cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }
    release();
}

bool PeerMailbox::serves(const ExecutionContext& ec) const noexcept {
    if (ec.width() != 2) { return false; }
    return ec.device(0).device().index() == devices_[0] &&
           ec.device(1).device().index() == devices_[1];
}

int PeerMailbox::take_capture_slot(std::size_t bytes) noexcept {
    SlotSet& set   = sets_[set_for(bytes)];
    const int slot = set.next_slot;
    set.next_slot  = (set.next_slot + 1) % slots_;
    return slot;
}

int PeerMailbox::slot_for(std::size_t index, std::size_t bytes) const noexcept {
    // The set is chosen for its slot width, and both sets have the same slot count, so the index
    // alone decides the slot; the parameter is accepted so a caller reads it as the same question
    // take_capture_slot() answers.
    (void)bytes;
    return static_cast<int>(index % static_cast<std::size_t>(slots_));
}

namespace {
detail::PeerPipelinedTuning pipelined_tuning(bool release_dependents) {
    detail::PeerPipelinedTuning tuning{};
    tuning.release_dependents = release_dependents;
    return tuning;
}
} // namespace

const void* PeerMailbox::payload_base(int rank) const noexcept {
    if (rank < 0 || rank > 1) { return nullptr; }
    return sets_[0].payload[rank];
}

void PeerMailbox::set_l2_prefetch(int rank, std::span<const PeerL2Range> ranges) {
    if (rank < 0 || rank > 1 || ranges.size() > static_cast<std::size_t>(kL2PrefetchRanges)) {
        throw std::invalid_argument("PeerMailbox: rank or L2 prefetch range count out of range");
    }
    L2Hint hint{};
    for (const PeerL2Range& range : ranges) {
        const auto begin           = reinterpret_cast<std::uintptr_t>(range.data);
        const std::uintptr_t first = (begin + 15u) & ~std::uintptr_t{15};
        const std::uintptr_t last  = (begin + range.bytes) & ~std::uintptr_t{15};
        if (range.data == nullptr || last <= first) { continue; }
        const std::size_t bytes   = std::min<std::size_t>(last - first, std::size_t{1} << 30);
        hint.ranges[hint.count++] = {reinterpret_cast<const void*>(first), bytes};
    }
    l2_hint_[rank] = hint;
}

namespace {
// Takes rank `rank`'s pending hint into `tuning` behind `blocks` exchange blocks; returns the
// prefetching blocks to add to the grid.
template <class Hint>
int take_l2_hint(Hint& hint, int blocks, int threads, detail::PeerPipelinedTuning& tuning) {
    detail::PeerL2Prefetch& prefetch = tuning.prefetch;
    for (int r = 0; r < hint.count; ++r) {
        prefetch.data[r]  = hint.ranges[r].data;
        prefetch.bytes[r] = static_cast<std::uint32_t>(hint.ranges[r].bytes);
    }
    prefetch.ranges      = hint.count;
    prefetch.first_block = blocks;
    hint.count           = 0;
    const int extra      = detail::peer_prefetch_blocks(prefetch, threads);
    if (extra == 0) { prefetch.ranges = 0; }
    return extra;
}
} // namespace



void PeerMailbox::require_no_hang(const char* call) const {
    // The guard's word is sticky by design: once one poller abandons an exchange, that rank's
    // partial holds only its own operand and its later waits return immediately, so every value it
    // produces from here on is wrong. A wrong number nobody can see is the worst outcome in the
    // lane, so the next mailbox call stops the round instead of continuing to compute.
    if (!hang_reported()) { return; }
    throw std::runtime_error(
        std::string("mailbox: a poller gave up in an earlier exchange of this round, so the ranks' "
                    "results have diverged and ") +
        call + " cannot continue; see the kernel's [mailbox] GAVE UP line for the rank, warp and "
               "probe count");
}

void PeerMailbox::enqueue_exchange_sum(int rank, int slot, void* data, std::size_t bytes,
                                       cudaStream_t stream, const char* site) const {
    require_no_hang("enqueue_exchange_sum");
    if (rank < 0 || rank > 1 || slot < 0 || slot >= slots_) {
        throw std::invalid_argument("PeerMailbox: rank or slot out of range");
    }
    const SlotSet& set = sets_[set_for(bytes)];
    if (bytes == 0 || bytes > set.slot_bytes || (bytes % sizeof(detail::PeerVec)) != 0 ||
        (reinterpret_cast<std::uintptr_t>(data) % sizeof(detail::PeerVec)) != 0) {
        throw std::invalid_argument(
            "PeerMailbox: an exchange carries whole, aligned 16-byte vectors within one slot");
    }
    const std::size_t offset = static_cast<std::size_t>(slot) * set.slot_bytes;
    const std::size_t lane_base =
        static_cast<std::size_t>(slot) * static_cast<std::size_t>(set.lanes);
    const std::size_t flag_slot = lane_base * detail::kPeerFlagStride;
    ::ninfer::core::audit::record(rank, site);
    const std::size_t vecs      = bytes / sizeof(detail::PeerVec);
    std::uint32_t* words        = set.device_words[rank];
    auto* partial               = static_cast<detail::PeerVecBf16*>(data);
    auto* mine_payload = reinterpret_cast<detail::PeerVecBf16*>(set.payload[rank] + offset);
    const auto* peer_payload =
        reinterpret_cast<const detail::PeerVecBf16*>(set.payload[1 - rank] + offset);
    if (kernel_ == PeerExchangeKernel::Legacy) {
        l2_hint_[rank].count = 0; // the original kernel has no prefetching blocks
        detail::peer_exchange_sum_kernel<<<detail::peer_exchange_blocks(bytes), 256, 0, stream>>>(
            partial, mine_payload, set.flags[rank] + flag_slot, peer_payload,
            set.flags[1 - rank] + flag_slot, words + slots_ + slot, words + slot, hang_,
            static_cast<int>(vecs));
    } else {
        constexpr int kVecsPerLane = detail::kPeerPipelinedVecsPerLane;
        constexpr int kThreads     = detail::kPeerPipelinedThreads;
        const int blocks           = detail::peer_pipelined_blocks(vecs, kVecsPerLane, kThreads);
        auto tuning                = pipelined_tuning(release_dependents_);
        tuning.status              = status_ != nullptr ? status_ + 4 * rank : nullptr;
        tuning.rank                = rank;
        tuning.slot                = slot;
        // This rank's FIRST exchange is the one place where a healthy peer can be legitimately late:
        // the two Programs start independently, so the peer may not have launched its round yet. The
        // steady-state budget is a dead-peer detector and would abandon the exchange (leaving a
        // partial that holds only this rank's operand, and a sticky word that disables every later
        // wait in the round), so the first exchange gets the orders-of-magnitude-larger budget. See
        // kPeerFirstSpinLimit.
        if (!started_[rank]) {
            started_[rank]    = true;
            tuning.spin_limit = detail::kPeerFirstSpinLimit;
            // The first exchange is also the one the residual dump brackets, so it is the one worth
            // dumping: one print per rank keeps the output readable.
            if (trace_) { tuning.trace = true; }
        }
        const int prefetch_blocks  = take_l2_hint(l2_hint_[rank], blocks, kThreads, tuning);
        // THE SHAPE, FROM THE HOST, WHERE IT CANNOT BE LOST. A device-side printf is buffered and
        // flushed at sync, and the give-up kills the process before that happens, which is why rank
        // 1's shape never appeared. This is written to unbuffered stderr at the enqueue site, once
        // per distinct (rank, vecs, blocks, warps) shape -- a diagnostic whose output is
        // proportional to the number of calls cannot be read.
        note_mailbox_shape(rank, vecs, blocks + prefetch_blocks, kThreads);
        // THE ENGINE'S OWN CLOCK AROUND THE LAUNCH. Every other number in this workstream came from
        // nsys or a harness; this is the engine timing itself, so it can be compared with the
        // harness's 1.2 us tight loop on the same terms. Percentiles are printed because the
        // question is whether the 128 us median was a real median or a profiler-inflated tail.
        std::timespec launch_t0{}, launch_t1{};
        (void)::clock_gettime(CLOCK_MONOTONIC, &launch_t0);
        detail::peer_exchange_pipelined_kernel<kVecsPerLane>
            <<<blocks + prefetch_blocks, kThreads, 0, stream>>>(
                partial, mine_payload, set.flags[rank] + flag_slot, peer_payload,
                set.flags[1 - rank] + flag_slot, words + lane_base, hang_, static_cast<int>(vecs),
                tuning);
        (void)::clock_gettime(CLOCK_MONOTONIC, &launch_t1);
        record_launch_time(launch_t0, launch_t1);
    }
    CUDA_CHECK(cudaGetLastError());
}

namespace {
// Launch-time histogram for the mailbox kernel, in microseconds. Static because the measurement is
// per process, and printed once at exit so the run's distribution is available without a profiler.
struct LaunchTimes {
    std::mutex mutex;
    std::vector<double> us;
    void add(double v) {
        std::lock_guard<std::mutex> lock(mutex);
        us.push_back(v);
    }
    ~LaunchTimes() {
        if (us.empty()) {
            std::fprintf(stderr, "[mailbox-launch] no samples\n");
            return;
        }
        std::sort(us.begin(), us.end());
        const auto at = [&](double q) { return us[(std::size_t)(q * (us.size() - 1))]; };
        std::size_t above = 0;
        double total = 0;
        for (double v : us) { total += v; if (v > 100.0) ++above; }
        std::fprintf(stderr,
                     "[mailbox-launch] n=%zu p50=%.2f p90=%.2f p99=%.2f max=%.2f us above100=%zu "
                     "total=%.1f ms\n",
                     us.size(), at(0.50), at(0.90), at(0.99), us.back(), above, total / 1000.0);
    }
};
LaunchTimes& launch_times() {
    static LaunchTimes times;
    return times;
}
void record_launch_time(const std::timespec& t0, const std::timespec& t1) {
    const double us = (double)(t1.tv_sec - t0.tv_sec) * 1e6 + (double)(t1.tv_nsec - t0.tv_nsec) / 1e3;
    launch_times().add(us);
}
} // namespace

namespace {
// One line per distinct launch shape, per rank. The give-up is always at a high warp number, so the
// question is whether the other rank's grid even HAS that warp: a warp that exists on one side and
// not the other spins forever on a publish no thread will make.
void note_mailbox_shape(int rank, int vecs, int blocks, int threads) {
    struct Key {
        int rank, vecs, blocks, threads;
        bool operator==(const Key& o) const {
            return rank == o.rank && vecs == o.vecs && blocks == o.blocks && threads == o.threads;
        }
    };
    static std::mutex mutex;
    static std::vector<Key> seen;
    const Key key{rank, vecs, blocks, threads};
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (std::find(seen.begin(), seen.end(), key) != seen.end()) { return; }
        seen.push_back(key);
    }
    const int warps = (blocks * threads) / 32;
    std::fprintf(stderr, "[mailbox-shape] rank=%d vecs=%d blocks=%d threads=%d warps=%d\n", rank,
                 vecs, blocks, threads, warps);
    std::fflush(stderr);
}
} // namespace

void PeerMailbox::enqueue_exchange_sum_f32(int rank, int slot, void* data, std::size_t bytes,
                                           void* bf16_out, cudaStream_t stream) const {
    require_no_hang("enqueue_exchange_sum_f32");
    if (kernel_ != PeerExchangeKernel::Pipelined) {
        throw std::logic_error("PeerMailbox: the FP32 exchange needs the pipelined kernel");
    }
    if (rank < 0 || rank > 1 || slot < 0 || slot >= slots_) {
        throw std::invalid_argument("PeerMailbox: rank or slot out of range");
    }
    const SlotSet& set = sets_[set_for(bytes)];
    if (bytes == 0 || bytes > set.slot_bytes || (bytes % sizeof(detail::PeerVec)) != 0 ||
        (reinterpret_cast<std::uintptr_t>(data) % sizeof(detail::PeerVec)) != 0 ||
        (reinterpret_cast<std::uintptr_t>(bf16_out) % sizeof(detail::PeerVec)) != 0) {
        throw std::invalid_argument(
            "PeerMailbox: an exchange carries whole, aligned 16-byte vectors within one slot");
    }
    const std::size_t offset = static_cast<std::size_t>(slot) * set.slot_bytes;
    const std::size_t lane_base =
        static_cast<std::size_t>(slot) * static_cast<std::size_t>(set.lanes);
    const std::size_t flag_slot = lane_base * detail::kPeerFlagStride;
    ::ninfer::core::audit::record(rank, "f32");
    const std::size_t vecs      = bytes / sizeof(detail::PeerVec);
    constexpr int kVecsPerLane  = detail::kPeerPipelinedVecsPerLane;
    constexpr int kThreads      = detail::kPeerPipelinedThreads;
    const int blocks            = detail::peer_pipelined_blocks(vecs, kVecsPerLane, kThreads);
    auto tuning                 = pipelined_tuning(release_dependents_);
    const int prefetch_blocks   = take_l2_hint(l2_hint_[rank], blocks, kThreads, tuning);
    detail::peer_exchange_pipelined_f32_kernel<kVecsPerLane>
        <<<blocks + prefetch_blocks, kThreads, 0, stream>>>(
            static_cast<detail::PeerVec*>(data),
            reinterpret_cast<detail::PeerVec*>(set.payload[rank] + offset),
            set.flags[rank] + flag_slot,
            reinterpret_cast<const detail::PeerVec*>(set.payload[1 - rank] + offset),
            set.flags[1 - rank] + flag_slot, set.device_words[rank] + lane_base, hang_,
            static_cast<int>(vecs), static_cast<__nv_bfloat16*>(bf16_out), tuning);
    CUDA_CHECK(cudaGetLastError());
}

PeerExchangeSlot PeerMailbox::exchange_slot(int rank, int slot, std::size_t bytes) const {
    require_no_hang("exchange_slot");
    if (kernel_ != PeerExchangeKernel::Pipelined) {
        throw std::logic_error("PeerMailbox: a fused exchange needs the pipelined protocol");
    }
    if (rank < 0 || rank > 1 || slot < 0 || slot >= slots_) {
        throw std::invalid_argument("PeerMailbox: rank or slot out of range");
    }
    const SlotSet& set = sets_[set_for(bytes)];
    if (bytes == 0 || bytes > set.slot_bytes || (bytes % sizeof(detail::PeerVec)) != 0) {
        throw std::invalid_argument(
            "PeerMailbox: an exchange carries whole 16-byte vectors within one slot");
    }
    const std::size_t offset = static_cast<std::size_t>(slot) * set.slot_bytes;
    const std::size_t lane_base =
        static_cast<std::size_t>(slot) * static_cast<std::size_t>(set.lanes);
    const std::size_t flag_slot = lane_base * detail::kPeerFlagStride;
    PeerExchangeSlot view;
    view.mine_payload = set.payload[rank] + offset;
    view.peer_payload = set.payload[1 - rank] + offset;
    view.mine_flags   = set.flags[rank] + flag_slot;
    view.peer_flags   = set.flags[1 - rank] + flag_slot;
    view.epochs       = set.device_words[rank] + lane_base;
    view.hang         = hang_;
    return view;
}

bool PeerMailbox::hang_reported() const noexcept {
    return *static_cast<volatile const std::uint32_t*>(hang_) != 0;
}

void PeerMailbox::report_hang() noexcept { *static_cast<volatile std::uint32_t*>(hang_) = 1u; }

} // namespace ninfer::ops
