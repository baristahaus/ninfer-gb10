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

#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

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
                                       cudaStream_t stream) const {
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
        detail::peer_exchange_pipelined_kernel<kVecsPerLane>
            <<<blocks + prefetch_blocks, kThreads, 0, stream>>>(
                partial, mine_payload, set.flags[rank] + flag_slot, peer_payload,
                set.flags[1 - rank] + flag_slot, words + lane_base, hang_, static_cast<int>(vecs),
                tuning);
    }
    CUDA_CHECK(cudaGetLastError());
}

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
