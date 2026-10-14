#pragma once

#include <cstdint>

namespace ninfer::ops {

// SP-4/SP-5 adaptive expert cache: device-resident adaptation state. One pass of
// expert_cache_adapt promotes and evicts tail-limited warm experts directly on device —
// reading the always-armed activation counters and the residency table, swapping the
// weakest promoted experts for the hottest uncovered ones with hysteresis, copying the
// promoted blocks from the pinned-host warm mini-bank into the cache tail, and flipping the
// residency entries in place. No host readback and no synchronization: the kernel launches
// stream-ordered at the prefill chunk and decode round boundaries, and the reporting
// accumulators ride device memory until the final readback.

// Per-bank plane geometry: the codes/high/scales plane offsets and per-row bytes for the
// device mini-bank (hot experts plus the cache tail) and the pinned-host warm mini-bank.
struct ExpertCacheBankParams {
    const std::byte* device   = nullptr;
    const std::byte* warm    = nullptr;
    std::uint64_t device_code_offset  = 0;
    std::uint64_t device_high_offset  = 0;
    std::uint64_t device_scale_offset = 0;
    std::uint64_t warm_code_offset    = 0;
    std::uint64_t warm_high_offset    = 0;
    std::uint64_t warm_scale_offset   = 0;
    std::uint64_t code_bytes_per_row  = 0;
    std::uint64_t high_bytes_per_row  = 0;
    std::uint64_t scale_bytes_per_row = 0;
    std::uint64_t rows_per_expert     = 0;
};

// Per-layer adaptation parameters: the routed gate/up and down banks, the tier counts, and
// this layer's slice of the warm-rank table (experts entries; hot-tier experts carry
// ExpertCacheAdaptState::kNoWarmRank).
struct ExpertCacheLayerParams {
    ExpertCacheBankParams gate_up;
    ExpertCacheBankParams down;
    std::uint32_t hot        = 0;
    std::uint32_t warm_count = 0;
    std::uint32_t slots      = 0;
};

struct ExpertCacheAdaptState {
    static constexpr std::uint32_t kNoWarmRank = 0xffffffffu;
    // One u32 activation counter per routed expert per MoE layer.
    const std::uint32_t* counters = nullptr;
    // The previous pass's counters (the recency-delta base), updated in place per pass.
    std::uint32_t* snapshot = nullptr;
    // The residency table (bit 31 marks the warm tier; the low bits are the compacted rank),
    // flipped in place.
    std::uint32_t* residency = nullptr;
    // Per routed expert per layer: the expert's compacted rank in the warm mini-bank, or
    // kNoWarmRank for hot-tier experts.
    const std::uint32_t* warm_rank = nullptr;
    const ExpertCacheLayerParams* layers = nullptr;
    std::uint32_t layer_count = 0;
    std::uint32_t experts     = 0;
    // Device reporting accumulators, read back once at teardown.
    unsigned long long* hits       = nullptr;
    unsigned long long* totals     = nullptr;
    unsigned long long* promotions = nullptr;
    unsigned long long* evictions  = nullptr;
};

// One adaptation pass over every routed MoE layer: 256 threads per layer (one per expert).
// Tiers the tail fully covers — and untiered layers — only accumulate the warm-tier hit
// rate; tail-limited layers additionally swap at most two experts per pass with hysteresis
// (a newcomer must clear a minimum recent count and double the incumbent it evicts).
void launch_expert_cache_adapt(const ExpertCacheAdaptState& state, cudaStream_t stream);

} // namespace ninfer::ops
