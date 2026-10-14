#include "ops/sparse_moe/expert_cache_adapt.h"

#include "core/device.h"

namespace ninfer::ops {
namespace {

// A newcomer must clear this many recent selections before it can displace an incumbent.
constexpr std::uint32_t kMinimumRecentCount = 4;
// A newcomer must double the incumbent's recent count: hysteresis against swap thrashing.
constexpr std::uint32_t kEvictionMargin = 2;
// At most two swaps per layer per pass; the adaptation converges over passes.
constexpr int kMaximumSwaps = 2;
// The per-expert copy granularity: one 16-byte vector per thread iteration.
constexpr std::uint64_t kCopyChunk = 16;

// Copies one plane (codes, high, or scales) of one expert: `rows` bytes from the warm
// mini-bank at the expert's warm rank into the device mini-bank at its tail slot. The
// pinned-host source is read over PCIe exactly as the MoE kernels read the warm tier.
__device__ __forceinline__ void copy_expert_plane(const std::byte* __restrict__ device_base,
                                                   std::uint64_t device_offset,
                                                   const std::byte* __restrict__ warm_base,
                                                   std::uint64_t warm_offset,
                                                   std::uint64_t bytes, std::uint32_t tid) {
    const std::uint64_t chunk_count = (bytes + kCopyChunk - 1) / kCopyChunk;
    for (std::uint64_t chunk = tid; chunk < chunk_count; chunk += blockDim.x) {
        const std::uint64_t offset = chunk * kCopyChunk;
        const std::uint64_t remaining = bytes - offset;
        const std::uint64_t length = remaining < kCopyChunk ? remaining : kCopyChunk;
        const char* source      = reinterpret_cast<const char*>(warm_base + warm_offset + offset);
        char* destination       = reinterpret_cast<char*>(const_cast<std::byte*>(
            device_base + device_offset + offset));
        for (std::uint64_t byte = 0; byte < length; ++byte) { destination[byte] = source[byte]; }
    }
}

// Copies one expert's full slice (both banks, all planes) from the warm mini-bank into the
// cache tail slot. blockDim threads cooperate per plane.
__device__ __forceinline__ void copy_expert_slice(const ExpertCacheLayerParams& layer,
                                                   std::uint32_t warm_rank,
                                                   std::uint32_t tail_slot, std::uint32_t tid) {
    const std::uint32_t device_rank = layer.hot + tail_slot;
    for (const auto* bank : {&layer.gate_up, &layer.down}) {
        const auto plane = [&](std::uint64_t device_offset, std::uint64_t warm_offset,
                              std::uint64_t per_row) {
            if (per_row == 0) { return; }
            // One expert's plane block: rows_per_expert rows of per_row bytes each.
            const std::uint64_t expert_block = bank->rows_per_expert * per_row;
            copy_expert_plane(bank->device,
                              device_offset +
                                  static_cast<std::uint64_t>(device_rank) * expert_block,
                              bank->warm,
                              warm_offset + static_cast<std::uint64_t>(warm_rank) * expert_block,
                              expert_block, tid);
        };
        plane(bank->device_code_offset, bank->warm_code_offset, bank->code_bytes_per_row);
        plane(bank->device_high_offset, bank->warm_high_offset, bank->high_bytes_per_row);
        plane(bank->device_scale_offset, bank->warm_scale_offset, bank->scale_bytes_per_row);
    }
}

// One adaptation pass for one MoE layer: 256 threads, one per expert.
__global__ void expert_cache_adapt_kernel(ExpertCacheAdaptState state) {
    const std::uint32_t layer = static_cast<std::uint32_t>(blockIdx.x);
    if (layer >= state.layer_count) { return; }
    const ExpertCacheLayerParams& lp = state.layers[layer];
    const std::uint32_t base = layer * state.experts;
    const std::uint32_t tid  = static_cast<std::uint32_t>(threadIdx.x);

    // Phase 0: the recency delta per expert, and the warm-tier hit-rate accumulation.
    const bool warm_tier = state.warm_rank[base + tid] != ExpertCacheAdaptState::kNoWarmRank;
    const std::uint32_t count = state.counters[base + tid];
    const std::uint32_t delta = count - state.snapshot[base + tid];
    const bool promoted =
        warm_tier && (state.residency[base + tid] & 0x80000000u) == 0u;
    if (warm_tier) {
        atomicAdd(state.totals, static_cast<unsigned long long>(delta));
        if (promoted) { atomicAdd(state.hits, static_cast<unsigned long long>(delta)); }
    }

    // Phase 1: thread 0 selects the swaps — the hottest uncovered warm experts against the
    // weakest promoted ones, with the same hysteresis as the host pass this replaces.
    __shared__ int swaps;
    __shared__ std::uint32_t newcomer[2];
    __shared__ std::uint32_t incumbent[2];
    if (tid == 0) {
        swaps = 0;
        // Tiers the tail fully covers have no candidates; skip straight to the snapshot.
        if (lp.slots < lp.warm_count) {
            std::uint32_t candidate[2][2] = {{0, 0}, {0, 0}};  // (expert, delta) top-2 by delta
            std::uint32_t weakest[2][2]   = {{0xffffffffu, 0xffffffffu},
                                              {0xffffffffu, 0xffffffffu}};  // bottom-2 promoted
            for (std::uint32_t e = 0; e < state.experts; ++e) {
                if (state.warm_rank[base + e] == ExpertCacheAdaptState::kNoWarmRank) { continue; }
                const std::uint32_t expert_delta =
                    state.counters[base + e] - state.snapshot[base + e];
                if ((state.residency[base + e] & 0x80000000u) != 0u) {
                    // Uncovered: track the top-2 by delta.
                    if (expert_delta > candidate[0][1]) {
                        candidate[1][0] = candidate[0][0];
                        candidate[1][1] = candidate[0][1];
                        candidate[0][0] = e;
                        candidate[0][1] = expert_delta;
                    } else if (expert_delta > candidate[1][1]) {
                        candidate[1][0] = e;
                        candidate[1][1] = expert_delta;
                    }
                } else {
                    // Promoted: track the bottom-2 by delta.
                    if (weakest[0][0] == 0xffffffffu || expert_delta < weakest[0][1]) {
                        weakest[1][0] = weakest[0][0];
                        weakest[1][1] = weakest[0][1];
                        weakest[0][0] = e;
                        weakest[0][1] = expert_delta;
                    } else if (weakest[1][0] == 0xffffffffu || expert_delta < weakest[1][1]) {
                        weakest[1][0] = e;
                        weakest[1][1] = expert_delta;
                    }
                }
            }
            for (int swap = 0; swap < kMaximumSwaps; ++swap) {
                const std::uint32_t incoming = candidate[swap][0];
                const std::uint32_t incoming_count = candidate[swap][1];
                const std::uint32_t outgoing = weakest[swap][0];
                const std::uint32_t outgoing_count = weakest[swap][1];
                if (incoming == 0 || outgoing == 0xffffffffu) { break; }
                if (incoming_count < kMinimumRecentCount) { break; }
                if (incoming_count <= outgoing_count * kEvictionMargin) { break; }
                newcomer[swaps]  = incoming;
                incumbent[swaps] = outgoing;
                ++swaps;
            }
        }
    }
    __syncthreads();

    // Phase 2: the swaps — the block copies each newcomer's slice into the freed tail slot
    // and thread 0 flips the residency entries.
    for (int swap = 0; swap < swaps; ++swap) {
        const std::uint32_t outgoing = incumbent[swap];
        const std::uint32_t incoming = newcomer[swap];
        const std::uint32_t tail_slot =
            (state.residency[base + outgoing] & 0x7fffffffu) - lp.hot;
        __syncthreads();
        copy_expert_slice(lp, state.warm_rank[base + incoming], tail_slot, tid);
        __syncthreads();
        if (tid == 0) {
            state.residency[base + outgoing] =
                0x80000000u | state.warm_rank[base + outgoing];
            state.residency[base + incoming] = lp.hot + tail_slot;
            atomicAdd(state.promotions, 1ULL);
            atomicAdd(state.evictions, 1ULL);
        }
        __syncthreads();
    }

    // Phase 3: the snapshot update — this pass's counters become the next pass's delta base.
    state.snapshot[base + tid] = count;
}

} // namespace

void launch_expert_cache_adapt(const ExpertCacheAdaptState& state, cudaStream_t stream) {
    if (state.layer_count == 0) { return; }
    expert_cache_adapt_kernel<<<state.layer_count, state.experts, 0, stream>>>(state);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
