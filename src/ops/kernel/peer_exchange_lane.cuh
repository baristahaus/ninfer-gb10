// SPDX-FileCopyrightText: ValerioDolci/capy-flash
// SPDX-License-Identifier: Apache-2.0
#pragma once

// ninfer::ops::detail - the pipelined mailbox protocol's constants and its per-warp lane exchange,
// shared by the exchange kernels (peer_exchange.cuh). Device functions and constants only: the
// kernels stay in peer_exchange.cuh, compiled once (peer_mailbox.cu), so relocatable device code
// links. The protocol itself is documented in peer_exchange.cuh.
//
// Ported from capy-flash (formerly ninfer-tp2), src/ops/kernel/peer_exchange_lane.cuh, which in
// turn descends from the two-GPU forks credited in PROVENANCE.md. The protocol and its constants
// are unchanged; only the file reference in this comment differs.

#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Probes of __nanosleep(100) polling before the poller gives up and reports. One probe costs about
// 0.8 us (4M probes measured 3.25 s on two RTX 5070 Ti), so the limit is ~0.8 s: long enough for
// any legitimate skew between the two ranks' schedules, short enough to surface a missing peer
// before a 2 s display watchdog would reset the device (test_allreduce checks the bound).
inline constexpr std::uint32_t kPeerSpinLimit = 1000000u;

// A poller re-reads the pinned hang word once every this many probes, so a round that already
// reported a hang does not spin the full limit again at every later exchange.
inline constexpr std::uint32_t kPeerHangProbe = 1024u;

using PeerVec = uint4; // 16 bytes = 8 BF16 elements, one PCIe transaction per access

// Words between two warps' release flags: a 64-byte host line each.
inline constexpr int kPeerFlagStride = 16;

// Sleep between two polls of the peer's flag; kPeerSpinLimit is calibrated with it.
inline constexpr std::uint32_t kPeerPollSleepNs = 100u;

// 16-byte vectors per lane of the pipelined geometry (one warp lane = 1 KiB of payload).
inline constexpr int kPeerPipelinedVecsPerLane = 2;

union PeerVecF32 {
    PeerVec raw;
    float4 value;
};

// One warp lane of the pipelined protocol, for any kernel that exchanges FP32 data with the peer
// at the pipelined geometry (warp `warp` owns vectors warp*32*G + j*32 + lane, j < G, and one
// release line / epoch of the slot): publishes `mine` (the warp's vectors, already in registers),
// releases, polls the peer's line with acquire loads (HANG GUARD as above), records the epoch and
// loads the peer's vectors into `peer`. Returns false, warp-uniformly, when the poller gave up;
// `peer` is then unspecified. The caller must call it with the whole warp converged and only for
// warps with warp*32*G < vecs.
template <int G>
__device__ __forceinline__ bool
peer_exchange_lane(int warp, int lane, const PeerVec (&mine)[G], PeerVec (&peer)[G],
                   PeerVec* mine_payload, std::uint32_t* mine_flags, const PeerVec* peer_payload,
                   std::uint32_t* peer_flags, std::uint32_t* epochs, std::uint32_t* hang, int vecs,
                   std::uint32_t spin_limit, std::uint32_t sleep_ns) {
    using SystemWord = cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system>;
    const int base   = warp * (32 * G) + lane;
#pragma unroll
    for (int j = 0; j < G; ++j) {
        const int v = base + j * 32;
        if (v < vecs) { mine_payload[v] = mine[j]; }
    }
    __syncwarp();
    int published = 1;
    if (lane == 0) {
        const std::uint32_t target = epochs[warp] + 1u;
        SystemWord(mine_flags[warp * kPeerFlagStride]).store(target, cuda::memory_order_release);
        const SystemWord flag(peer_flags[warp * kPeerFlagStride]);
        const SystemWord fault(*hang);
        std::uint32_t seen  = flag.load(cuda::memory_order_acquire);
        std::uint32_t spins = 0;
        while (static_cast<std::int32_t>(seen - target) < 0) {
            __nanosleep(sleep_ns);
            ++spins;
            if (spins % kPeerHangProbe == 0u &&
                (spins > spin_limit || fault.load(cuda::memory_order_relaxed) != 0u)) {
                fault.store(1u, cuda::memory_order_relaxed);
                published = 0;
                break;
            }
            seen = flag.load(cuda::memory_order_acquire);
        }
        epochs[warp] = target;
    }
    __syncwarp();
    if (__shfl_sync(0xffffffffu, published, 0) == 0) { return false; }
#pragma unroll
    for (int j = 0; j < G; ++j) {
        const int v = base + j * 32;
        if (v < vecs) { peer[j] = __ldcv(&peer_payload[v]); }
    }
    return true;
}

__device__ __forceinline__ PeerVecF32 peer_sum_f32(const PeerVec& a, const PeerVec& b) {
    PeerVecF32 x;
    PeerVecF32 y;
    x.raw = a;
    y.raw = b;
    PeerVecF32 sum;
    sum.value.x = __fadd_rn(x.value.x, y.value.x);
    sum.value.y = __fadd_rn(x.value.y, y.value.y);
    sum.value.z = __fadd_rn(x.value.z, y.value.z);
    sum.value.w = __fadd_rn(x.value.w, y.value.w);
    return sum;
}

// Four floats rounded once to BF16 (RNE), as 8 bytes.
__device__ __forceinline__ uint2 peer_pack_bf16x4(const float4& v) {
    const __nv_bfloat162 low  = __floats2bfloat162_rn(v.x, v.y);
    const __nv_bfloat162 high = __floats2bfloat162_rn(v.z, v.w);
    uint2 packed;
    packed.x = *reinterpret_cast<const std::uint32_t*>(&low);
    packed.y = *reinterpret_cast<const std::uint32_t*>(&high);
    return packed;
}

} // namespace ninfer::ops::detail
