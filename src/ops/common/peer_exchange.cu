// Explicit instantiations of the pinned-host mailbox exchange kernels.
//
// The launchers land with the PeerMailbox owner in src/ops/common/peer_mailbox.cu; these keep the
// templates instantiated now, so the ported header cannot rot between this change and that one and
// the device code is compiled and linked on both lanes from the start.
//
// The production geometry is kPeerPipelinedVecsPerLane vectors per lane, acquire polling, untimed.
// The FP32 variant is the same geometry and moves the row-parallel partials the MoE path exchanges.

#include "ops/kernel/peer_exchange.cuh"

namespace ninfer::ops::detail {

template __global__ void peer_exchange_pipelined_kernel<kPeerPipelinedVecsPerLane, kPeerPollAcquire,
                                                        false>(
    PeerVecBf16* partial, PeerVecBf16* mine_payload, std::uint32_t* mine_flags,
    const PeerVecBf16* peer_payload, std::uint32_t* peer_flags, std::uint32_t* epochs,
    std::uint32_t* hang, int vecs, PeerPipelinedTuning tuning);

template __global__ void peer_exchange_pipelined_f32_kernel<kPeerPipelinedVecsPerLane>(
    PeerVec* partial, PeerVec* mine_payload, std::uint32_t* mine_flags, const PeerVec* peer_payload,
    std::uint32_t* peer_flags, std::uint32_t* epochs, std::uint32_t* hang, int vecs,
    __nv_bfloat16* bf16_out, PeerPipelinedTuning tuning);

} // namespace ninfer::ops::detail
