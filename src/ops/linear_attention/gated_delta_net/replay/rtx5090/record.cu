#include "ops/linear_attention/gated_delta_net/replay/launch.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/linear_attention/gated_delta_net/replay/record_io.cuh"
#include "core/device.h"
#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {
namespace {
template <bool Masked>
void launch_recurrent_record_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& ssm_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, Tensor& key_record,
                                   Tensor& value_record, Tensor& gate_record, Tensor& out,
                                   cudaStream_t stream) {
    const auto heads = HeadMap::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), static_cast<unsigned>(q.ne[3]),
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(common::kWarpSize, kNumWarps, 1);
    const std::int64_t state_slot_stride =
        ssm_states.nb[3] / static_cast<std::int64_t>(sizeof(float));
    const RecordAccess<Masked> access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<const float*>(ssm_states.data),
        Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        static_cast<const std::int32_t*>(initial_state_slots.data),
        static_cast<__nv_bfloat16*>(key_record.data),
        static_cast<__nv_bfloat16*>(value_record.data),
        reinterpret_cast<uint2*>(gate_record.data),
        static_cast<__nv_bfloat16*>(out.data),
        heads,
        q.ne[2],
        state_slot_stride,
        scale,
    };
    recurrent_record_kernel<Masked><<<grid, block, 0, stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
}

// The fused fold-then-record pass in this target's plain sequence formulation: the pending fold
// runs the same arithmetic as the fold op's recurrent_fold_kernel (load_state_tile +
// run_recurrent_sequence<FoldEffects> + state store), then the record body matches
// recurrent_record_kernel exactly, so the fused result equals fold-then-record bit for bit.
template <bool Masked>
__global__ void __launch_bounds__(common::kWarpSize * kNumWarps, 2)
    recurrent_fold_record_plain_kernel(RecordAccess<Masked> access, PendingFoldTable pending) {
    const RecurrentCoordinates coord = access.coordinates();
    const std::int32_t valid        = access.active_columns(coord);
    const std::int32_t slot         = access.initial_slots[coord.batch];
    for (std::int32_t entry = 0; entry < pending.count; ++entry) {
        const GdnReplayFoldKernelRow row = pending.rows[entry];
        if (row.commit_columns <= 0 || row.source_state_slot != slot ||
            row.destination_state_slot != slot) {
            continue;
        }
        const PendingFoldAccess fold{pending.key_record,
                                     pending.value_record,
                                     pending.gate_record,
                                     const_cast<float*>(access.states),
                                     slot,
                                     coord.batch,
                                     entry,
                                     access.heads,
                                     pending.width,
                                     access.state_slot_stride};
        __align__(16) float fold_state[kDvPerWarp][kQkPerLane];
        load_state_tile(fold_state, fold.state_read_base(coord), coord);
        run_recurrent_sequence<true, FoldEffects>(fold_state, fold, coord, row.commit_columns);
        store_state_tile(fold_state, fold.state_write_base(coord), coord);
        break;
    }
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);
    run_recurrent_sequence<true, RecordEffects>(state, access, coord, valid);
    zero_output_suffix(access, coord, valid, access.width);
}

template <bool Masked>
void launch_recurrent_fold_record_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                        const Tensor& g, const Tensor& beta, float scale,
                                        Tensor& ssm_states, const Tensor& valid_columns,
                                        const Tensor& initial_state_slots, Tensor& key_record,
                                        Tensor& value_record, Tensor& gate_record, Tensor& out,
                                        const PendingFoldTable& pending, cudaStream_t stream) {
    const auto heads = HeadMap::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), static_cast<unsigned>(q.ne[3]),
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(common::kWarpSize, kNumWarps, 1);
    const std::int64_t state_slot_stride =
        ssm_states.nb[3] / static_cast<std::int64_t>(sizeof(float));
    const RecordAccess<Masked> access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<const float*>(ssm_states.data),
        Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        static_cast<const std::int32_t*>(initial_state_slots.data),
        static_cast<__nv_bfloat16*>(key_record.data),
        static_cast<__nv_bfloat16*>(value_record.data),
        reinterpret_cast<uint2*>(gate_record.data),
        static_cast<__nv_bfloat16*>(out.data),
        heads,
        q.ne[2],
        state_slot_stride,
        scale,
    };
    recurrent_fold_record_plain_kernel<Masked><<<grid, block, 0, stream>>>(access, pending);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void launch_recurrent_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                             const Tensor& beta, float scale, const Tensor& ssm_states,
                             const Tensor& valid_columns, const Tensor& initial_state_slots,
                             Tensor& key_record, Tensor& value_record, Tensor& gate_record,
                             Tensor& out, cudaStream_t stream) {
    if (valid_columns.data == nullptr) {
        launch_recurrent_record_fixed<false>(q, k, v, g, beta, scale, ssm_states, valid_columns,
                                             initial_state_slots, key_record, value_record,
                                             gate_record, out, stream);
    } else {
        launch_recurrent_record_fixed<true>(q, k, v, g, beta, scale, ssm_states, valid_columns,
                                            initial_state_slots, key_record, value_record,
                                            gate_record, out, stream);
    }
}

void launch_recurrent_fold_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& g, const Tensor& beta, float scale,
                                  Tensor& ssm_states, const Tensor& valid_columns,
                                  const Tensor& initial_state_slots, Tensor& key_record,
                                  Tensor& value_record, Tensor& gate_record, Tensor& out,
                                  const GdnReplayFoldKernelRow* pending_rows,
                                  std::int32_t pending_count, const Tensor& pending_key_record,
                                  const Tensor& pending_value_record,
                                  const Tensor& pending_gate_record, cudaStream_t stream) {
    const PendingFoldTable pending{
        pending_rows,
        pending_count,
        static_cast<const __nv_bfloat16*>(pending_key_record.data),
        static_cast<const __nv_bfloat16*>(pending_value_record.data),
        reinterpret_cast<const uint2*>(pending_gate_record.data),
        pending_key_record.ne[2],
    };
    if (valid_columns.data == nullptr) {
        launch_recurrent_fold_record_fixed<false>(q, k, v, g, beta, scale, ssm_states,
                                                  valid_columns, initial_state_slots, key_record,
                                                  value_record, gate_record, out, pending, stream);
    } else {
        launch_recurrent_fold_record_fixed<true>(q, k, v, g, beta, scale, ssm_states, valid_columns,
                                                 initial_state_slots, key_record, value_record,
                                                 gate_record, out, pending, stream);
    }
}

} // namespace ninfer::ops::detail::gated_delta_net
