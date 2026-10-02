#pragma once

#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {

struct alignas(16) GdnReplayFoldKernelRow {
    std::int32_t source_state_slot;
    std::int32_t destination_state_slot;
    std::int32_t commit_columns;
    std::int32_t reserved = 0;
};

struct alignas(16) GdnReplayFoldKernelRows {
    GdnReplayFoldKernelRow row[8];
};

void launch_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, bool normalize_qk, Tensor& ssm_state,
                      Tensor& out, cudaStream_t stream);

void launch_recurrent_inout(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk,
                            const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                            cudaStream_t stream);

void launch_recurrent_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   bool normalize_qk, Tensor& ssm_states,
                                   const Tensor& source_state_slots,
                                   const Tensor& destination_state_slots, Tensor& out,
                                   cudaStream_t stream);

void launch_recurrent_batch_update_packed_qkv(
    const Tensor& packed_qkv, std::int32_t qk_heads, std::int32_t value_heads,
    const Tensor& g, const Tensor& beta, float scale, Tensor& ssm_states,
    const Tensor& source_state_slots, const Tensor& destination_state_slots,
    Tensor& out, cudaStream_t stream);

void launch_recurrent_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                             const Tensor& beta, float scale, const Tensor& ssm_states,
                             const Tensor& valid_columns, const Tensor& initial_state_slots,
                             Tensor& key_record, Tensor& value_record, Tensor& gate_record,
                             Tensor& out, cudaStream_t stream);

// launch_recurrent_record preceded by each row's pending fold (recurrent_fold_record_kernel).
// pending_rows is device memory with pending_count descriptors; the pending records are BF16
// key [128,Hq,T,R], BF16 value [128,Hv,T,R] and FP32 gate [2,Hv,T,R] of the previous round.
void launch_recurrent_fold_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& g, const Tensor& beta, float scale,
                                  Tensor& ssm_states, const Tensor& valid_columns,
                                  const Tensor& initial_state_slots, Tensor& key_record,
                                  Tensor& value_record, Tensor& gate_record, Tensor& out,
                                  const GdnReplayFoldKernelRow* pending_rows,
                                  std::int32_t pending_count, const Tensor& pending_key_record,
                                  const Tensor& pending_value_record,
                                  const Tensor& pending_gate_record, cudaStream_t stream);

// device_rows, when non-null, replaces rows: the kernel reads each row's descriptor from device
// memory at execution time.
void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const GdnReplayFoldKernelRows& rows,
                        const GdnReplayFoldKernelRow* device_rows, std::int32_t active_rows,
                        cudaStream_t stream);

} // namespace ninfer::ops::detail::gated_delta_net
