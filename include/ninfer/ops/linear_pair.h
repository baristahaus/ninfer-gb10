#pragma once

#include "core/device.h"
#include "ninfer/ops/projection_weights.h"
#include "core/tensor.h"

namespace ninfer::ops {

// Capacity for every extent in the inclusive interval; invalid profiles or intervals throw.
[[nodiscard]] std::size_t linear_pair_workspace_capacity_bytes(const Weight& first_weight,
                                                               const Weight& second_weight,
                                                               std::int32_t min_tokens,
                                                               std::int32_t max_tokens);

[[nodiscard]] std::size_t linear_pair_workspace_capacity_bytes(LinearProfile first,
                                                               LinearProfile second,
                                                               std::int32_t min_tokens,
                                                               std::int32_t max_tokens);

/**
 * first_out[:,t] = W_first * x[:,t]; second_out[:,t] = W_second * x[:,t].
 *
 * x [K,T] and both outputs [N,T] are contiguous, 16-byte-aligned BF16 matrices, T > 0.
 * N is the weights' own row count, not a fixed 1024: the two weights must agree on it, and both
 * the MMA and the SIMT arm derive their row extent from it (the MMA kernel masks `grow < m`). The
 * 27B's MTP K/V projection is [1024,5120] at width 1 and [512,5120] at width 2, because its four
 * KV heads shard by rows; both are this one Op.
 * Both weights use Q8_G32_FP16 RowSplit encoding on the same [N,K] geometry;
 * code and scale planes require 16-byte alignment.
 * The K=2048 geometry additionally requires N == 1024 and adjacent row views [4096,5120) and
 * [5120,6144) of the same [6144,2048] parent, including its code and scale planes.
 * Inputs, outputs and weight planes must be mutually non-overlapping.
 * Overwrites both outputs; requires no scratch and has no persistent state.
 */
void linear_pair(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                 Tensor& first_out, Tensor& second_out, CudaExecutionView execution);

} // namespace ninfer::ops
