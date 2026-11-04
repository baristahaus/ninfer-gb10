#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// No workspace is required.
[[nodiscard]] constexpr std::size_t column_topk_workspace_capacity_bytes() noexcept {
    return 0;
}

/**
 * Op: column_topk
 *
 * scores[:,:,b][:,t] , ids[:,:,b][:,t] = the K highest (logits[:,t,b][v], v) pairs over v.
 *
 * logits is contiguous BF16 [V,T,B] with the vocabulary axis stored fastest per column, the
 * layout the vocabulary producers emit (column c occupies logits[c*V, (c+1)*V) where
 * c = t*B + b); V >= 1, T >= 1, B >= 1, and K is 1..16.
 * ids is contiguous I32 [K,T,B] and scores is contiguous FP32 [K,T,B]; rank is the
 * fastest-moving axis. Candidates are ordered by descending logit value, with ties
 * resolved by the lower row index. Consumed logits must be finite. Inputs and outputs
 * require 16-byte alignment on execution.device().
 *
 * This is the DFlash2 candidate extractor for a BF16 head whose geometry has no
 * registered fused linear_topk profile; the head projection is a separate linear.
 */
void column_topk(const Tensor& logits, Tensor& ids, Tensor& scores, CudaExecutionView execution);

} // namespace ninfer::ops
