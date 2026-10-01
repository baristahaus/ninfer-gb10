#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void mtp_prepare_next_round_launch(const Tensor& verify_ids, const Tensor& next_anchors,
                                   const Tensor& accepted, const Tensor& updated_frontiers,
                                   const Tensor& remaining_budgets, const Tensor& licensed_counts,
                                   const Tensor& rope_deltas, Tensor& alignment_ids,
                                   Tensor& next_extents, Tensor& ar_positions,
                                   Tensor& ar_rope_positions, Tensor& ar_valid_columns,
                                   std::int32_t max_context, cudaStream_t stream);

void mtp_advance_round_launch(const Tensor& anchors, const Tensor& frontiers,
                              const Tensor& licensed_counts, const Tensor& next_extents,
                              const Tensor& next_drafts, const Tensor& rope_deltas,
                              const Tensor& state_slots, Tensor& remaining_budgets,
                              Tensor& current_extents, Tensor& target_valid_columns,
                              Tensor& current_drafts, Tensor& target_rope_positions,
                              Tensor& pending_folds, const Tensor& verify_ids,
                              const Tensor& licensed_tokens, Tensor& ple_history,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
