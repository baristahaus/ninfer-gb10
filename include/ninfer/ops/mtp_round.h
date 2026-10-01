#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: mtp_prepare_next_round
 *
 * Math / indexing:
 *   For T=K+1 and each row b with A=accepted[b], L=licensed_counts[b]:
 *     alignment_ids[j,b] = verify_ids[j+1,b]  for 0<=j<A
 *                          next_anchors[b]     otherwise;
 *     remaining_after = max(remaining_budgets[b]-L,0);
 *     context_after   = max(max_context-updated_frontiers[b]-1,0);
 *     next_extents[b] = min(K,max(remaining_after-1,0),context_after);
 *     For S=max(K-1,1) and 0<=s<S:
 *       ar_positions[b,s]      = updated_frontiers[b]+s;
 *       ar_rope_positions[b,s] = ar_positions[b,s]+rope_deltas[b];
 *       ar_valid_columns[b,s]  = (s+1 < next_extents[b]).
 *
 * Logical shapes / effects:
 *   verify_ids/alignment_ids are distinct contiguous I32 [K+1,B]. ar_positions,
 *   ar_rope_positions, and ar_valid_columns are I32 [B,max(K-1,1)] with contiguous rows and one
 *   shared step stride at least B; this permits an exact-B prefix of a fixed-capacity frame. All
 *   other tensors are contiguous I32 [B]. B>=1, 1<=K<=5, 0<=accepted[b]<=K,
 *   licensed_counts[b]=accepted[b]+1, updated_frontiers and remaining_budgets are non-negative,
 *   and max_context is positive. The Op writes every output slot, including safe invalid-tail
 *   values. Inputs remain unchanged. No workspace or other state is used.
 */
void mtp_prepare_next_round(const Tensor& verify_ids, const Tensor& next_anchors,
                            const Tensor& accepted, const Tensor& updated_frontiers,
                            const Tensor& remaining_budgets, const Tensor& licensed_counts,
                            const Tensor& rope_deltas, Tensor& alignment_ids, Tensor& next_extents,
                            Tensor& ar_positions, Tensor& ar_rope_positions,
                            Tensor& ar_valid_columns, std::int32_t max_context,
                            cudaStream_t stream);

/**
 * Op: mtp_advance_round
 *
 * Math / indexing:
 *   The device-resident MTP round frame for the round after this one, assuming every row
 *   continues and commits its whole licensed output. For each row b with L=licensed_counts[b],
 *   P=clamp(next_extents[b],0,K), anchor=anchors[b] (the round's correction/bonus token) and
 *   frontier=frontiers[b] (the frontier after the round):
 *     remaining_budgets[b]       = max(remaining_budgets[b]-L,0);
 *     current_extents[b]         = P;
 *     target_valid_columns[b]    = P+1;
 *     current_drafts[j,b]        = next_drafts[b,j] for 0<=j<P, anchor for P<=j<K;
 *     target_rope_positions[j,b] = frontier+min(j,P)+rope_deltas[b] for 0<=j<=K;
 *     pending_folds[:,b]         = {state_slots[b], state_slots[b], L, 0}.
 *
 * Logical shapes / effects:
 *   anchors, frontiers, licensed_counts, next_extents, rope_deltas, state_slots,
 *   remaining_budgets, current_extents and target_valid_columns are contiguous I32 [B].
 *   next_drafts is I32 [B,K] with contiguous rows and a step stride of at least B (an exact-B
 *   prefix of a fixed-capacity frame). current_drafts is contiguous I32 [K,B],
 *   target_rope_positions contiguous I32 [K+1,B], and pending_folds contiguous I32 [4,B].
 *   B>=1 and 1<=K<=5. remaining_budgets is updated in place; the other outputs are written for
 *   every slot and are distinct from the inputs. No workspace or other state is used.
 */
void mtp_advance_round(const Tensor& anchors, const Tensor& frontiers,
                       const Tensor& licensed_counts, const Tensor& next_extents,
                       const Tensor& next_drafts, const Tensor& rope_deltas,
                       const Tensor& state_slots, Tensor& remaining_budgets,
                       Tensor& current_extents, Tensor& target_valid_columns,
                       Tensor& current_drafts, Tensor& target_rope_positions, Tensor& pending_folds,
                       cudaStream_t stream);

} // namespace ninfer::ops
