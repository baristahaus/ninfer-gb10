#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

struct SpeculativeAcceptExecutionEnvelope {
    // Execution promise: every row has temperature<=0, both penalties disabled and no token mask.
    // When false, the general route remains valid for any supported mixture of greedy and
    // stochastic rows.
    bool all_rows_greedy_without_penalties = false;
};

// Caller-owned transient capacity for every draft-count and batch-size pair in the inclusive
// domains. token_domain is the fixed sampling profile; invalid domains throw.
[[nodiscard]] std::size_t speculative_accept_greedy_drafts_workspace_capacity_bytes(
    std::int32_t token_domain, std::int32_t min_drafts, std::int32_t max_drafts,
    std::int32_t min_batch, std::int32_t max_batch);

/**

 * Op: speculative_select_accepted_hidden
 *
 * Math / indexing:
 *   out[:,b] = hidden[:,selectors[b],b].
 *
 * Shape / numeric / effects:
 *   hidden is contiguous BF16 [D,T,B], selectors is contiguous I32 [B] with every value in [0,T),
 *   and out is distinct contiguous BF16 [D,B]. The Op exactly copies BF16 bits, writes all of out,
 *   and uses no workspace or other state.
 */
void speculative_select_accepted_hidden(const Tensor& hidden, const Tensor& selectors, Tensor& out,
                                        cudaStream_t stream);

/**
 * Op: proposal_remap_token_ids
 *
 * Math / indexing:
 *   proposal_tokens[i]' = id_map[proposal_tokens[i]] for every proposal token.
 *
 * Effects:
 *   Updates the contiguous non-empty I32 proposal_tokens vector in place; every input id is in
 *   [0,count), and id_map is a distinct device I32 array [count]. There is no workspace or other
 *   state side effect.
 */
void proposal_remap_token_ids(Tensor& proposal_tokens, const std::int32_t* id_map,
                              std::int32_t count, cudaStream_t stream);

/**
 * Op: speculative_prepare_verify_inputs
 *
 * Math / indexing:
 *   For row b and 0<=j<=K:
 *     verify_ids[j,b] = anchors[b]                         when j=0
 *                       drafts[j-1,b]                     when 0<j<=Pcur[b]
 *                       anchors[b]                        otherwise;
 *     positions[j,b]  = base_positions[b] + min(j,Pcur[b]).
 *
 * Logical shapes:
 *   All tensors are contiguous I32. anchors/base_positions/current_extents are [B], drafts is
 *   [K,B] with K>=1 and B>=1, and verify_ids/positions are [K+1,B]. Each current extent is in
 *   [0,K]. Inputs and outputs do not overlap.
 *
 * Effects:
 *   Writes every physical output element, including safe invalid-tail values. Inputs remain
 *   unchanged.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues work on execution.stream() and owns no persistent state.
 */
void speculative_prepare_verify_inputs(const Tensor& anchors, const Tensor& drafts,
                                       const Tensor& base_positions, const Tensor& current_extents,
                                       Tensor& verify_ids, Tensor& positions, cudaStream_t stream);

/**
 * Op: speculative_prepare_verify_ids
 *
 * Math / indexing:
 *   For row b and 0<=j<=K:
 *     verify_ids[j,b] = anchors[b]             when j=0
 *                       drafts[j-1,b]          when 0<j<=Pcur[b]
 *                       anchors[b]             otherwise.
 *
 * Logical shapes:
 *   All tensors are contiguous I32. anchors/current_extents are [B], drafts is [K,B] with K>=1
 *   and B>=1, and verify_ids is [K+1,B]. Each current extent is in [0,K]. Inputs and outputs do
 *   not overlap.
 *
 * Effects:
 *   Writes every physical output element, including safe invalid-tail values. Inputs remain
 *   unchanged.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues work on execution.stream() and owns no persistent state.
 */
void speculative_prepare_verify_ids(const Tensor& anchors, const Tensor& drafts,
                                    const Tensor& current_extents, Tensor& verify_ids,
                                    cudaStream_t stream);

/**
 * Op: speculative_accept_greedy_drafts
 *
 * Math / indexing:
 *   For row b: extend accepted[b] greedily while draft j-1 equals argmax(logits[:,j-1,b]) and
 *   the running extent is below max_extents[b]; the last accepted id is committed and the
 *   remaining drafts are rejected.
 *
 * Logical shapes:
 *   target_tokens/logits/current_extents are contiguous; logits is BF16 [V,C,T] with
 *   T=K>=1; target_tokens is I32 [T,B]; drafts is I32 [K,B]; current_extents is I32 [B] with
 *   values in [0,K]; accepted is I32 [K+1,B]; report is I32 [3,B] (row 0 committed count, row 1
 *   accepted count, row 2 last accepted id or -1).
 *
 * Effects:
 *   Writes every element of accepted and report. Inputs remain unchanged.
 *
 * Workspace:
 *   speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, min_drafts,
 *   max_drafts, min_batch, max_batch).
 *
 * Execution:
 *   Enqueues work on execution.stream() and owns no persistent state.
 */
void speculative_accept_greedy_drafts(const Tensor& target_tokens, const Tensor& logits,
                                      const Tensor& drafts, const Tensor& current_extents,
                                      std::int32_t token_domain, Tensor& accepted, Tensor& report,
                                      cudaStream_t stream);

/**
 * Workspace capacity for speculative_accept_greedy_drafts over the given draft and batch range.
 */
[[nodiscard]] std::size_t
speculative_accept_greedy_drafts_workspace_capacity_bytes(std::int32_t token_domain,
                                                          std::int32_t min_drafts,
                                                          std::int32_t max_drafts,
                                                          std::int32_t min_batch,
                                                          std::int32_t max_batch);

} // namespace ninfer::ops
