#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

struct SpeculativeAcceptExecutionEnvelope {
    // Execution promise: every row has temperature<=0 and both penalties disabled. When false,
    // the general route remains valid for any supported mixture of greedy and stochastic rows.
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

} // namespace ninfer::ops
