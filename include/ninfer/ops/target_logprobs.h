#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Highest rank count one call reports. This is the sampler's own candidate ceiling, so a reported
// ranking is never wider than the set a stochastic draw could have selected from.
inline constexpr std::int32_t kMaxReportedLogprobRanks = 20;

// Written to top_ids for ranks past the reported count.
inline constexpr std::int32_t kNoReportedLogprobRank = -1;

/**
 * Op: target token log-probabilities
 *
 * Math / indexing:
 *   Let l[r,c] be the exact real value represented by logits[r,c]. With V = valid_rows:
 *
 *     p(r,c) = l[r,c] - log(sum_{u=0..V-1} exp(l[u,c])).
 *
 *   That is the model's own distribution over the token domain: no temperature scaling and no
 *   presence or frequency adjustment, which is what vLLM reports and what an external scorer,
 *   calibrator or verifier needs to agree with the checkpoint. The request's sampling policy is a
 *   separate question, so a token drawn under temperature or penalties need not be the reported
 *   top-1, and the Op cannot express the policy even by accident.
 *
 *   output[c] = p(target_ids[c], c). When top_ids and top_logprobs are both non-null they receive
 *   the leading K ranks of each column ordered by descending l - equivalently descending p, the
 *   normalizer being column-constant - with the lower token id breaking an exact tie, stored
 *   column-major so rank r of column c lives at r + c*K.
 *
 * Logical shapes:
 *   logits is [physical_rows,C], target_ids and output are [C], and top_ids/top_logprobs are [K,C]
 *   with K in [1,kMaxReportedLogprobRanks], with C>0 and 1<=valid_rows<=physical_rows. Values in
 *   target_ids are in [0,valid_rows). Physical rows [valid_rows,physical_rows) participate in
 *   neither the denominator, the target lookup, nor the reported ranking.
 *
 * Supported domain:
 *   logits is contiguous finite BF16, target_ids is contiguous I32, output is contiguous FP32, and a
 *   requested ranking is contiguous I32 / FP32. Storage has its dtype's natural alignment.
 *
 * Numeric:
 *   output and top_logprobs are the FP32 numerical approximation of ideal. Reduction association and
 *   private accumulator precision are implementation choices; the independent oracle evaluates the
 *   full formula in FP64 from the represented BF16 inputs.
 *
 * Effects:
 *   Writes every element of output, and of top_ids/top_logprobs when requested, giving ranks past
 *   min(K, valid_rows) the id kNoReportedLogprobRank and negative infinity. Preserves every input. No
 *   output may overlap another output or any input.
 *
 * Workspace:
 *   None. Reads only logits: no committed-token counts, no sampling config and no round-local token
 *   array, so the report of a column does not depend on when in the round it is taken.
 *
 * Execution:
 *   Enqueues one CTA per column on stream and owns no persistent state. The route is sized for the
 *   few columns one round publishes, not for a full prefill matrix. Ranking is a compile-time choice,
 *   so a call that requests none allocates no ranking storage.
 */
void target_logprobs(const Tensor& logits, const Tensor& target_ids, std::int32_t valid_rows,
                     Tensor& output, Tensor* top_ids, Tensor* top_logprobs, cudaStream_t stream);

} // namespace ninfer::ops
