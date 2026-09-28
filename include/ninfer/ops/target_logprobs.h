#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Highest rank count one call reports. This is the sampler's own candidate ceiling, so a reported
// ranking is never wider than the set a stochastic draw could have selected from.
inline constexpr std::int32_t kMaxReportedLogprobRanks = 20;

// Written to top_ids for ranks past the reported count.
inline constexpr std::int32_t kNoReportedLogprobRank = -1;

// Bound on the round-local draft prefix one column can reconcile against, matching the widest
// speculative window in the Flash-Next family: a 16-column DFlash verify carries 15 drafts. MTP
// reaches 5.
inline constexpr std::int32_t kMaxRoundDrafts = 16;

struct TargetLogprobOptions {
    // Device [columns] sampling configs. Null reports the plain full-domain log-softmax of the
    // represented logits, which is the causal-scoring route. Non-null applies exactly the sampler's
    // adjustment for that column: presence/frequency penalties and temperature scaling, read from
    // the same struct the sampler used, so a reported probability cannot drift from the draw. Only
    // temperature, presence_penalty, frequency_penalty and token_counts are read; the selection
    // fields belong to the sampler.
    const SamplingConfig* configs = nullptr;
    // Speculative verify: device [draft_rows, batch] I32 drafts, laid out like the verify route's
    // own draft tensor, or null for a non-speculative round. Column g of such a round is verify
    // column w of lane b under g = w + b*verify_width, and the prefix that column was drawn against
    // is drafts[0..w-1] of lane b. This is the same round-local overlay ops::speculative_accept_*
    // applies inline, so a verified token's reported probability accounts for the drafts that
    // preceded it just as the draw did.
    const std::int32_t* round_drafts = nullptr;
    std::int32_t draft_rows          = 0;
    std::int32_t verify_width        = 0;
};

/**
 * Op: target token log-probabilities
 *
 * Math / indexing:
 *   Let l[r,c] be the exact real value represented by logits[r,c]. Penalty adjustment matches
 *   ops::sample: with config the column's configs[c], cnt(r) = config.token_counts[r] (zero when
 *   that pointer is null) plus, for a speculative round, the number of that column's draft prefix
 *   entries equal to r,
 *
 *     a[r,c] = l[r,c] - presence_penalty * (cnt(r) > 0) - frequency_penalty * cnt(r),
 *     tau    = config.temperature > 0 ? config.temperature : 1  (tau = 1 when configs is null, so
 *              a greedy column still reports a distribution),
 *     p(r,c) = a[r,c]/tau - log(sum_{u=0..valid_rows-1} exp(a[u,c]/tau)).
 *
 *   output[c] = p(target_ids[c], c). Because tau is positive, ranking by a[r,c] and by p(r,c)
 *   coincide: when top_ids and top_logprobs are both non-null they receive the leading K ranks of
 *   each column ordered by descending p, lower token id breaking exact ties, stored column-major so
 *   rank r of column c lives at r + c*K.
 *
 * Logical shapes:
 *   logits is [physical_rows,C], target_ids and output are [C], round_drafts is
 *   [draft_rows,C/verify_width], and top_ids/top_logprobs are [K,C] with K in
 *   [1,kMaxReportedLogprobRanks], with C>0 and 1<=valid_rows<=physical_rows. Values in target_ids
 *   are in [0,valid_rows). Physical rows [valid_rows,physical_rows) participate in neither the
 *   denominator, the target lookup, nor the reported ranking.
 *
 * Supported domain:
 *   logits is contiguous finite BF16, target_ids is contiguous I32, output is contiguous FP32, and
 *   a requested ranking is contiguous I32 / FP32. Storage has its dtype's natural alignment.
 *
 * Numeric:
 *   output and top_logprobs are the FP32 numerical approximation of ideal. Reduction association
 *   and private accumulator precision are implementation choices; the independent oracle evaluates
 *   the full formula in FP64 from the represented BF16 inputs.
 *
 * Effects:
 *   Writes every element of output, and of top_ids/top_logprobs when requested, giving ranks past
 *   min(K, valid_rows) the id kNoReportedLogprobRank and negative infinity. Preserves every input.
 *   No output may overlap another output, any input, the configs array, or the drafts array.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues one CTA per column on stream and owns no persistent state. The route is sized for the
 *   few columns one round publishes, not for a full prefill matrix.
 */
void target_logprobs(const Tensor& logits, const Tensor& target_ids, std::int32_t valid_rows,
                     const TargetLogprobOptions& options, Tensor& output, Tensor* top_ids,
                     Tensor* top_logprobs, cudaStream_t stream);

} // namespace ninfer::ops
