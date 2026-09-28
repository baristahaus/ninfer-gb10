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

struct TargetLogprobOptions {
    // Device [columns / columns_per_lane] sampling configs, or null to report the plain full-domain
    // log-softmax of the represented logits, which is the causal-scoring route. Non-null applies the
    // sampler's own adjustment - temperature scaling and presence/frequency penalties - read from
    // the same struct the draw used. Only temperature, presence_penalty, frequency_penalty and
    // token_counts are read; the selection fields belong to the sampler.
    const SamplingConfig* configs = nullptr;
    // How many reported columns share one config entry. A batched decode round has one column per
    // lane; an MTP verify round has one config per lane shared by that lane's verify columns.
    std::int32_t columns_per_lane = 1;
    // Device [columns_per_lane, lanes] token ids this round published, or null. `ops::sample` and
    // `ops::speculative_accept_*` add the tokens they select to `token_counts` before this Op runs,
    // so the counts already carry this round's own choices. Column w of a lane was drawn before the
    // tokens at and after it, so those are removed here to recover the counts the draw actually saw.
    // Reporting after the counts are updated is deliberate: it keeps this Op off the critical path
    // between the draw and the next accepted-token commit.
    const std::int32_t* round_tokens   = nullptr;
    const std::int32_t* round_produced = nullptr; // device [lanes], or null for columns_per_lane
};

/**
 * Op: target token log-probabilities
 *
 * Math / indexing:
 *   Let l[r,c] be the exact real value represented by logits[r,c]. Column c belongs to lane
 *   b = c / columns_per_lane at verify column w = c - b*columns_per_lane, and q(i) is
 *   round_tokens[b*columns_per_lane + i] with i < p, where p is round_produced[b] or
 *   columns_per_lane. With config = configs[b], cnt(r) = config.token_counts[r] (zero when that
 *   pointer is null) minus the number of q(i) with w <= i < p equal to r:
 *
 *     a[r,c] = l[r,c] - presence_penalty * (cnt(r) > 0) - frequency_penalty * cnt(r),
 *     tau    = config.temperature > 0 ? config.temperature : 1  (tau = 1 when configs is null, so
 *              a greedy column still reports a distribution),
 *     p(r,c) = a[r,c]/tau - log(sum_{u=0..valid_rows-1} exp(a[u,c]/tau)).
 *
 *   The subtraction recovers the committed-token counts as they were when column w was drawn: the
 *   draw's own token and every later token of this round are already in the counts, and neither
 *   belonged to that column's prefix. A column at or beyond p is not a published token and no
 *   consumer reads its report.
 *
 *   output[c] = p(target_ids[c], c). Because tau is positive, ranking by a[r,c] and by p(r,c)
 *   coincide: when top_ids and top_logprobs are both non-null they receive the leading K ranks of
 *   each column ordered by descending p, lower token id breaking exact ties, stored column-major so
 *   rank r of column c lives at r + c*K.
 *
 * Logical shapes:
 *   logits is [physical_rows,C], target_ids and output are [C], round_tokens is [C], round_produced
 *   is [C/columns_per_lane], and top_ids/top_logprobs are [K,C] with K in
 *   [1,kMaxReportedLogprobRanks], with C>0, columns_per_lane >= 1, columns_per_lane dividing C, and
 *   1<=valid_rows<=physical_rows. Values in target_ids are in [0,valid_rows). Physical rows
 *   [valid_rows,physical_rows) participate in neither the denominator, the target lookup, nor the
 *   reported ranking.
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
 *   No output may overlap another output, any input, the configs array, or either round array.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues one CTA per column on stream and owns no persistent state. The route is sized for the
 *   few columns one round publishes, not for a full prefill matrix. Ranking is a compile-time
 *   choice, so a call that requests none allocates no ranking storage.
 */
void target_logprobs(const Tensor& logits, const Tensor& target_ids, std::int32_t valid_rows,
                     const TargetLogprobOptions& options, Tensor& output, Tensor* top_ids,
                     Tensor* top_logprobs, cudaStream_t stream);

} // namespace ninfer::ops
