#pragma once

// Shared adjustment and ordering primitives for the sampler, the speculative accept route and
// target log probabilities. Every consumer of a token's reported probability must apply the same
// penalty adjustment and the same tie-break, so the three live here rather than in one caller's
// header. Pure device helpers with no storage of their own.

#include "ninfer/ops/sampling.h"

#include <cstdint>

namespace ninfer::ops {

// Candidate ordering: higher value wins, ties broken by lower vocab index.
__device__ __forceinline__ bool sampling_better(float v, int i, float bv, int bi) {
    return v > bv || (v == bv && i < bi);
}

__device__ __forceinline__ void sampling_insert_candidate(float* vals, int* idxs, int cap, float v,
                                                          int idx) {
    if (cap <= 0 || !sampling_better(v, idx, vals[cap - 1], idxs[cap - 1])) { return; }
    int pos = cap - 1;
    while (pos > 0 && sampling_better(v, idx, vals[pos - 1], idxs[pos - 1])) {
        vals[pos] = vals[pos - 1];
        idxs[pos] = idxs[pos - 1];
        --pos;
    }
    vals[pos] = v;
    idxs[pos] = idx;
}

// The penalty arithmetic itself: presence applies once to a token that has been seen at all,
// frequency scales with how often. Every route that adjusts a logit for committed history ends here,
// so a reported probability and the draw it describes cannot drift apart by arithmetic. What the
// caller assembles into `count` is its own business, because the sampler's live counts and a
// report's draw-time view are different questions about the same round.
__device__ __forceinline__ float sampling_penalized(float raw, int count,
                                                    const SamplingConfig& config) {
    float value = raw;
    if (count > 0) { value -= config.presence_penalty; }
    if (config.frequency_penalty != 0.0f) {
        value -= config.frequency_penalty * static_cast<float>(count);
    }
    return value;
}

// Applies the sampler's penalties to a raw logit. `overlay`/`overlay_len` carry a round-local count
// overlay: tokens committed earlier in the current speculative round but not yet flushed to the
// global `token_counts`. For speculative verify column `col` the overlay is exactly drafts[0..col-1]
// (statically known, since column `col` is only consumed when every earlier draft was accepted), so
// the penalty at each column sees the same prefix a per-token sampler would. Non-speculative callers
// pass no overlay. The scan is bounded by k and only runs when penalties are active.
__device__ __forceinline__ float sampling_adjusted_logit(float raw, int v, const SamplingConfig& c,
                                                         const std::int32_t* overlay = nullptr,
                                                         int overlay_len             = 0) {
    if (c.presence_penalty == 0.0f && c.frequency_penalty == 0.0f) { return raw; }
    int cnt = c.token_counts != nullptr ? c.token_counts[v] : 0;
    for (int j = 0; j < overlay_len; ++j) {
        if (overlay[j] == v) { ++cnt; }
    }
    return sampling_penalized(raw, cnt, c);
}

} // namespace ninfer::ops
