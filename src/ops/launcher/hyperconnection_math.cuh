#pragma once

// HyperConnection elementwise mathematics shared by the mix kernels and the fused FP8 up-mix
// epilogue, so every route produces the same per-element values.

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail::hyperconnection {

inline constexpr int kStreams = 4;
inline constexpr int kHidden  = 2560;
inline constexpr int kHyper   = kStreams * kHidden;
inline constexpr int kRank    = 320;

// Injection partials: for token t, source stream s and destination stream e, the dot of
// destination e's injection row with source stream s's normalized slice, at
// [t * 16 + s * 4 + e].
inline constexpr int kInjectionPartialsPerToken = kStreams * kStreams;

// block_input[d, t] = bf16(0.25 * sum_s sigmoid(bf16(logit_s)) * normalized[s, d, t]), with the
// gate logits the up projection's outputs for rows s * 2560 + d, summed over s in order.
__device__ __forceinline__ void gate_mix_position(const float* logits,
                                                  const __nv_bfloat16* normalized,
                                                  __nv_bfloat16* block_input, int position,
                                                  int token) {
    float sum = 0.0F;
#pragma unroll
    for (int stream = 0; stream < kStreams; ++stream) {
        const float logit = __bfloat162float(__float2bfloat16_rn(logits[stream]));
        const float gate  = 1.0F / (1.0F + expf(-logit));
        sum               = fmaf(gate,
                                 __bfloat162float(normalized[position + static_cast<std::int64_t>(kHidden) *
                                                              (stream + kStreams * token)]),
                                 sum);
    }
    block_input[position + static_cast<std::int64_t>(kHidden) * token] =
        __float2bfloat16_rn(sum * 0.25F);
}

// injection[e, t] = bf16(sum over source streams s, in order, of the partials).
__device__ __forceinline__ void finish_injection(const float* partials, __nv_bfloat16* injection,
                                                 int destination, int token) {
    float sum = 0.0F;
#pragma unroll
    for (int source = 0; source < kStreams; ++source) {
        sum += partials[token * kInjectionPartialsPerToken + source * kStreams + destination];
    }
    injection[destination + kStreams * token] = __float2bfloat16_rn(sum);
}

} // namespace ninfer::ops::detail::hyperconnection
