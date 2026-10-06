#pragma once
#include "core/weight.h"

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail::flash_next {

inline constexpr int kFp8MoeEntryMaxTokens = 16;

// MoE entry for row-scaled FP8 weights and 1-16 tokens: router scores [512,T] and the shared-expert
// SwiGLU activation [640,T] from one BF16 input in a single grid. Each projection keeps the
// production FP8 route of its token count (GEMV for one token, K-split for 2-16), so results equal
// the separate linears; gate and up are rounded to BF16 before SwiGLU, as in linear + silu_mul.
void launch_fp8_moe_entry_decode(const Tensor& x, const Weight& router, const Weight& shared_gate,
                                 const Weight& shared_up, Tensor& scores, Tensor& activation,
                                 cudaStream_t stream);

} // namespace ninfer::ops::detail::flash_next
