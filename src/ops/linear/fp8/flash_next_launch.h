#pragma once
#include "core/weight.h"

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail::flash_next {

// One-token MoE entry for row-scaled FP8 weights: router scores [512] and the shared-expert
// SwiGLU activation [640] from one BF16 input in a single grid. Each projection keeps the
// production FP8 GEMV accumulation; gate and up are rounded to BF16 before SwiGLU, as in the
// unfused linear + silu_mul route.
void launch_fp8_moe_entry_decode(const Tensor& x, const Weight& router, const Weight& shared_gate,
                                 const Weight& shared_up, Tensor& scores, Tensor& activation,
                                 cudaStream_t stream);

} // namespace ninfer::ops::detail::flash_next
