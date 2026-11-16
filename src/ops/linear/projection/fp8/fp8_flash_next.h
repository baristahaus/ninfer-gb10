#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail::flash_next {

// Row-scaled FP8 counterparts of the fused BF16 Flash-Next projections, on the same routes as the
// exact Linear problems and differing only in their epilogue or output mapping.
// The HyperConnection down projection with its scaled SiLU, at any token count: out is
// BF16 [320, T].
void launch_fp8_hc_down_silu(const Tensor& x, const Weight& weight, Tensor& out,
                             cudaStream_t stream);
// The HyperConnection up projection at 2..64 tokens with the gate mix fused into its epilogue:
// writes block_input BF16 [2560, T] from low_rank BF16 [320, T] and normalized BF16 [10240, T],
// and, when given, finishes injection BF16 [4, T] from injection_partials FP32 [16, T].
void launch_fp8_hc_up_mix(const Tensor& low_rank, const Weight& weight, const Tensor& normalized,
                          const Tensor* injection_partials, Tensor& block_input, Tensor* injection,
                          cudaStream_t stream);
// The QSA packed query/gate projection split into query and gate heads at T=1.
void launch_fp8_query_gate_decode(const Tensor& x, const Weight& weight, Tensor& query,
                                  Tensor& gate, cudaStream_t stream);

} // namespace ninfer::ops::detail::flash_next
