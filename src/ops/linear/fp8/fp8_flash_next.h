#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail::flash_next {

// T=1 row-scaled FP8 counterparts of the fused BF16 Flash-Next decode projections. They share the
// FP8 GEMV mainloop of the exact linear problems and differ only in their output mapping.
void launch_fp8_hc_down_silu_decode(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream);
void launch_fp8_query_gate_decode(const Tensor& x, const Weight& weight, Tensor& query,
                                  Tensor& gate, cudaStream_t stream);

} // namespace ninfer::ops::detail::flash_next
