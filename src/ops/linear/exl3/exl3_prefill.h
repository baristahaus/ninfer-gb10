#pragma once

// The dense EXL3 prefill composition (SP-7 contract: the prefill token route). For token
// counts above the register route's domain (T <= 8) the callers compose the linear from
// the decoded weight and a BF16 GEMM; both pieces here are transient per call.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer::ops::detail {

// Transient bytes of one composed projection: the decoded W'[n, k] BF16 plane plus the
// transformed activation u[k, tokens] BF16.
[[nodiscard]] std::size_t exl3_prefill_workspace_bytes(std::int32_t n, std::int32_t k,
                                                       std::int32_t tokens);

// Decodes the dense weight (a complete object or a row slice) into `out` [n, k] BF16 with
// the output-side 128 Hadamard and svh folded in; the input-side transform stays with
// `exl3_prefill_transform`.
void exl3_prefill_dequant(const Weight& w, Tensor& out, cudaStream_t stream);

// Writes u[k, T] = (H128 (x suh))[k, T] per 128-input block; x and u are [k, T] BF16.
void exl3_prefill_transform(const Tensor& x, const Weight& w, Tensor& u, cudaStream_t stream);

} // namespace ninfer::ops::detail
