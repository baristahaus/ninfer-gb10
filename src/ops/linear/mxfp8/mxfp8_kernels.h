#pragma once

// Dense MXFP8 (dense_block_scale_k32_mxfp8_v1) x BF16-activation linear.
//
//   out[T,N] = x[T,K] * W[N,K]^T
//
// The weight words are decoded in the kernel: E4M3FN code * 2^(UE8M0 - 127)
// per K-group of 32, accumulated in FP32. This is the bounded fallback route;
// the stored format and the oracle stay MXFP8 (no conversion-side dequant).

#include <cstdint>

namespace ninfer::ops::detail {

struct Tensor;
struct Weight;

// Small-token SIMT route (decode per weight word, one warp per output row).
void mxfp8_gemv_launch(const struct Tensor& x, const struct Weight& weight, struct Tensor& out,
                       cudaStream_t stream);

// Tensor-core route: dequant weight tiles to BF16 in shared memory, BF16 MMA,
// FP32 accumulation. Handles arbitrary row/token extents.
void mxfp8_mma_launch(const struct Tensor& x, const struct Weight& weight, struct Tensor& out,
                      cudaStream_t stream);

bool mxfp8_native_weight_ok(const struct Weight& weight);

} // namespace ninfer::ops::detail
