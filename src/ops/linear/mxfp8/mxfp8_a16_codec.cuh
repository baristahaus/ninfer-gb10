#pragma once

// MXFP8 operand helpers shared by the A16 (BF16-activation) routes.
//
// A stored group scale is one unsigned 8-bit UE8M0 exponent word per K-group of
// 32; the represented group scale is exactly 2^(word - 127), so folding it into
// the exact E4M3-to-BF16 widening is a power-of-two rescale computed in FP32.

#include "ops/linear/fp8/fp8_a16_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail {

__device__ __forceinline__ float2 decode_mxfp8_e4m3x2(std::uint16_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = storage;
    return static_cast<float2>(value);
}

__device__ __forceinline__ float mxfp8_group_scale(std::uint8_t word) {
    return ldexpf(1.0F, static_cast<int>(word) - 127);
}

// Widen two adjacent E4M3 codes of one group to BF16 with the group scale
// folded in. The pair never crosses a group boundary: callers pass even
// in-group column offsets.
__device__ __forceinline__ __nv_bfloat162
mxfp8_widen_pair_scaled(std::uint16_t packed_codes, float group_scale) {
    const unsigned widened = fp8_e4m3x2_to_bf16x2_bits(packed_codes);
    const float2 values    = bf16x2_bits_to_float2(widened);
    __nv_bfloat162 result;
    result.x = __float2bfloat16_rn(values.x * group_scale);
    result.y = __float2bfloat16_rn(values.y * group_scale);
    return result;
}

__device__ __forceinline__ int mxfp8_shared_col_64(int row, int col) {
    return fp8_a16_shared_col_64(row, col);
}

} // namespace ninfer::ops::detail
