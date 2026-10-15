#pragma once

// The block-diagonal Sylvester Hadamard of the SP-7 EXL3 contract (section 2.1): Hn_m holds
// m/128 copies of H128 / sqrt(128), H128[i][j] = (-1)^popcount(i AND j). One warp transforms
// one aligned 128-value block: each lane owns four consecutive values, the spans 1 and 2 pair
// inside that quartet, and the spans 4..64 pair across lanes by shuffle.

#include <cstdint>

namespace ninfer::ops::hadamard {

constexpr float kInverseSqrt128 = 0.08838834764831845F;

// In-place transform of one lane quartet: spans 1 and 2.
__device__ __forceinline__ void quartet_butterfly(float (&v)[4]) {
    {
        const float a = v[0], b = v[1];
        v[0] = a + b;
        v[1] = a - b;
    }
    {
        const float a = v[2], b = v[3];
        v[2] = a + b;
        v[3] = a - b;
    }
    {
        const float a = v[0], b = v[2];
        v[0] = a + b;
        v[2] = a - b;
    }
    {
        const float a = v[1], b = v[3];
        v[1] = a + b;
        v[3] = a - b;
    }
}

// One warp, one 128-value block: `load` fills the lane quartet [4*lane, 4*lane+4), the
// butterfly runs, and `store` receives the transformed quartet scaled by 1/sqrt(128).
template <class Load, class Store>
__device__ __forceinline__ void warp_transform_128(Load&& load, Store&& store) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float v[4];
    load(v);
    quartet_butterfly(v);
#pragma unroll
    for (int step = 1; step <= 16; step <<= 1) {
        float partner[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            partner[q] = __shfl_xor_sync(0xffffffffu, v[q], step);
        }
        const bool low = (lane & step) == 0;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            // The standard butterfly: (a, b) -> (a + b, a - b). The high lane's partner is
            // the low element, so its slot is partner - v.
            v[q] = low ? v[q] + partner[q] : partner[q] - v[q];
        }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) { v[q] *= kInverseSqrt128; }
    store(v);
}

} // namespace ninfer::ops::hadamard
