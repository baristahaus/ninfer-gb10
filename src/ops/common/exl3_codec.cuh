#pragma once

// The EXL3 ring-decode and codebook device functions shared by every fused EXL3 consumer
// (docs/maintainer/exl3-trellis-contract.md sections 2.4 and 2.5, kernel contract section 7).
// Bitrates travel as k2 = 2K (2..16, or 3/5/7 for the MUL1-only half-integer rates). 3INST
// banks never reach a kernel: admission refuses them until a qualified route exists.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::exl3 {

constexpr int kMaxTileWords = 64;  // 4 * k2 at the k2 = 16 (K = 8) maximum

// One weight's 16-bit codebook state at sequence position `position` of a tile: the 16 ring
// bits ending at the element's bit-field end, ((position + 1) * 2K) / 2, cyclically (contract
// sections 2.4 and 7). Ring bits are MSB-first within each stored 32-bit word, so the staged
// words arrive bit-reversed and the window [offset, offset + 16) is read MSB-first here -
// the 16-bit reverse of the plain funnel window. Verified bit-exactly against the test-owned
// codec oracle at every rate and both codebooks.
__device__ __forceinline__ std::uint32_t state(const std::uint32_t* words,
                                               std::uint32_t words_per_tile,
                                               std::uint32_t k2, int position) {
    const std::uint32_t ring_bits = 128 * k2;
    const std::uint32_t end = ((static_cast<std::uint32_t>(position) + 1) * k2) >> 1;
    const std::uint32_t begin = end + ring_bits - 16;  // wraps once at most
    const std::uint32_t offset = begin & 31;
    const std::uint32_t first = (begin >> 5) % words_per_tile;
    const std::uint32_t last = ((begin + 15) >> 5) % words_per_tile;
    const std::uint64_t concat =
        (static_cast<std::uint64_t>(words[last]) << 32) | words[first];
    const std::uint32_t window = static_cast<std::uint32_t>((concat >> offset) & 0xffffu);
    return __brev(window) >> 16;
}

// Two mul1 codebook values from two precomputed state-multiplier products (contract 2.5).
// The FP16 fused multiply-add rounds once, matching the exact oracle's single rounding;
// the dp4a folds the byte sum into the 1024.0 bit-pattern base.
__device__ __forceinline__ __half2 decode_mul1_2(std::uint32_t x0, std::uint32_t x1) {
    const std::uint32_t sum0 = __dp4a(x0, 0x01010101u, 0x6400u);
    const std::uint32_t sum1 = __dp4a(x1, 0x01010101u, 0x6400u);
    const __half2 k_inv = __half2half2(__ushort_as_half(0x1eee));
    const __half2 k_bias = __half2half2(__ushort_as_half(0xc931));
    const __half2 h = __halves2half2(__ushort_as_half(static_cast<std::uint16_t>(sum0)),
                                     __ushort_as_half(static_cast<std::uint16_t>(sum1)));
    return __hfma2(h, k_inv, k_bias);
}

// Two mcg codebook values: the lop3-masked product halves summed in FP16.
__device__ __forceinline__ __half2 decode_mcg_2(std::uint32_t x0, std::uint32_t x1) {
    x0 = (x0 & 0x8fff8fffu) ^ 0x3b603b60u;
    x1 = (x1 & 0x8fff8fffu) ^ 0x3b603b60u;
    const __half2 halves0 =
        __halves2half2(__ushort_as_half(static_cast<std::uint16_t>(x0)),
                       __ushort_as_half(static_cast<std::uint16_t>(x0 >> 16)));
    const __half2 halves1 =
        __halves2half2(__ushort_as_half(static_cast<std::uint16_t>(x1)),
                       __ushort_as_half(static_cast<std::uint16_t>(x1 >> 16)));
    return __hadd2(halves0, halves1);
}

// One weight's FP16 value from its 16-bit state through the bank's codebook (contract 2.5).
// The FP16 add and fused multiply-add each round once, matching the exact oracle's single
// rounding.
__device__ __forceinline__ float weight(std::uint32_t state, std::uint32_t codebook) {
    if (codebook == 1) {  // mcg
        const std::uint32_t x = (state * 0xCBAC1FEDu & 0x8fff8fffu) ^ 0x3b603b60u;
        const __half low  = __ushort_as_half(static_cast<std::uint16_t>(x));
        const __half high = __ushort_as_half(static_cast<std::uint16_t>(x >> 16));
        return __half2float(__hadd(low, high));
    }
    // mul1 (codebook 2)
    const std::uint32_t sum = __dp4a(state * 0x83DCD12Du, 0x01010101u, 0x6400u);
    const __half h          = __ushort_as_half(static_cast<std::uint16_t>(sum));
    return __half2float(__hfma(h, __ushort_as_half(static_cast<std::uint16_t>(0x1eee)),
                               __ushort_as_half(static_cast<std::uint16_t>(0xc931))));
}

// The tile element (r, c) that sequence position i decodes (contract section 2.3).
__device__ __forceinline__ void tile_element(int position, int& r, int& c) {
    const int t = position >> 3, j = position & 7;
    r = 2 * (t & 3) + (j & 1) + 8 * ((j >> 1) & 1);
    c = (t >> 2) + 8 * ((j >> 2) & 1);
}

} // namespace ninfer::ops::exl3
