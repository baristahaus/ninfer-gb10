// The dense EXL3 Linear route's fused decode kernels (SP-7 contract sections 4 and 7).
//
// The logical weight is W[n, k] = (diag(suh) Hn_k W_hat Hn_n diag(svh))^T, where W_hat is the
// FP16 inner weight decoded from the trellis (contract section 2.1). The kernels never
// materialize W_hat (section 7): stage 1 decodes tiles on the fly and accumulates FP32 partial
// inner products z = (x ⊙ suh) Hn_k W_hat per (k chunk, 128-output block, token); stage 2
// reduces the chunks, applies the output-side Hn_n ⊙ svh, and writes the BF16 destination.
// Activations are represented BF16 read as FP32; decoded weights are FP16 promoted exactly to
// FP32; every accumulator is FP32. The route is A16 compute under every LinearPolicy.
//
// Tensor layout follows the Linear contract: dimension zero is stored fastest, so x (k, t)
// addresses k + t*K and out (n, t) addresses n + t*N.

#include "ops/linear/exl3/exl3_dense_dispatch.h"

#include "ops/common/exl3_codec.cuh"
#include "ops/common/hadamard.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr int kChunkKBlocks    = 4;    // 128-input Hadamard blocks per stage-1 block
constexpr int kMaxTokens       = 8;
constexpr int kThreads         = 256;  // 8 warps: up to one token transform per warp

// Stage 1: grid (n/128, ceil(k/512)), 256 threads. Block (ob, chunk) accumulates the FP32
// partial z for outputs [128*ob, 128*ob + 128), the chunk's k inputs, and every token.
__global__ void linear_exl3_partial_kernel(const __nv_bfloat16* __restrict__ x,
                                           const std::uint8_t* __restrict__ payload,
                                           float* __restrict__ partial, int n, int k,
                                           int tokens, int k_blocks_total,
                                           std::uint32_t k2, std::uint32_t codebook,
                                           std::size_t trellis_bytes) {
    const int ob = static_cast<int>(blockIdx.x);
    const int chunk = static_cast<int>(blockIdx.y);
    const int first_k_block = chunk * kChunkKBlocks;
    const int k_blocks = min(kChunkKBlocks, k_blocks_total - first_k_block);
    if (k_blocks <= 0) { return; }

    __shared__ float u_sh[128 * kMaxTokens];
    __shared__ float acc[128 * kMaxTokens];
    __shared__ float tiles_sh[8][256];            // the k-tile row's eight decoded 16x16 tiles
    __shared__ float tile_partial[kMaxTokens * 256];
    __shared__ std::uint32_t tile_ring[exl3::kMaxTileWords];
    const std::uint32_t words_per_tile = 4 * k2;
    const std::uint32_t* trellis = reinterpret_cast<const std::uint32_t*>(payload);
    const __half* suh = reinterpret_cast<const __half*>(payload + trellis_bytes);
    const int tiles_n = n / 16;

    for (int i = static_cast<int>(threadIdx.x); i < 128 * tokens; i += kThreads) {
        acc[i] = 0.0F;
    }
    __syncthreads();

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    for (int kb = 0; kb < k_blocks; ++kb) {
        const int k_block = first_k_block + kb;
        const int k_base = k_block * 128;
        // Input transform: u = H128 (x_block ⊙ suh_block) per token, one warp per token.
        if (warp < tokens) {
            const __nv_bfloat16* x_block =
                x + static_cast<std::size_t>(k_base) + static_cast<std::size_t>(warp) * k;
            hadamard::warp_transform_128(
                [&](float (&v)[4]) {
                    const int base = 4 * (static_cast<int>(threadIdx.x) & 31);
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        v[q] = __bfloat162float(x_block[base + q]) *
                               __half2float(suh[k_base + base + q]);
                    }
                },
                [&](const float (&v)[4]) {
                    const int base = 4 * (static_cast<int>(threadIdx.x) & 31);
#pragma unroll
                    for (int q = 0; q < 4; ++q) { u_sh[warp * 128 + base + q] = v[q]; }
                });
        }
        __syncthreads();

        // This 128-input block's 8 k-tiles against this output block's 8 n-tiles: the eight
        // tiles are decoded into shared first, then one MAC pass assigns each output column
        // the products of its own tile only.
        const int k_tile_base = k_block * 8;
        for (int kt = 0; kt < 8; ++kt) {
            for (int b4 = 0; b4 < 8; ++b4) {
                const std::size_t tile = static_cast<std::size_t>(k_tile_base + kt) *
                                             static_cast<std::size_t>(tiles_n) +
                                         static_cast<std::size_t>(ob * 8 + b4);
                // Stage the tile's ring words bit-reversed (the state reader's convention,
                // as in the qualified MoE kernels), then decode: one sequence position per
                // thread, scattered into the 16x16 tile.
                const std::uint32_t* words = trellis + tile * words_per_tile;
                if (static_cast<int>(threadIdx.x) < static_cast<int>(words_per_tile)) {
                    tile_ring[threadIdx.x] = __brev(words[threadIdx.x]);
                }
                __syncthreads();
                const int position = static_cast<int>(threadIdx.x);
                const std::uint32_t state = exl3::state(tile_ring, words_per_tile, k2, position);
                int r, c;
                exl3::tile_element(position, r, c);
                tiles_sh[b4][r * 16 + c] = exl3::weight(state, codebook);
                __syncthreads();
            }
            // MAC: thread out_idx = (b4, c) accumulates its own tile's k-row products against
            // u; the two threads of an output split the 16 rows and combine through the
            // shared partial buffer.
            const int out_idx = static_cast<int>(threadIdx.x) & 127;
            const int col = out_idx & 15;               // n-within-tile column c
            const int my_b4 = out_idx >> 4;
            const int half = static_cast<int>(threadIdx.x) >> 7;
            for (int t = 0; t < tokens; ++t) {
                float sum = 0.0F;
                for (int r = half * 8; r < half * 8 + 8; ++r) {
                    sum += tiles_sh[my_b4][r * 16 + col] * u_sh[t * 128 + kt * 16 + r];
                }
                tile_partial[t * 256 + out_idx + half * 128] = sum;
            }
            __syncthreads();
            if (half == 0) {
                for (int t = 0; t < tokens; ++t) {
                    acc[out_idx * tokens + t] +=
                        tile_partial[t * 256 + out_idx] + tile_partial[t * 256 + 128 + out_idx];
                }
            }
            __syncthreads();
        }
    }

    // Stage the chunk's partial z: [(chunk, n), tokens], dimension zero fastest.
    float* destination = partial + (static_cast<std::size_t>(chunk) * n +
                                    static_cast<std::size_t>(ob) * 128) * tokens;
    for (int i = static_cast<int>(threadIdx.x); i < 128 * tokens; i += kThreads) {
        destination[i] = acc[i];
    }
}

// Stage 2: grid (n/128), 256 threads. Reduces the k chunks, applies the output-side
// Hn_n ⊙ svh, and writes the BF16 destination.
__global__ void linear_exl3_finish_kernel(const float* __restrict__ partial,
                                          const std::uint8_t* __restrict__ payload,
                                          __nv_bfloat16* __restrict__ out, int n, int tokens,
                                          int chunks, std::size_t svh_offset) {
    const int ob = static_cast<int>(blockIdx.x);
    __shared__ float z_sh[128 * kMaxTokens];
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const __half* svh = reinterpret_cast<const __half*>(payload + svh_offset);

    // Reduce: thread (row = tid & 127, parity = tid >> 7) sums the chunk partials, tokens
    // strided by two so every token count up to eight is covered.
    const int row = static_cast<int>(threadIdx.x) & 127;
    const int parity = static_cast<int>(threadIdx.x) >> 7;
    for (int t = parity; t < tokens; t += 2) {
        float sum = 0.0F;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            sum += partial[(static_cast<std::size_t>(chunk) * n +
                            static_cast<std::size_t>(ob) * 128 + static_cast<std::size_t>(row)) *
                               tokens +
                           t];
        }
        z_sh[t * 128 + row] = sum;
    }
    __syncthreads();

    // Output transform: y = (H128 z) ⊙ svh per token, one warp per token.
    if (warp < tokens) {
        hadamard::warp_transform_128(
            [&](float (&v)[4]) {
                const int base = 4 * (static_cast<int>(threadIdx.x) & 31);
#pragma unroll
                for (int q = 0; q < 4; ++q) { v[q] = z_sh[warp * 128 + base + q]; }
            },
            [&](const float (&v)[4]) {
                const int base = 4 * (static_cast<int>(threadIdx.x) & 31);
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int n_row = ob * 128 + base + q;
                    out[n_row + warp * n] =
                        __float2bfloat16(v[q] * __half2float(svh[n_row]));
                }
            });
    }
}

} // namespace

void launch_exl3_dense(const Tensor& x, const Weight& w, Tensor& out, void* workspace,
                       cudaStream_t stream) {
    const int n = w.n;
    const int k = w.k;
    const int tokens = out.ne[1];
    if (tokens <= 0 || tokens > kMaxTokens) {
        throw std::invalid_argument("exl3 dense linear: registers decode shapes T in 1..8, got " +
                                    std::to_string(tokens));
    }
    if (n % 128 != 0 || k % 128 != 0) {
        throw std::invalid_argument("exl3 dense linear: dimensions must be multiples of 128");
    }
    // Dense payload planes (contract section 4): the trellis at offset 0 with its aligned
    // extent, then suh [k] and svh [n]. The aligned extent also determines the bitrate:
    // admissible bitrates differ by k*n/16 bytes (at least 1024), far above the 256-byte
    // alignment, so 16*extent/(k*n) is exact.
    const std::size_t trellis_bytes = w.exl3_suh_offset;
    if (w.exl3_trellis_offset != 0 || trellis_bytes == 0) {
        throw std::invalid_argument("exl3 dense linear: payload planes are not dense layout");
    }
    const std::size_t extent_pair = trellis_bytes * 16;
    if (extent_pair % (static_cast<std::size_t>(n) * static_cast<std::size_t>(k)) != 0) {
        throw std::invalid_argument("exl3 dense linear: trellis extent matches no bitrate");
    }
    const std::uint32_t k2 = static_cast<std::uint32_t>(
        extent_pair / (static_cast<std::size_t>(n) * static_cast<std::size_t>(k)));
    if (k2 < 2 || k2 > 16) {
        throw std::invalid_argument("exl3 dense linear: bitrate 2K = " + std::to_string(k2) +
                                    " outside the admitted range");
    }

    const auto* activation = static_cast<const __nv_bfloat16*>(x.data);
    auto* destination = static_cast<__nv_bfloat16*>(out.data);
    auto* partial = static_cast<float*>(workspace);
    const auto* payload = static_cast<const std::uint8_t*>(w.payload);

    const int chunks = (k + kChunkKBlocks * 128 - 1) / (kChunkKBlocks * 128);
    const int k_blocks = k / 128;
    dim3 grid(static_cast<unsigned>(n / 128), static_cast<unsigned>(chunks));
    linear_exl3_partial_kernel<<<grid, kThreads, 0, stream>>>(
        activation, payload, partial, n, k, tokens,
        k_blocks, k2, w.exl3_codebook, trellis_bytes);
    linear_exl3_finish_kernel<<<static_cast<unsigned>(n / 128), kThreads, 0, stream>>>(
        partial, payload, destination, n, tokens, chunks,
        trellis_bytes + static_cast<std::size_t>(k) * 2);
}

} // namespace ninfer::ops::detail
