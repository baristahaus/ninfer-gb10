// The dense EXL3 prefill composition (SP-7 contract: the prefill token route). Decode
// token shapes (T <= 8) run the register route; larger token counts compose the same
// logical formula from the decoded weight and a BF16 GEMM:
//
//   out = D_svh Hn (W_hat)^T Hk D_suh x   (per 128 block on both sides)
//
// The composition materializes W'[n, k] = (W_hat^T Hn D_svh) per 128-row block (the
// output-side transform folded into the decoded matrix) and u[k, t] = (Hk (x suh))[k, t]
// (the input-side transform applied to the activation), then runs the registered BF16
// linear route. Both pieces are transient per call; nothing persists.

#include "ops/linear/exl3/exl3_prefill.h"

#include "ops/linear/exl3/exl3_dense_dispatch.h"

#include "ops/common/hadamard.cuh"

#include <cuda_bf16.h>
#include "ops/common/exl3_codec.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kThreads = 256;  // 8 warps: the tile's sixteen k rows, two per warp

// Grid (n/128, k/16): one block decodes the eight n-tiles of one (k-tile, n-block) and
// writes the n-transformed, svh-scaled rows W'[n0..n0+128, k0..k0+16).
__global__ void exl3_dequant_kernel(const std::uint8_t* __restrict__ payload, int n, int k,
                                    std::uint32_t k2, std::uint32_t codebook, int tiles_stride,
                                    std::size_t svh_offset, __nv_bfloat16* __restrict__ out) {
    const int n_block = static_cast<int>(blockIdx.x);
    const int k_tile  = static_cast<int>(blockIdx.y);
    const int warp    = static_cast<int>(threadIdx.x) >> 5;
    const std::uint32_t words_per_tile = 4 * k2;
    const std::uint32_t* trellis = reinterpret_cast<const std::uint32_t*>(payload);
    const __half* svh = reinterpret_cast<const __half*>(payload + svh_offset);
    __shared__ float tiles_sh[8][256];  // eight n-tiles of this (k-tile, n-block)
    __shared__ std::uint32_t tile_ring[exl3::kMaxTileWords];

    const int k0 = k_tile * 16;
    // The payload pointer sits at the slice's first trellis tile and the svh pointer at
    // the slice's first parent row; every index here is slice-local.
    const int n0 = n_block * 128;
    for (int nt = 0; nt < 8; ++nt) {
        // The payload pointer already sits at the slice's first trellis tile, so the
        // tile index is slice-local: k-tile row stride of the parent's tile count.
        const std::size_t tile = static_cast<std::size_t>(k_tile) *
                                     static_cast<std::size_t>(tiles_stride) +
                                 static_cast<std::size_t>(n_block * 8 + nt);
        const std::uint32_t* words = trellis + tile * words_per_tile;
        if (static_cast<int>(threadIdx.x) < static_cast<int>(words_per_tile)) {
            tile_ring[threadIdx.x] = __brev(words[threadIdx.x]);
        }
        __syncthreads();
        const int position = static_cast<int>(threadIdx.x);
        const std::uint32_t state = exl3::state(tile_ring, words_per_tile, k2, position);
        int r, c;
        exl3::tile_element(position, r, c);
        tiles_sh[nt][r * 16 + c] = exl3::weight(state, codebook);
        __syncthreads();
    }
    // Each warp transforms two of the tile's sixteen k rows: the 128-vector of decoded
    // weights (one value per parent n) scaled by svh, through the 128 Hadamard, straight
    // to the destination's transposed [n, k] layout.
    for (int pass = 0; pass < 2; ++pass) {
        const int row = warp + pass * 8;
        hadamard::warp_transform_128(
            [&](float (&v)[4]) {
                const int base = 4 * (static_cast<int>(threadIdx.x) & 31);
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int m = base + q;  // n within the 128 block
                    v[q] = tiles_sh[m / 16][row * 16 + m % 16];
                }
            },
            [&](const float (&v)[4]) {
                const int base = 4 * (static_cast<int>(threadIdx.x) & 31);
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    // The output-side transform: the Hadamard contracts the decoded row,
                    // then svh scales the result row (diagonal outside the transform, as
                    // in the finish kernel). The destination rows are the weight's own
                    // (slice-local) space; svh addresses the parent's rows.
                    out[static_cast<std::size_t>(n_block * 128 + base + q) *
                            static_cast<std::size_t>(k) +
                        static_cast<std::size_t>(k0 + row)] =
                        __float2bfloat16(v[q] *
                                         __half2float(svh[static_cast<std::size_t>(n0 + base + q)]));
                }
            });
    }
}

// Grid (ceil(T/8), k/128): one warp per token transforms u[k0..k0+128, t] = H(x suh).
__global__ void exl3_input_transform_kernel(const __nv_bfloat16* __restrict__ x,
                                            const std::uint8_t* __restrict__ payload,
                                            __nv_bfloat16* __restrict__ u, int k, int tokens,
                                            std::size_t suh_offset) {
    const int k0   = static_cast<int>(blockIdx.y) * 128;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int token = static_cast<int>(blockIdx.x) * 8 + warp;
    if (token >= tokens) { return; }
    const __half* suh = reinterpret_cast<const __half*>(payload + suh_offset);
    const __nv_bfloat16* x_block = x + static_cast<std::size_t>(k0) +
                                   static_cast<std::size_t>(token) * static_cast<std::size_t>(k);
    __nv_bfloat16* u_block = u + static_cast<std::size_t>(k0) +
                             static_cast<std::size_t>(token) * static_cast<std::size_t>(k);
    hadamard::warp_transform_128(
        [&](float (&v)[4]) {
            const int base = 4 * (static_cast<int>(threadIdx.x) & 31);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                v[q] = __bfloat162float(x_block[base + q]) * __half2float(suh[k0 + base + q]);
            }
        },
        [&](const float (&v)[4]) {
            const int base = 4 * (static_cast<int>(threadIdx.x) & 31);
#pragma unroll
            for (int q = 0; q < 4; ++q) { u_block[base + q] = __float2bfloat16(v[q]); }
        });
}

} // namespace

std::size_t exl3_prefill_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t tokens) {
    if (n <= 0 || k <= 0 || tokens <= 0) {
        throw std::invalid_argument("exl3 prefill workspace: invalid shapes");
    }
    // W'[n, k] BF16 plus u[k, tokens] BF16.
    return static_cast<std::size_t>(n) * static_cast<std::size_t>(k) * 2 +
           static_cast<std::size_t>(k) * static_cast<std::size_t>(tokens) * 2;
}

void exl3_prefill_dequant(const Weight& w, Tensor& out, cudaStream_t stream) {
    const int n = w.n;
    const int k = w.k;
    if (n % 128 != 0 || k % 16 != 0 || w.exl3_experts != 1 ||
        w.layout != QuantLayout::Exl3Dense) {
        throw std::invalid_argument("exl3 prefill dequant: unsupported dense weight");
    }
    if (w.exl3_codebook == 0) {
        throw std::invalid_argument("exl3 prefill dequant: 3INST has no qualified route");
    }
    const std::uint32_t k2 = exl3_dense_k2(w);
    const int row_stride = w.exl3_row_stride != 0 ? static_cast<int>(w.exl3_row_stride) : n;
    const int row_begin = static_cast<int>(
        (w.exl3_svh_offset - (w.exl3_suh_offset + static_cast<std::size_t>(k) * 2)) / 2);
    const auto* payload = static_cast<const std::uint8_t*>(w.payload) + w.exl3_trellis_offset;
    auto* destination = static_cast<__nv_bfloat16*>(out.data);
    dim3 grid(static_cast<unsigned>(n / 128), static_cast<unsigned>(k / 16));
    exl3_dequant_kernel<<<grid, kThreads, 0, stream>>>(
        payload, n, k, k2, w.exl3_codebook, row_stride / 16,
        w.exl3_svh_offset - w.exl3_trellis_offset, destination);
}

void exl3_prefill_transform(const Tensor& x, const Weight& w, Tensor& u, cudaStream_t stream) {
    const int k = w.k;
    const int tokens = x.ne[1];
    if (x.ne[0] != k || u.ne[0] != k || u.ne[1] != tokens || tokens <= 0) {
        throw std::invalid_argument("exl3 prefill transform: shapes disagree");
    }
    const auto* payload = static_cast<const std::uint8_t*>(w.payload) + w.exl3_trellis_offset;
    dim3 grid(static_cast<unsigned>((tokens + 7) / 8), static_cast<unsigned>(k / 128));
    exl3_input_transform_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), payload,
        static_cast<__nv_bfloat16*>(u.data), k, tokens,
        w.exl3_suh_offset - w.exl3_trellis_offset);
}

} // namespace ninfer::ops::detail
