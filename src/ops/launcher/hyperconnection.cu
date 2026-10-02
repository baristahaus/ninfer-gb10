#include "ninfer/ops/hyperconnection.h"

#include "ops/flash_next_work.h"

#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/bf16/flash_next/bf16_launch.h"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::ops {
namespace {

constexpr int kStreams = 4;
constexpr int kHidden = 2560;
constexpr int kHyper = kStreams * kHidden;
constexpr int kRank = 320;

__global__ void repeat_kernel(const __nv_bfloat16* input, __nv_bfloat16* hyper,
                              std::int64_t count) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const int d = static_cast<int>(i % kHyper);
        const std::int64_t token = i / kHyper;
        hyper[i] = input[(d % kHidden) + static_cast<std::int64_t>(kHidden) * token];
    }
}

__global__ void add_repeated_kernel(const __nv_bfloat16* embedding, __nv_bfloat16* hyper,
                                    std::int64_t count) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const int d = static_cast<int>(i % kHyper);
        const std::int64_t token = i / kHyper;
        hyper[i] = __float2bfloat16_rn(
            __bfloat162float(hyper[i]) +
            __bfloat162float(embedding[(d % kHidden) +
                                       static_cast<std::int64_t>(kHidden) * token]));
    }
}

// Grouped RMSNorm kernels: one 256-thread block per (stream lane, token), each thread holding
// kPerThread values of the lane in registers.
constexpr int kNormThreads = 256;
constexpr int kPerThread   = kHidden / kNormThreads;
static_assert(kHidden % kNormThreads == 0);

__device__ __forceinline__ float block_sum(float value, float* shared) {
    for (int shift = 16; shift; shift >>= 1) value += __shfl_down_sync(0xffffffffU, value, shift);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (!lane) shared[warp] = value;
    __syncthreads();
    if (!warp) {
        value = lane < kNormThreads / 32 ? shared[lane] : 0;
        for (int shift = 16; shift; shift >>= 1)
            value += __shfl_down_sync(0xffffffffU, value, shift);
        if (!lane) shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}

// Activation steering of one residual lane: y = x - a * sum_k dot(x, d_k) d_k with simultaneous
// dot products from the represented input, optional norm preservation, one final BF16 boundary.
// Strength, rank and lane mask are device data, so the decision is uniform per block and needs
// no host involvement. Returns false (x untouched, no arithmetic) when the lane is not steered.
__device__ __forceinline__ bool steer_lane(float (&x)[kPerThread], const ActivationDevice* c,
                                           int layer, int width, int lane, int token,
                                           float* shared) {
    if (c == nullptr) { return false; }
    const float strength = c->rows[token / width].strength;
    const int rank       = c->ranks[layer];
    if (strength == 0 || rank == 0 || !(c->masks[layer] & (1 << lane))) { return false; }
    float before = 0, delta[kPerThread]{};
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) before = fmaf(x[j], x[j], before);
    for (int r = 0; r < rank; ++r) {
        const float* d =
            c->directions + ((layer * kSteeringMaxRank + r) * kStreams + lane) * kHidden;
        float dot = 0;
#pragma unroll
        for (int j = 0; j < kPerThread; ++j)
            dot = fmaf(x[j], d[threadIdx.x + j * kNormThreads], dot);
        dot = block_sum(dot, shared);
#pragma unroll
        for (int j = 0; j < kPerThread; ++j)
            delta[j] = fmaf(dot, d[threadIdx.x + j * kNormThreads], delta[j]);
        __syncthreads();
    }
    float y[kPerThread], after = 0;
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        y[j]  = x[j] - strength * delta[j];
        after = fmaf(y[j], y[j], after);
    }
    float scale = 1;
    if (c->norm_preserve) {
        before = block_sum(before, shared);
        __syncthreads();
        after = block_sum(after, shared);
        // A zero projection has no direction to rescale; preserve zero.
        if (after > 0) scale = sqrtf(before / after);
    }
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) x[j] = __bfloat162float(__float2bfloat16_rn(y[j] * scale));
    return true;
}

// RMS-normalizes represented lane values held in registers and writes the normalized row.
__device__ __forceinline__ void normalize_lane(const float (&x)[kPerThread],
                                               const __nv_bfloat16* weight,
                                               __nv_bfloat16* normalized, std::int64_t base,
                                               int stream) {
    float sum = 0.0F;
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) sum = fmaf(x[j], x[j], sum);
    for (int offset = 16; offset != 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    __shared__ float partial[8];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { partial[warp] = sum; }
    __syncthreads();
    if (warp == 0) {
        float value = lane < 8 ? partial[lane] : 0.0F;
        for (int offset = 16; offset != 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        if (lane == 0) { partial[0] = rsqrtf(value / static_cast<float>(kHidden) + 1.0e-6F); }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const int d       = static_cast<int>(threadIdx.x) + j * kNormThreads;
        const float scale = 1.0F + __bfloat162float(weight[kHidden * stream + d]);
        normalized[base + d] = __float2bfloat16_rn(x[j] * partial[0] * scale);
    }
}

// Steering (when active) adjusts hyper in place before normalization.
__global__ void grouped_rmsnorm_kernel(__nv_bfloat16* hyper, const __nv_bfloat16* weight,
                                       __nv_bfloat16* normalized,
                                       const ActivationDevice* steering, int layer, int width) {
    const int stream = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const std::int64_t base = static_cast<std::int64_t>(kHyper) * token + kHidden * stream;
    float x[kPerThread];
#pragma unroll
    for (int j = 0; j < kPerThread; ++j)
        x[j] = __bfloat162float(hyper[base + threadIdx.x + j * kNormThreads]);
    __shared__ float steer_shared[kNormThreads / 32];
    if (steer_lane(x, steering, layer, width, stream, token, steer_shared)) {
#pragma unroll
        for (int j = 0; j < kPerThread; ++j)
            hyper[base + threadIdx.x + j * kNormThreads] = __float2bfloat16_rn(x[j]);
    }
    normalize_lane(x, weight, normalized, base, stream);
}

// Commits the pending branch output at the represented BF16 combine boundary, steers it when
// active, writes hyper once, and normalizes the final represented values.
__global__ void combine_grouped_rmsnorm_kernel(
    __nv_bfloat16* hyper, const __nv_bfloat16* block,
    const __nv_bfloat16* injection, const __nv_bfloat16* weight,
    __nv_bfloat16* normalized, const ActivationDevice* steering, int layer, int width) {
    const int stream = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const std::int64_t base = static_cast<std::int64_t>(kHyper) * token + kHidden * stream;
    const float logit = __bfloat162float(injection[stream + kStreams * token]) * 0.25F;
    const float branch_scale = 2.0F / (1.0F + expf(-logit));
    float x[kPerThread];
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const int d = static_cast<int>(threadIdx.x) + j * kNormThreads;
        x[j] = __bfloat162float(__float2bfloat16_rn(
            __bfloat162float(hyper[base + d]) +
            branch_scale * __bfloat162float(block[d + kHidden * token])));
    }
    __shared__ float steer_shared[kNormThreads / 32];
    steer_lane(x, steering, layer, width, stream, token, steer_shared);
#pragma unroll
    for (int j = 0; j < kPerThread; ++j)
        hyper[base + threadIdx.x + j * kNormThreads] = __float2bfloat16_rn(x[j]);
    normalize_lane(x, weight, normalized, base, stream);
}

__global__ void scaled_silu_kernel(__nv_bfloat16* values, std::int64_t count) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const float value = __bfloat162float(values[i]) * 0.25F;
        values[i] = __float2bfloat16_rn(value / (1.0F + expf(-value)));
    }
}

__global__ void gate_mix_kernel(const __nv_bfloat16* normalized,
                                const __nv_bfloat16* gate_logits,
                                __nv_bfloat16* block_input, int tokens) {
    const std::int64_t count = static_cast<std::int64_t>(kHidden) * tokens;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const int d = static_cast<int>(i % kHidden);
        const int t = static_cast<int>(i / kHidden);
        float sum = 0.0F;
#pragma unroll
        for (int stream = 0; stream < kStreams; ++stream) {
            const std::int64_t offset = d + static_cast<std::int64_t>(kHidden) *
                (stream + static_cast<std::int64_t>(kStreams) * t);
            const float gate = 1.0F / (1.0F + expf(-__bfloat162float(gate_logits[offset])));
            sum = fmaf(gate, __bfloat162float(normalized[offset]), sum);
        }
        block_input[i] = __float2bfloat16_rn(sum * 0.25F);
    }
}

__global__ void gate_mix_injection_kernel(const __nv_bfloat16* normalized,
                                          const __nv_bfloat16* gate_logits,
                                          const __nv_bfloat16* injection_weight,
                                          __nv_bfloat16* block_input,
                                          __nv_bfloat16* injection) {
    const int token = static_cast<int>(blockIdx.x);
    float injection_sum[kStreams] = {};
    for (int d = static_cast<int>(threadIdx.x); d < kHidden;
         d += static_cast<int>(blockDim.x)) {
        float mixed = 0.0F;
#pragma unroll
        for (int source_stream = 0; source_stream < kStreams; ++source_stream) {
            const int k = source_stream * kHidden + d;
            const std::int64_t offset = k + static_cast<std::int64_t>(kHyper) * token;
            const float value = __bfloat162float(normalized[offset]);
            const float gate =
                1.0F / (1.0F + expf(-__bfloat162float(gate_logits[offset])));
            mixed = fmaf(gate, value, mixed);
#pragma unroll
            for (int destination_stream = 0; destination_stream < kStreams;
                 ++destination_stream) {
                injection_sum[destination_stream] = fmaf(
                    __bfloat162float(injection_weight[
                        static_cast<std::int64_t>(destination_stream) * kHyper + k]),
                    value, injection_sum[destination_stream]);
            }
        }
        block_input[d + static_cast<std::int64_t>(kHidden) * token] =
            __float2bfloat16_rn(mixed * 0.25F);
    }

    __shared__ float partial[8][kStreams];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int stream = 0; stream < kStreams; ++stream) {
        for (int offset = 16; offset != 0; offset >>= 1) {
            injection_sum[stream] +=
                __shfl_down_sync(0xffffffffU, injection_sum[stream], offset);
        }
        if (lane == 0) { partial[warp][stream] = injection_sum[stream]; }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int stream = 0; stream < kStreams; ++stream) {
            float value = lane < 8 ? partial[lane][stream] : 0.0F;
            for (int offset = 16; offset != 0; offset >>= 1) {
                value += __shfl_down_sync(0xffffffffU, value, offset);
            }
            if (lane == 0) {
                injection[stream + static_cast<std::int64_t>(kStreams) * token] =
                    __float2bfloat16_rn(value);
            }
        }
    }
}

__global__ void injection_kernel(const __nv_bfloat16* normalized,
                                 const __nv_bfloat16* weight,
                                 __nv_bfloat16* injection, int tokens) {
    const int stream = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    float sum = 0.0F;
    for (int k = static_cast<int>(threadIdx.x); k < kHyper; k += static_cast<int>(blockDim.x)) {
        sum = fmaf(__bfloat162float(weight[static_cast<std::int64_t>(stream) * kHyper + k]),
                   __bfloat162float(normalized[k + static_cast<std::int64_t>(kHyper) * token]), sum);
    }
    for (int offset = 16; offset != 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    __shared__ float partial[8];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { partial[warp] = sum; }
    __syncthreads();
    if (warp == 0) {
        float value = lane < 8 ? partial[lane] : 0.0F;
        for (int offset = 16; offset != 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        if (lane == 0) { injection[stream + kStreams * token] = __float2bfloat16_rn(value); }
    }
}

__global__ void gate_mix_injection_decode_kernel(
    const __nv_bfloat16* normalized, const __nv_bfloat16* gate_logits,
    const __nv_bfloat16* injection_weight, __nv_bfloat16* block_input,
    __nv_bfloat16* injection) {
    constexpr int kMixBlocks = kHidden / 256;
    const int work = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    if (work < kMixBlocks) {
        const int d = work * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
        float sum = 0.0F;
#pragma unroll
        for (int stream = 0; stream < kStreams; ++stream) {
            const std::int64_t offset = d + static_cast<std::int64_t>(kHidden) *
                (stream + static_cast<std::int64_t>(kStreams) * token);
            const float gate =
                1.0F / (1.0F + expf(-__bfloat162float(gate_logits[offset])));
            sum = fmaf(gate, __bfloat162float(normalized[offset]), sum);
        }
        block_input[d + static_cast<std::int64_t>(kHidden) * token] =
            __float2bfloat16_rn(sum * 0.25F);
        return;
    }

    const int destination_stream = work - kMixBlocks;
    float sum = 0.0F;
    for (int k = static_cast<int>(threadIdx.x); k < kHyper;
         k += static_cast<int>(blockDim.x)) {
        sum = fmaf(
            __bfloat162float(injection_weight[
                static_cast<std::int64_t>(destination_stream) * kHyper + k]),
            __bfloat162float(normalized[k + static_cast<std::int64_t>(kHyper) * token]), sum);
    }
    for (int offset = 16; offset != 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    __shared__ float partial[8];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { partial[warp] = sum; }
    __syncthreads();
    if (warp == 0) {
        float value = lane < 8 ? partial[lane] : 0.0F;
        for (int offset = 16; offset != 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        if (lane == 0) {
            injection[destination_stream + static_cast<std::int64_t>(kStreams) * token] =
                __float2bfloat16_rn(value);
        }
    }
}

__global__ void combine_kernel(__nv_bfloat16* hyper, const __nv_bfloat16* block,
                               const __nv_bfloat16* injection, int tokens) {
    const std::int64_t count = static_cast<std::int64_t>(kHyper) * tokens;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const int d = static_cast<int>(i % kHidden);
        const std::int64_t row = i / kHidden;
        const int stream = static_cast<int>(row % kStreams);
        const int token = static_cast<int>(row / kStreams);
        const float logit = __bfloat162float(injection[stream + kStreams * token]) * 0.25F;
        const float scale = 2.0F / (1.0F + expf(-logit));
        hyper[i] = __float2bfloat16_rn(__bfloat162float(hyper[i]) +
                                       scale * __bfloat162float(block[d + kHidden * token]));
    }
}

// Small-token mix route: one cooperative launch with a single grid barrier.
// Phase 1 (per lane-owned CTA): commit the pending combine, steer, and fold the lane's RMSNorm
// weight into BF16 shared activations; eight-row tiles of Down and injection rows form
// lane-segment partial dots. The lane's inverse RMS factor is applied after the barrier, so
// normalized rows are never materialized. Each CTA also loads its Up rows before the barrier.
// Phase 2 (per hidden-column CTA): rebuild low = SiLU(Down n / 4) at the BF16 low-rank
// boundary, apply Up for the CTA's columns of all four lanes, and gate-mix the block input.
// Every reduction has a fixed order, so repeated launches are bitwise reproducible.
constexpr int kFusedMaxTokens      = 8;
constexpr int kFusedWarps          = kNormThreads / 32;
constexpr int kProjectionRows      = kRank + kStreams;

struct FusedMixParams {
    __nv_bfloat16* hyper;
    const __nv_bfloat16* previous_block;     // nullptr: no pending combine
    const __nv_bfloat16* previous_injection;
    const __nv_bfloat16* norm;
    const __nv_bfloat16* down;
    const __nv_bfloat16* up;
    const __nv_bfloat16* injection_weight;   // nullptr: no injection output
    __nv_bfloat16* block_input;
    __nv_bfloat16* injection;
    __nv_bfloat16* staged;                   // [kHyper,tokens] committed hyper
    float* partials;                         // [tokens,kProjectionRows,kStreams]
    float* square_sums;                      // [tokens,kStreams]
    const ActivationDevice* steering;
    int layer;
    int width;
    int tokens;
};

__device__ __forceinline__ uint4 load_streaming(const __nv_bfloat16* pointer) {
    uint4 bits;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
                 : "l"(pointer));
    return bits;
}

template <int Count>
__device__ __forceinline__ void warp_sum_all(float (&values)[Count]) {
#pragma unroll
    for (int shift = 16; shift; shift >>= 1) {
#pragma unroll
        for (int i = 0; i < Count; ++i)
            values[i] += __shfl_xor_sync(0xffffffffU, values[i], shift);
    }
}

// Both projections run on BF16 m16n8k16 MMA with the (at most eight) tokens as N. Within each
// 32-wide K chunk, lane (g, q) loads eight consecutive weights of row g (and g + 8) and the same
// eight activations of token g; dot products are permutation invariant, so the chunk is mapped
// onto the fragment K order without any shuffling: weights 0-3 feed the first MMA, 4-7 the second.
constexpr int kActivationStride = kHidden + 32; // BF16; 16-word skew between token rows
constexpr int kLowStride        = kRank + 32;
constexpr int kDownWarpK        = kHidden / kFusedWarps;  // 320 per warp
constexpr int kDownChunks       = kDownWarpK / 32;        // 10
constexpr int kUpTiles          = 4;                      // 64 Up rows per CTA
constexpr int kUpChunks         = kRank / 32 / 2;         // two K halves of 5 chunks
static_assert(kHidden % (kFusedWarps * 32) == 0 && kRank % 64 == 0);
static_assert(kUpTiles * 2 == kFusedWarps);

__device__ __forceinline__ void mma_bf16(float (&c)[4], std::uint32_t a0, std::uint32_t a1,
                                         std::uint32_t a2, std::uint32_t a3, std::uint32_t b0,
                                         std::uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One 32-wide K chunk: rows g and g + 8 against token g.
__device__ __forceinline__ void mma_chunk(float (&c)[4], const uint4& top, const uint4& bottom,
                                          const uint4& activation) {
    mma_bf16(c, top.x, bottom.x, top.y, bottom.y, activation.x, activation.y);
    mma_bf16(c, top.z, bottom.z, top.w, bottom.w, activation.z, activation.w);
}

__device__ __forceinline__ std::uint32_t pack_bf16(float low, float high) {
    const __nv_bfloat162 value = __floats2bfloat162_rn(low, high);
    return *reinterpret_cast<const std::uint32_t*>(&value);
}

template <int Tokens>
__global__ void __launch_bounds__(kNormThreads, 1) fused_mix_decode_kernel(FusedMixParams p) {
    extern __shared__ float4 fused_shared_storage[];
    auto* activations = reinterpret_cast<__nv_bfloat16*>(fused_shared_storage);
    __shared__ float reduce_shared[kFusedWarps];
    __shared__ float square_shared[kFusedWarps][Tokens];
    __shared__ float tile_shared[kFusedWarps][8][8];
    __shared__ float inverse_rms[Tokens * kStreams];
    const int tid   = static_cast<int>(threadIdx.x);
    const int lane  = tid & 31;
    const int warp  = tid >> 5;
    const int group = lane >> 2;
    const int quad  = lane & 3;

    const int slices = static_cast<int>(gridDim.x) / kStreams;
    const int stream = static_cast<int>(blockIdx.x) % kStreams;
    const int slice  = static_cast<int>(blockIdx.x) / kStreams;
    const int rows   = p.injection_weight != nullptr ? kProjectionRows : kRank;
    // Lane g of this warp holds row (tile * 8 + g) over the warp's K slice of this lane segment.
    const auto load_down = [&](int tile, uint4 (&bits)[kDownChunks]) {
        const int row = tile * 8 + group;
        if (row >= rows) {
#pragma unroll
            for (int c = 0; c < kDownChunks; ++c) bits[c] = make_uint4(0, 0, 0, 0);
            return;
        }
        const __nv_bfloat16* weight =
            (row < kRank ? p.down + static_cast<std::int64_t>(row) * kHyper
                         : p.injection_weight + static_cast<std::int64_t>(row - kRank) * kHyper) +
            kHidden * stream + warp * kDownWarpK + 8 * quad;
#pragma unroll
        for (int c = 0; c < kDownChunks; ++c) bits[c] = load_streaming(weight + 32 * c);
    };
    uint4 bits[kDownChunks];
    load_down(slice, bits);  // independent of the activations: issue before the setup

    const int columns_per_cta = (kHidden + static_cast<int>(gridDim.x) - 1) / gridDim.x;
    const int column_begin    = static_cast<int>(blockIdx.x) * columns_per_cta;
    const int columns         = max(0, min(columns_per_cta, kHidden - column_begin));
    const int up_rows         = kStreams * columns;

    // Phase 1: commit every token's lane, steer, and fold the lane's norm weight. All activation
    // loads are issued before any use so their latency overlaps the weight stream once.
    {
        float x[Tokens][kPerThread];
        const std::int64_t lane_base = static_cast<std::int64_t>(kHidden) * stream + tid;
        if (p.previous_block != nullptr) {
            float branch_logit[Tokens];
            __nv_bfloat16 h[Tokens][kPerThread], b[Tokens][kPerThread];
#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                branch_logit[token] =
                    __bfloat162float(p.previous_injection[stream + kStreams * token]);
#pragma unroll
                for (int j = 0; j < kPerThread; ++j) {
                    h[token][j] = p.hyper[lane_base + static_cast<std::int64_t>(kHyper) * token +
                                          j * kNormThreads];
                    b[token][j] = p.previous_block[tid + j * kNormThreads + kHidden * token];
                }
            }
#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                const float scale = 2.0F / (1.0F + expf(-branch_logit[token] * 0.25F));
#pragma unroll
                for (int j = 0; j < kPerThread; ++j)
                    x[token][j] = __bfloat162float(__float2bfloat16_rn(
                        __bfloat162float(h[token][j]) + scale * __bfloat162float(b[token][j])));
            }
        } else {
#pragma unroll
            for (int token = 0; token < Tokens; ++token)
#pragma unroll
                for (int j = 0; j < kPerThread; ++j)
                    x[token][j] = __bfloat162float(
                        p.hyper[lane_base + static_cast<std::int64_t>(kHyper) * token +
                                j * kNormThreads]);
        }
        if (p.steering != nullptr) {
#pragma unroll
            for (int token = 0; token < Tokens; ++token)
                steer_lane(x[token], p.steering, p.layer, p.width, stream, token, reduce_shared);
        }
        float norm_scale[kPerThread];
#pragma unroll
        for (int j = 0; j < kPerThread; ++j)
            norm_scale[j] =
                1.0F + __bfloat162float(p.norm[kHidden * stream + tid + j * kNormThreads]);
        float square[Tokens] = {};
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
#pragma unroll
            for (int j = 0; j < kPerThread; ++j) {
                activations[token * kActivationStride + tid + j * kNormThreads] =
                    __float2bfloat16_rn(x[token][j] * norm_scale[j]);
                square[token] = fmaf(x[token][j], x[token][j], square[token]);
            }
        }
        if (slice == 0) {
#pragma unroll
            for (int token = 0; token < Tokens; ++token)
#pragma unroll
                for (int j = 0; j < kPerThread; ++j)
                    p.staged[static_cast<std::int64_t>(kHyper) * token + lane_base +
                             j * kNormThreads] = __float2bfloat16_rn(x[token][j]);
            warp_sum_all(square);
            if (lane == 0) {
#pragma unroll
                for (int token = 0; token < Tokens; ++token) square_shared[warp][token] = square[token];
            }
        }
    }
    __syncthreads();
    if (slice == 0 && tid < Tokens) {
        float sum = 0.0F;
#pragma unroll
        for (int w = 0; w < kFusedWarps; ++w) sum += square_shared[w][tid];
        p.square_sums[stream + kStreams * tid] = sum;
    }

    // Lane-segment partial dots: eight rows per tile, K split across the eight warps and reduced
    // in a fixed order.
    for (int tile = slice; tile * 8 < rows; tile += slices) {
        if (tile != slice) load_down(tile, bits);
        float c[4] = {};
#pragma unroll
        for (int chunk = 0; chunk < kDownChunks; ++chunk) {
            uint4 activation = make_uint4(0, 0, 0, 0);
            if (group < Tokens) {
                activation = *reinterpret_cast<const uint4*>(
                    activations + group * kActivationStride + warp * kDownWarpK + 32 * chunk +
                    8 * quad);
            }
            mma_chunk(c, bits[chunk], make_uint4(0, 0, 0, 0), activation);
        }
        tile_shared[warp][group][2 * quad]     = c[0];
        tile_shared[warp][group][2 * quad + 1] = c[1];
        __syncthreads();
        if (tid < 64) {
            const int row   = tile * 8 + tid / 8;
            const int token = tid % 8;
            float sum       = 0.0F;
#pragma unroll
            for (int w = 0; w < kFusedWarps; ++w) sum += tile_shared[w][tid / 8][token];
            if (row < rows && token < Tokens)
                p.partials[(token * kProjectionRows + row) * kStreams + stream] = sum;
        }
        __syncthreads();
    }

    // Up rows do not depend on low: load them before the barrier. Warp pair (tile, K half) owns
    // rows tile * 16 + g and tile * 16 + g + 8 over five K chunks.
    const int up_tile = warp >> 1;
    const int up_half = warp & 1;
    uint4 up_top[kUpChunks], up_bottom[kUpChunks];
    const auto load_up = [&](int up_row, uint4 (&out)[kUpChunks]) {
        if (up_row >= up_rows) {
#pragma unroll
            for (int c = 0; c < kUpChunks; ++c) out[c] = make_uint4(0, 0, 0, 0);
            return;
        }
        const __nv_bfloat16* weight =
            p.up +
            static_cast<std::int64_t>((up_row / columns) * kHidden + column_begin +
                                      up_row % columns) *
                kRank +
            up_half * (kRank / 2) + 8 * quad;
#pragma unroll
        for (int c = 0; c < kUpChunks; ++c) out[c] = load_streaming(weight + 32 * c);
    };
    load_up(up_tile * 16 + group, up_top);
    load_up(up_tile * 16 + group + 8, up_bottom);

    cooperative_groups::this_grid().sync();

    // Phase 2: low-rank activation, injection, Up for this CTA's columns, and the gate mix.
    auto* low      = activations;                                             // [8,kLowStride]
    float* up_sums = reinterpret_cast<float*>(activations + 8 * kLowStride);  // [2,64,8]
    const int mix_token  = tid / max(columns, 1);
    const int mix_column = tid % max(columns, 1);
    const bool mixes     = tid < Tokens * columns;
    __nv_bfloat16 committed[kStreams];
    float mix_norm[kStreams];
    if (mixes) {
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
            const int d  = column_begin + mix_column;
            committed[s] = p.staged[static_cast<std::int64_t>(kHyper) * mix_token + kHidden * s + d];
            mix_norm[s]  = 1.0F + __bfloat162float(p.norm[kHidden * s + d]);
        }
    }
    if (tid < Tokens * kStreams)
        inverse_rms[tid] = rsqrtf(p.square_sums[tid] / static_cast<float>(kHidden) + 1.0e-6F);
    __syncthreads();
    constexpr int kLowPairs        = Tokens * kRank / 2;
    constexpr int kLowPerThread    = (kLowPairs + kNormThreads - 1) / kNormThreads;
#pragma unroll
    for (int k = 0; k < kLowPerThread; ++k) {
        const int i = tid + k * kNormThreads;
        if (i < kLowPairs) {
            const int token    = i / (kRank / 2);
            const int rank     = 2 * (i % (kRank / 2));
            const float* scale = inverse_rms + token * kStreams;
            const float* row   = p.partials + (token * kProjectionRows + rank) * kStreams;
            const float4 first  = *reinterpret_cast<const float4*>(row);
            const float4 second = *reinterpret_cast<const float4*>(row + kStreams);
            const float v0 = 0.25F * (first.x * scale[0] + first.y * scale[1] +
                                      first.z * scale[2] + first.w * scale[3]);
            const float v1 = 0.25F * (second.x * scale[0] + second.y * scale[1] +
                                      second.z * scale[2] + second.w * scale[3]);
            *reinterpret_cast<std::uint32_t*>(low + token * kLowStride + rank) =
                pack_bf16(v0 / (1.0F + expf(-v0)), v1 / (1.0F + expf(-v1)));
        }
    }
    if (blockIdx.x == 0 && p.injection != nullptr && tid < Tokens * kStreams) {
        const int token      = tid / kStreams;
        const float4 partial = *reinterpret_cast<const float4*>(
            p.partials + (token * kProjectionRows + kRank + tid % kStreams) * kStreams);
        const float* scale = inverse_rms + token * kStreams;
        p.injection[tid]   = __float2bfloat16_rn(partial.x * scale[0] + partial.y * scale[1] +
                                                 partial.z * scale[2] + partial.w * scale[3]);
    }
    __syncthreads();
    if (columns == 0) return;

    {
        float c[4] = {};
#pragma unroll
        for (int chunk = 0; chunk < kUpChunks; ++chunk) {
            uint4 activation = make_uint4(0, 0, 0, 0);
            if (group < Tokens) {
                activation = *reinterpret_cast<const uint4*>(
                    low + group * kLowStride + up_half * (kRank / 2) + 32 * chunk + 8 * quad);
            }
            mma_chunk(c, up_top[chunk], up_bottom[chunk], activation);
        }
        float* sums = up_sums + up_half * 64 * 8;
        const int row = up_tile * 16 + group;
        sums[row * 8 + 2 * quad]           = c[0];
        sums[row * 8 + 2 * quad + 1]       = c[1];
        sums[(row + 8) * 8 + 2 * quad]     = c[2];
        sums[(row + 8) * 8 + 2 * quad + 1] = c[3];
    }
    __syncthreads();

    if (mixes) {
        const int d = column_begin + mix_column;
        float mixed = 0.0F;
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
            p.hyper[static_cast<std::int64_t>(kHyper) * mix_token + kHidden * s + d] = committed[s];
            const float normalized = __bfloat162float(committed[s]) *
                                     inverse_rms[mix_token * kStreams + s] * mix_norm[s];
            const int row     = (s * columns + mix_column) * 8 + mix_token;
            const float logit = up_sums[row] + up_sums[64 * 8 + row];
            mixed             = fmaf(normalized, 1.0F / (1.0F + expf(-logit)), mixed);
        }
        p.block_input[d + static_cast<std::int64_t>(kHidden) * mix_token] =
            __float2bfloat16_rn(mixed * 0.25F);
    }
}

std::size_t fused_mix_shared_bytes(int tokens) {
    const std::size_t phase1 = static_cast<std::size_t>(tokens) * kActivationStride * 2;
    const std::size_t phase2 = 8 * kLowStride * 2 + 2 * 64 * 8 * sizeof(float);
    return phase1 > phase2 ? phase1 : phase2;
}

using FusedMixKernel = void (*)(FusedMixParams);

template <int... Tokens>
constexpr auto fused_mix_kernels(std::integer_sequence<int, Tokens...>) {
    return std::array<FusedMixKernel, sizeof...(Tokens)>{fused_mix_decode_kernel<Tokens + 1>...};
}

constexpr auto kFusedMixKernels =
    fused_mix_kernels(std::make_integer_sequence<int, kFusedMaxTokens>{});

// The cooperative grid is one resident CTA per SM, rounded down to whole lane groups. Phase 2
// holds a CTA's Up rows in registers and its mix outputs in one thread each, which bounds the
// columns per CTA; a device that cannot meet that or host the grid selects the general route.
int fused_mix_grid() {
    static const int grid = [] {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        int multiprocessors = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
        const int grid            = multiprocessors / kStreams * kStreams;
        const int columns_per_cta = grid > 0 ? (kHidden + grid - 1) / grid : kHidden;
        if (kStreams * columns_per_cta > kUpTiles * 16 ||
            kFusedMaxTokens * columns_per_cta > kNormThreads) {
            return 0;
        }
        for (int tokens = 1; tokens <= kFusedMaxTokens; ++tokens) {
            const std::size_t bytes     = fused_mix_shared_bytes(tokens);
            const FusedMixKernel kernel = kFusedMixKernels[tokens - 1];
            CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(bytes)));
            int resident = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel,
                                                                     kNormThreads, bytes));
            if (resident == 0) return 0;
        }
        return grid;
    }();
    return grid;
}

bool launch_fused_mix(FusedMixParams params, cudaStream_t stream) {
    const int grid = fused_mix_grid();
    if (grid == 0 || params.tokens > kFusedMaxTokens) return false;
    cudaLaunchConfig_t config{};
    config.gridDim          = dim3(static_cast<unsigned>(grid));
    config.blockDim         = dim3(kNormThreads);
    config.dynamicSmemBytes =
        fused_mix_shared_bytes(params.tokens);
    config.stream = stream;
    cudaLaunchAttribute cooperative{};
    cooperative.id              = cudaLaunchAttributeCooperative;
    cooperative.val.cooperative = 1;
    config.attrs                = &cooperative;
    config.numAttrs             = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&config, kFusedMixKernels[params.tokens - 1], params));
    return true;
}

int grid_for(std::int64_t count) {
    constexpr int block = 256;
    const std::int64_t blocks = (count + block - 1) / block;
    return static_cast<int>(blocks < 4096 ? blocks : 4096);
}

void validate(const Tensor& hyper, const HyperConnectionWeights& weights, const Tensor& block,
              const Tensor* injection) {
    const int tokens = hyper.ne[1];
    if (hyper.dtype != DType::BF16 || !hyper.is_contiguous() || hyper.ne[0] != kHyper ||
        hyper.ne[2] != 1 || hyper.ne[3] != 1 || tokens <= 0 || hyper.data == nullptr ||
        block.dtype != DType::BF16 || !block.is_contiguous() || block.ne[0] != kHidden ||
        block.ne[1] != tokens || block.ne[2] != 1 || block.ne[3] != 1 || block.data == nullptr ||
        weights.norm.dtype != DType::BF16 || weights.norm.ne[0] != kHyper ||
        !weights.norm.is_contiguous() || weights.down.n != kRank || weights.down.k != kHyper ||
        weights.up.n != kHyper || weights.up.k != kRank) {
        throw std::invalid_argument("hyperconnection_mix: invalid Flash-Next geometry");
    }
    if (injection != nullptr &&
        (injection->dtype != DType::BF16 || !injection->is_contiguous() ||
         injection->ne[0] != kStreams || injection->ne[1] != tokens ||
         weights.injection.n != kStreams || weights.injection.k != kHyper)) {
        throw std::invalid_argument("hyperconnection_mix: invalid injection geometry");
    }
}

void validate_combine_inputs(const Tensor& hyper, const Tensor& block_output,
                             const Tensor& injection) {
    if (hyper.dtype != DType::BF16 || !hyper.is_contiguous() || hyper.ne[0] != kHyper ||
        block_output.dtype != DType::BF16 || !block_output.is_contiguous() ||
        block_output.ne[0] != kHidden || block_output.ne[1] != hyper.ne[1] ||
        injection.dtype != DType::BF16 || !injection.is_contiguous() ||
        injection.ne[0] != kStreams || injection.ne[1] != hyper.ne[1]) {
        throw std::invalid_argument("hyperconnection_combine: invalid Flash-Next geometry");
    }
}

void validate_steering(const HyperConnectionActivation* activation, int tokens) {
    if (activation == nullptr || activation->steering == nullptr) { return; }
    if (activation->layer < 0 || activation->layer >= kActivationLayers || activation->width <= 0 ||
        tokens % activation->width != 0 || tokens / activation->width > kActivationRows) {
        throw std::invalid_argument("hyperconnection steering: layer/width check failed");
    }
}

void finish_mix(const Tensor& hyper, const Tensor& normalized,
                const HyperConnectionWeights& weights, Tensor& block_input, Tensor* injection,
                WorkspaceArena& workspace, cudaStream_t stream, Bf16GemmContext* bf16_gemm,
                const HyperConnectionActivation* activation) {
    const int tokens = normalized.ne[1];
    Tensor low_rank = workspace.alloc(DType::BF16, {kRank, tokens});
    const bool fused_down_silu = tokens >= 2 && tokens <= 16;
    if (fused_down_silu) {
        detail::flash_next::launch_bf16_hc_down_silu_small_t(normalized, weights.down, low_rank, stream);
    } else {
        linear(normalized, weights.down, low_rank, stream, bf16_gemm);
    }
    constexpr int block = 256;
    if (!fused_down_silu) {
        scaled_silu_kernel<<<grid_for(low_rank.numel()), block, 0, stream>>>(
            static_cast<__nv_bfloat16*>(low_rank.data), low_rank.numel());
    }
    Tensor gate = workspace.alloc(DType::BF16, {kHyper, tokens});
    linear(low_rank, weights.up, gate, stream, bf16_gemm);
    if (injection != nullptr) {
        if (tokens > 16) {
            gate_mix_injection_kernel<<<tokens, block, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(normalized.data),
                static_cast<const __nv_bfloat16*>(gate.data),
                static_cast<const __nv_bfloat16*>(weights.injection.qdata),
                static_cast<__nv_bfloat16*>(block_input.data),
                static_cast<__nv_bfloat16*>(injection->data));
        } else {
            constexpr int kDecodeMixBlocks = kHidden / block;
            gate_mix_injection_decode_kernel<<<
                dim3(kDecodeMixBlocks + kStreams, static_cast<unsigned int>(tokens)),
                block, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(normalized.data),
                static_cast<const __nv_bfloat16*>(gate.data),
                static_cast<const __nv_bfloat16*>(weights.injection.qdata),
                static_cast<__nv_bfloat16*>(block_input.data),
                static_cast<__nv_bfloat16*>(injection->data));
        }
    } else {
        gate_mix_kernel<<<grid_for(block_input.numel()), block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(normalized.data),
            static_cast<const __nv_bfloat16*>(gate.data),
            static_cast<__nv_bfloat16*>(block_input.data), tokens);
    }
    CUDA_CHECK(cudaGetLastError());
    if (activation != nullptr && activation->capture != nullptr) {
        activation_capture(hyper, normalized, gate, block_input, *activation->positions,
                           *activation->ids, *activation->valid, activation->capture,
                           activation->layer, activation->width, stream,
                           activation->speculative_columns);
    }
}

// Selects the fused small-token route; capture needs the materialized normalized and gate rows.
bool fused_mix(Tensor& hyper, const Tensor* previous_block_output,
               const Tensor* previous_injection, const HyperConnectionWeights& weights,
               Tensor& block_input, Tensor* injection, WorkspaceArena& workspace,
               cudaStream_t stream, const HyperConnectionActivation* activation) {
    const int tokens = hyper.ne[1];
    if (tokens > kFusedMaxTokens || (activation != nullptr && activation->capture != nullptr)) {
        return false;
    }
    Tensor staged      = workspace.alloc(DType::BF16, {kHyper, tokens});
    Tensor partials    = workspace.alloc(DType::FP32, {kProjectionRows * kStreams, tokens});
    Tensor square_sums = workspace.alloc(DType::FP32, {kStreams, tokens});
    FusedMixParams params{
        .hyper              = static_cast<__nv_bfloat16*>(hyper.data),
        .previous_block     = previous_block_output != nullptr
                                  ? static_cast<const __nv_bfloat16*>(previous_block_output->data)
                                  : nullptr,
        .previous_injection = previous_injection != nullptr
                                  ? static_cast<const __nv_bfloat16*>(previous_injection->data)
                                  : nullptr,
        .norm               = static_cast<const __nv_bfloat16*>(weights.norm.data),
        .down               = static_cast<const __nv_bfloat16*>(weights.down.qdata),
        .up                 = static_cast<const __nv_bfloat16*>(weights.up.qdata),
        .injection_weight   = injection != nullptr
                                  ? static_cast<const __nv_bfloat16*>(weights.injection.qdata)
                                  : nullptr,
        .block_input        = static_cast<__nv_bfloat16*>(block_input.data),
        .injection   = injection != nullptr ? static_cast<__nv_bfloat16*>(injection->data) : nullptr,
        .staged      = static_cast<__nv_bfloat16*>(staged.data),
        .partials    = static_cast<float*>(partials.data),
        .square_sums = static_cast<float*>(square_sums.data),
        .steering    = activation != nullptr ? activation->steering : nullptr,
        .layer       = activation != nullptr ? activation->layer : 0,
        .width       = activation != nullptr ? activation->width : 1,
        .tokens      = tokens,
    };
    return launch_fused_mix(params, stream);
}

} // namespace

void hyperconnection_repeat(const Tensor& input, Tensor& hyper, cudaStream_t stream) {
    NINFER_PERF_SCOPE("hyper.repeat", input.ne[1], 0, 0,
                       flash_next_work::dense(0, input.ne[1], 2 * (2560 + 10240) * input.ne[1]));

    if (input.dtype != DType::BF16 || !input.is_contiguous() || input.ne[0] != kHidden ||
        input.ne[1] <= 0 || input.ne[2] != 1 || input.ne[3] != 1 || input.data == nullptr ||
        hyper.dtype != DType::BF16 || !hyper.is_contiguous() || hyper.ne[0] != kHyper ||
        hyper.ne[1] != input.ne[1] || hyper.ne[2] != 1 || hyper.ne[3] != 1 ||
        hyper.data == nullptr) {
        throw std::invalid_argument("hyperconnection_repeat: invalid Flash-Next geometry");
    }
    repeat_kernel<<<grid_for(hyper.numel()), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data),
        static_cast<__nv_bfloat16*>(hyper.data), hyper.numel());
    CUDA_CHECK(cudaGetLastError());
}

void hyperconnection_add_repeated(const Tensor& embedding, Tensor& hyper,
                                  cudaStream_t stream) {
    NINFER_PERF_SCOPE("hyper.add", embedding.ne[1], 0, 0,
                       flash_next_work::dense(0, embedding.ne[1], 2 * (2560 + 2 * 10240) * embedding.ne[1]));

    if (embedding.dtype != DType::BF16 || !embedding.is_contiguous() ||
        embedding.ne[0] != kHidden || embedding.ne[1] <= 0 || embedding.ne[2] != 1 ||
        embedding.ne[3] != 1 || embedding.data == nullptr || hyper.dtype != DType::BF16 ||
        !hyper.is_contiguous() || hyper.ne[0] != kHyper ||
        hyper.ne[1] != embedding.ne[1] || hyper.ne[2] != 1 || hyper.ne[3] != 1 ||
        hyper.data == nullptr) {
        throw std::invalid_argument("hyperconnection_add_repeated: invalid Flash-Next geometry");
    }
    add_repeated_kernel<<<grid_for(hyper.numel()), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(embedding.data),
        static_cast<__nv_bfloat16*>(hyper.data), hyper.numel());
    CUDA_CHECK(cudaGetLastError());
}

std::size_t hyperconnection_mix_workspace_capacity_bytes(std::int32_t tokens,
                                                          bool with_injection) {
    if (tokens <= 0) { throw std::invalid_argument("HyperConnection token count must be positive"); }
    const std::uint64_t elements = static_cast<std::uint64_t>(tokens) * (kHyper + kRank + kHyper);
    const std::uint64_t bytes = elements * sizeof(__nv_bfloat16);
    if (bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("HyperConnection workspace size overflow");
    }
    (void)with_injection;
    return static_cast<std::size_t>(bytes) + 3 * 256;
}

void hyperconnection_mix(const Tensor& hyper, const HyperConnectionWeights& weights,
                         Tensor& block_input, Tensor* injection, WorkspaceArena& workspace,
                         cudaStream_t stream, Bf16GemmContext* bf16_gemm,
                         const HyperConnectionActivation* activation) {
    NINFER_PERF_SCOPE("hyper.mix", hyper.ne[1], 0, 0,
                       flash_next_work::hyper(hyper.ne[1], injection != nullptr, false));

    validate(hyper, weights, block_input, injection);
    const int tokens = hyper.ne[1];
    auto scope = workspace.scope();
    validate_steering(activation, tokens);
    Tensor mutable_hyper = hyper;
    if (fused_mix(mutable_hyper, nullptr, nullptr, weights, block_input, injection, workspace,
                  stream, activation)) {
        return;
    }
    Tensor normalized = workspace.alloc(DType::BF16, {kHyper, tokens});
    const ActivationDevice* steering = activation != nullptr ? activation->steering : nullptr;
    grouped_rmsnorm_kernel<<<dim3(kStreams, static_cast<unsigned int>(tokens)), kNormThreads, 0,
                             stream>>>(
        static_cast<__nv_bfloat16*>(hyper.data),
        static_cast<const __nv_bfloat16*>(weights.norm.data),
        static_cast<__nv_bfloat16*>(normalized.data), steering,
        activation != nullptr ? activation->layer : 0, activation != nullptr ? activation->width : 1);
    finish_mix(hyper, normalized, weights, block_input, injection, workspace, stream, bf16_gemm,
               activation);
}

void hyperconnection_combine_mix(Tensor& hyper, const Tensor& previous_block_output,
                                 const Tensor& previous_injection,
                                 const HyperConnectionWeights& weights,
                                 Tensor& block_input, Tensor* injection,
                                 WorkspaceArena& workspace, cudaStream_t stream,
                                 Bf16GemmContext* bf16_gemm,
                                 const HyperConnectionActivation* activation) {
    NINFER_PERF_SCOPE("hyper.combine_mix", hyper.ne[1], 0, 0,
                       flash_next_work::hyper(hyper.ne[1], injection != nullptr, true));

    validate(hyper, weights, block_input, injection);
    validate_combine_inputs(hyper, previous_block_output, previous_injection);
    const int tokens = hyper.ne[1];
    auto scope = workspace.scope();
    validate_steering(activation, tokens);
    if (fused_mix(hyper, &previous_block_output, &previous_injection, weights, block_input,
                  injection, workspace, stream, activation)) {
        return;
    }
    Tensor normalized = workspace.alloc(DType::BF16, {kHyper, tokens});
    const ActivationDevice* steering = activation != nullptr ? activation->steering : nullptr;
    combine_grouped_rmsnorm_kernel<<<
        dim3(kStreams, static_cast<unsigned int>(tokens)), kNormThreads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(hyper.data),
        static_cast<const __nv_bfloat16*>(previous_block_output.data),
        static_cast<const __nv_bfloat16*>(previous_injection.data),
        static_cast<const __nv_bfloat16*>(weights.norm.data),
        static_cast<__nv_bfloat16*>(normalized.data), steering,
        activation != nullptr ? activation->layer : 0, activation != nullptr ? activation->width : 1);
    finish_mix(hyper, normalized, weights, block_input, injection, workspace, stream, bf16_gemm,
               activation);
}

void hyperconnection_combine(Tensor& hyper, const Tensor& block_output, const Tensor& injection,
                             cudaStream_t stream) {
    NINFER_PERF_SCOPE("hyper.combine", hyper.ne[1], 0, 0,
                       flash_next_work::dense(0, hyper.ne[1], 2 * (2 * 10240 + 2560 + 4) * hyper.ne[1]));

    validate_combine_inputs(hyper, block_output, injection);
    constexpr int block = 256;
    combine_kernel<<<grid_for(hyper.numel()), block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(hyper.data),
        static_cast<const __nv_bfloat16*>(block_output.data),
        static_cast<const __nv_bfloat16*>(injection.data), hyper.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
