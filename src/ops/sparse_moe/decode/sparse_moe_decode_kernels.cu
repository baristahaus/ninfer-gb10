#include "core/weight.h"
#include "ops/sparse_moe/decode/sparse_moe_decode.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/exl3_codec.cuh"
#include "ops/common/hadamard.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"
#include "ops/linear/q8/q8_rowsplit_storage.cuh"
#include "ops/sparse_moe/sparse_moe_route.cuh"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden           = 2048;
constexpr int kExperts          = 256;
constexpr int kRouterRows       = kExperts + 1;
constexpr int kTopK             = 8;
constexpr int kIntermediate     = 512;
constexpr int kD1Warps          = 8;
constexpr int kAdaptiveD3Blocks = 5 * kIntermediate;
constexpr int kAdaptiveD4Blocks = 5 * (kHidden / 4);

__device__ __forceinline__ float dot_bf16_eight(const __nv_bfloat16* a, const __nv_bfloat16* b) {
    const uint4 av  = load_vec<uint4>(a);
    const uint4 bv  = load_vec<uint4>(b);
    const float2 a0 = bf16x2_bits_to_float2(av.x);
    const float2 a1 = bf16x2_bits_to_float2(av.y);
    const float2 a2 = bf16x2_bits_to_float2(av.z);
    const float2 a3 = bf16x2_bits_to_float2(av.w);
    const float2 b0 = bf16x2_bits_to_float2(bv.x);
    const float2 b1 = bf16x2_bits_to_float2(bv.y);
    const float2 b2 = bf16x2_bits_to_float2(bv.z);
    const float2 b3 = bf16x2_bits_to_float2(bv.w);
    float sum       = 0.0f;
    sum             = fmaf(a0.x, b0.x, sum);
    sum             = fmaf(a0.y, b0.y, sum);
    sum             = fmaf(a1.x, b1.x, sum);
    sum             = fmaf(a1.y, b1.y, sum);
    sum             = fmaf(a2.x, b2.x, sum);
    sum             = fmaf(a2.y, b2.y, sum);
    sum             = fmaf(a3.x, b3.x, sum);
    sum             = fmaf(a3.y, b3.y, sum);
    return sum;
}

__device__ __forceinline__ float router_row_dot(const __nv_bfloat16* x, const __nv_bfloat16* row) {
    constexpr int kSlice = kHidden / kD1Warps;
    constexpr int kVecs  = kSlice / (32 * 8);
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    float sum            = 0.0f;
#pragma unroll
    for (int vector = 0; vector < kVecs; ++vector) {
        const int k = warp * kSlice + vector * 32 * 8 + lane * 8;
        sum += dot_bf16_eight(row + k, x + k);
    }
    return warp_reduce_sum(sum);
}

__global__ void sparse_moe_d1_kernel(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ router,
                                     float* __restrict__ scores,
                                     const char* __restrict__ shared_down_codes,
                                     unsigned long long shared_down_bytes) {
    __shared__ float partial[kD1Warps];
    const int row   = static_cast<int>(blockIdx.x);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const float dot = router_row_dot(x, router + static_cast<std::int64_t>(row) * kHidden);
    if (lane == 0) { partial[warp] = dot; }
    __syncthreads();
    if (warp == 0) {
        float value = lane < kD1Warps ? partial[lane] : 0.0f;
        value       = warp_reduce_sum<kD1Warps>(value);
        if (lane == 0) { scores[row] = value; }
    }
    if (shared_down_codes != nullptr) {
        // Warm L2 for the shared-expert down codes that D4's shared warp
        // (the block's long pole) will stream at t~14.3us. Pure cache hint.
        const unsigned long long offset =
            (static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x) * 128ull;
        if (offset < shared_down_bytes) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(shared_down_codes + offset));
        }
    }
}

__global__ void sparse_moe_d2_warp_kernel(const float* __restrict__ scores, int* __restrict__ ids,
                                          float* __restrict__ alpha,
                                          float* __restrict__ shared_scale,
                                          std::uint32_t* __restrict__ expert_activations) {
    __shared__ float selected_logits[kTopK];
    if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    sparse_moe_select_top8_warp(scores, ids, alpha, shared_scale, selected_logits,
                                expert_activations);
}

struct Q4Codec {
    static constexpr int kGroupK         = 64;
    static constexpr bool kD3PackedWord8 = true;
};

struct Q5Codec {
    static constexpr int kGroupK       = 64;
    static constexpr bool kPackedWord8 = true;

    __device__ static __forceinline__ void
    load_eight(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
               std::int64_t group_index, int lane_in_group, float (&weights)[8]) {
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            codes + group_index * Q5RowSplitStorage::kCodeBytesPerGroup + lane_in_group * 4);
        const std::uint8_t high_bits =
            high[group_index * Q5RowSplitStorage::kHighBytesPerGroup + lane_in_group];
        const auto scale_bits = *reinterpret_cast<const std::uint16_t*>(
            scales + group_index * Q5RowSplitStorage::kScaleBytesPerGroup);
        Q5SimtDecodeAtom::decode_eight(packed, high_bits, scale_bits, weights);
    }
};

struct Q6Codec {
    static constexpr int kGroupK       = 64;
    static constexpr bool kPackedWord8 = true;

    __device__ static __forceinline__ void
    load_eight(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
               std::int64_t group_index, int lane_in_group, float (&weights)[8]) {
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            codes + group_index * Q6RowSplitStorage::kCodeBytesPerGroup + lane_in_group * 4);
        const std::uint16_t high_bits = *reinterpret_cast<const std::uint16_t*>(
            high + group_index * Q6RowSplitStorage::kHighBytesPerGroup + lane_in_group * 2);
        const auto scale_bits = *reinterpret_cast<const std::uint16_t*>(
            scales + group_index * Q6RowSplitStorage::kScaleBytesPerGroup);
        Q6SimtDecodeAtom::decode_eight(packed, high_bits, scale_bits, weights);
    }
};

struct Q8Codec {
    static constexpr int kGroupK                = 32;
    static constexpr bool kD3SingleValuePerLane = true;
    static constexpr bool kD3PackedWord8        = false;
    static constexpr bool kPackedWord8          = false;

    __device__ static __forceinline__ float load_one(const std::uint8_t* codes,
                                                     const std::uint8_t* scales,
                                                     std::int64_t group_index, int lane) {
        const float scale = __half2float(
            __ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scales + group_index * 2)));
        return static_cast<float>(static_cast<std::int8_t>(codes[group_index * kGroupK + lane])) *
               scale;
    }

    __device__ static __forceinline__ void
    load_pair(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
              std::int64_t group_index, int lane, float& w0, float& w1) {
        Q8ScalarDecodeAtom::load_pair(codes, high, scales, group_index, lane, w0, w1);
    }
};

template <class Codec, int K>
__device__ __forceinline__ void dot_two_rows(const std::uint8_t* codes, const std::uint8_t* high,
                                             const std::uint8_t* scales, int row0, int row1,
                                             const __nv_bfloat16* x, int k_begin, int k_end,
                                             float& result0, float& result1) {
    constexpr int kGroups = K / Codec::kGroupK;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    float acc0            = 0.0f;
    float acc1            = 0.0f;
    const int first_group = k_begin / Codec::kGroupK;
    const int last_group  = k_end / Codec::kGroupK;
    if constexpr (Codec::kD3PackedWord8) {
        // Four adjacent Q4 groups form one 128-byte warp transaction. Each lane owns eight
        // consecutive K values, so one mantissa decode feeds eight FP32 FMAs instead of issuing
        // four scalar code-pair/decode iterations.
        const int lane_group    = lane >> 3;
        const int lane_in_group = lane & 7;
        for (int group_base = first_group; group_base < last_group; group_base += 4) {
            const int group           = group_base + lane_group;
            const std::int64_t index0 = static_cast<std::int64_t>(row0) * kGroups + group;
            const std::int64_t index1 = static_cast<std::int64_t>(row1) * kGroups + group;
            const std::uint32_t packed0 =
                *reinterpret_cast<const std::uint32_t*>(codes + index0 * 32 + lane_in_group * 4);
            const std::uint32_t packed1 =
                *reinterpret_cast<const std::uint32_t*>(codes + index1 * 32 + lane_in_group * 4);
            const auto scale0 = *reinterpret_cast<const std::uint16_t*>(scales + index0 * 2);
            const auto scale1 = *reinterpret_cast<const std::uint16_t*>(scales + index1 * 2);
            float weights0[8];
            float weights1[8];
            Q4SimtDecodeAtom::decode_eight(packed0, scale0, weights0);
            Q4SimtDecodeAtom::decode_eight(packed1, scale1, weights1);
            const uint4 input     = load_vec<uint4>(x + group * Codec::kGroupK + lane_in_group * 8);
            const float2 x0       = bf16x2_bits_to_float2(input.x);
            const float2 x1       = bf16x2_bits_to_float2(input.y);
            const float2 x2       = bf16x2_bits_to_float2(input.z);
            const float2 x3       = bf16x2_bits_to_float2(input.w);
            const float values[8] = {x0.x, x0.y, x1.x, x1.y, x2.x, x2.y, x3.x, x3.y};
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                acc0 = fmaf(weights0[item], values[item], acc0);
                acc1 = fmaf(weights1[item], values[item], acc1);
            }
        }
    } else if constexpr (Codec::kD3SingleValuePerLane) {
        for (int group = first_group; group < last_group; ++group) {
            const std::int64_t index0 = static_cast<std::int64_t>(row0) * kGroups + group;
            const std::int64_t index1 = static_cast<std::int64_t>(row1) * kGroups + group;
            const float w0            = Codec::load_one(codes, scales, index0, lane);
            const float w1            = Codec::load_one(codes, scales, index1, lane);
            const float xv            = __bfloat162float(x[group * Codec::kGroupK + lane]);
            acc0                      = fmaf(w0, xv, acc0);
            acc1                      = fmaf(w1, xv, acc1);
        }
    } else if (lane < Codec::kGroupK / 2) {
        for (int group = first_group; group < last_group; ++group) {
            float w00, w01, w10, w11;
            Codec::load_pair(codes, high, scales, static_cast<std::int64_t>(row0) * kGroups + group,
                             lane, w00, w01);
            Codec::load_pair(codes, high, scales, static_cast<std::int64_t>(row1) * kGroups + group,
                             lane, w10, w11);
            const int k     = group * Codec::kGroupK + lane * 2;
            const float2 xv = __bfloat1622float2(load_vec<__nv_bfloat162>(x + k));
            acc0            = fmaf(w00, xv.x, acc0);
            acc0            = fmaf(w01, xv.y, acc0);
            acc1            = fmaf(w10, xv.x, acc1);
            acc1            = fmaf(w11, xv.y, acc1);
        }
    }
    result0 = warp_reduce_sum(acc0);
    result1 = warp_reduce_sum(acc1);
}

template <class RoutedCodec>
__global__ void sparse_moe_d3_nine_warp_kernel(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales, const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, float* __restrict__ act,
    const std::uint8_t* __restrict__ warm_codes, const std::uint8_t* __restrict__ warm_high,
    const std::uint8_t* __restrict__ warm_scales,
    const unsigned* __restrict__ expert_residency) {
    __shared__ __align__(16) __nv_bfloat16 x_shared[kHidden];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid == 0) { pdl::trigger_dependents(); }
    if (tid < 256) { store_vec(x_shared + tid * 8, load_vec<uint4>(x + tid * 8)); }
    __syncthreads();

    const int j = static_cast<int>(blockIdx.x);
    float gate  = 0.0f;
    float up    = 0.0f;
    if (warp < kTopK) {
        pdl::wait_for_dependencies();
        const int expert   = ids[warp];
        bool warm = false;
        int tier_row = expert;
        sparse_moe_residency(expert_residency, expert, warm, tier_row);
        const int row_base = tier_row * 1024;
        dot_two_rows<RoutedCodec, kHidden>(warm ? warm_codes : routed_codes,
                                           warm ? warm_high : routed_high,
                                           warm ? warm_scales : routed_scales, row_base + j,
                                           row_base + kIntermediate + j, x_shared, 0, kHidden, gate,
                                           up);
    } else {
        dot_two_rows<Q8Codec, kHidden>(shared_codes, nullptr, shared_scales, j, kIntermediate + j,
                                       x_shared, 0, kHidden, gate, up);
    }
    if (lane == 0) { act[static_cast<std::int64_t>(warp) * kIntermediate + j] = silu(gate) * up; }
}

template <class RoutedCodec, int PathsPerBlock, bool Adaptive>
__global__ void sparse_moe_d3_path_tiled_kernel(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ token_ids,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales, const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, float* __restrict__ token_activations,
    int tokens, const int* __restrict__ adaptive_route_jobs,
    const std::uint8_t* __restrict__ warm_codes, const std::uint8_t* __restrict__ warm_high,
    const std::uint8_t* __restrict__ warm_scales,
    const unsigned* __restrict__ expert_residency) {
    // Three path CTAs per token/output row expose enough blocks for the 170-SM target and keep the
    // heavier shared Q8 path from holding eight completed routed warps resident.
    static_assert(PathsPerBlock > 0 && (kTopK + 1) % PathsPerBlock == 0);
    constexpr int kPathBlocks = (kTopK + 1) / PathsPerBlock;
    __shared__ __align__(16) __nv_bfloat16 x_shared[kHidden];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if constexpr (Adaptive) {
        if (*adaptive_route_jobs >= 0) { return; }
    }
    const int total_work = tokens * kPathBlocks * kIntermediate;
    const int first_work =
        Adaptive ? static_cast<int>(blockIdx.x)
                 : static_cast<int>(blockIdx.y) * kIntermediate + static_cast<int>(blockIdx.x);
    const int work_stride = Adaptive ? static_cast<int>(gridDim.x) : total_work;
    for (int work = first_work; work < total_work; work += work_stride) {
        const int token_path = work / kIntermediate;
        const int j          = work - token_path * kIntermediate;
        const int token      = token_path / kPathBlocks;
        const int path_block = token_path - token * kPathBlocks;
        const int path       = path_block * PathsPerBlock + warp;
        const auto* input    = x + static_cast<std::int64_t>(token) * kHidden;
        for (int vector = tid; vector < kHidden / 8; vector += static_cast<int>(blockDim.x)) {
            store_vec(x_shared + vector * 8, load_vec<uint4>(input + vector * 8));
        }
        __syncthreads();

        constexpr int kRouterPartitions = 4;
        const int activation_begin      = (token * (kTopK + 1) + path) * kIntermediate;
        // Routed paths need S2's ids. A shared path is independent only when its output lies
        // beyond the partial-score prefix that S2 may still read from the lifetime-unioned scratch.
        const bool must_wait_for_s2 =
            path < kTopK || activation_begin < tokens * kRouterRows * kRouterPartitions;
        if constexpr (!Adaptive) {
            if (must_wait_for_s2) { pdl::wait_for_dependencies(); }
        }
        float gate = 0.0f;
        float up   = 0.0f;
        if (path < kTopK) {
            const int expert   = token_ids[token * kTopK + path];
            bool warm = false;
            int tier_row = expert;
            sparse_moe_residency(expert_residency, expert, warm, tier_row);
            const int row_base = tier_row * (2 * kIntermediate);
            dot_two_rows<RoutedCodec, kHidden>(warm ? warm_codes : routed_codes,
                                               warm ? warm_high : routed_high,
                                               warm ? warm_scales : routed_scales,
                                               row_base + j, row_base + kIntermediate + j, x_shared,
                                               0, kHidden, gate, up);
        } else {
            dot_two_rows<Q8Codec, kHidden>(shared_codes, nullptr, shared_scales, j,
                                           kIntermediate + j, x_shared, 0, kHidden, gate, up);
        }
        if (lane == 0) {
            token_activations[(static_cast<std::int64_t>(token) * (kTopK + 1) + path) *
                                  kIntermediate +
                              j] = silu(gate) * up;
        }
        if constexpr (Adaptive) { __syncthreads(); }
    }
}

template <class Codec, int Rows>
__device__ __forceinline__ void dot_fp32_rows(const std::uint8_t* codes, const std::uint8_t* high,
                                              const std::uint8_t* scales, int row_base,
                                              const float* x, int first_group, int last_group,
                                              float (&result)[Rows]) {
    constexpr int kGroups = kIntermediate / Codec::kGroupK;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    float acc[Rows];
#pragma unroll
    for (int row = 0; row < Rows; ++row) { acc[row] = 0.0f; }
    if constexpr (Codec::kPackedWord8) {
        // Q5/Q6 use the same eight-value lane ownership as D3. The high plane and FP16 scale are
        // decoded exactly from their registered row-split codec before FP32 accumulation.
        const int lane_group    = lane >> 3;
        const int lane_in_group = lane & 7;
        for (int group_base = first_group; group_base < last_group; group_base += 4) {
            const int group = group_base + lane_group;
            const float4 x0 = load_vec<float4>(x + group * Codec::kGroupK + lane_in_group * 8);
            const float4 x1 = load_vec<float4>(x + group * Codec::kGroupK + lane_in_group * 8 + 4);
            const float values[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                const std::int64_t group_index =
                    static_cast<std::int64_t>(row_base + row) * kGroups + group;
                float weights[8];
                Codec::load_eight(codes, high, scales, group_index, lane_in_group, weights);
#pragma unroll
                for (int item = 0; item < 8; ++item) {
                    acc[row] = fmaf(weights[item], values[item], acc[row]);
                }
            }
        }
    } else if (lane < Codec::kGroupK / 2) {
        for (int group = first_group; group < last_group; ++group) {
            const int k     = group * Codec::kGroupK + lane * 2;
            const float2 xv = load_vec<float2>(x + k);
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                float w0, w1;
                Codec::load_pair(codes, high, scales,
                                 static_cast<std::int64_t>(row_base + row) * kGroups + group, lane,
                                 w0, w1);
                acc[row] = fmaf(w0, xv.x, acc[row]);
                acc[row] = fmaf(w1, xv.y, acc[row]);
            }
        }
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row) { result[row] = warp_reduce_sum(acc[row]); }
}

template <class RoutedCodec, int Rows>
__global__ void sparse_moe_d4_nine_warp_kernel(
    const int* __restrict__ ids, const float* __restrict__ alpha,
    const float* __restrict__ shared_scale, const float* __restrict__ act,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales, const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, __nv_bfloat16* __restrict__ destination,
    const char* __restrict__ prefetch_data, unsigned long long prefetch_bytes,
    const std::uint8_t* __restrict__ warm_codes, const std::uint8_t* __restrict__ warm_high,
    const std::uint8_t* __restrict__ warm_scales,
    const unsigned* __restrict__ expert_residency) {
    __shared__ float paths[kTopK + 1][Rows];
    pdl::wait_for_dependencies();
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int row_base = static_cast<int>(blockIdx.x) * Rows;
    if (warp < kTopK) {
        const int expert = ids[warp];
        bool warm = false;
        int tier_row = expert;
        sparse_moe_residency(expert_residency, expert, warm, tier_row);
        float dot[Rows];
        dot_fp32_rows<RoutedCodec, Rows>(warm ? warm_codes : routed_codes,
                                         warm ? warm_high : routed_high,
                                         warm ? warm_scales : routed_scales,
                                         tier_row * kHidden + row_base,
                                         act + static_cast<std::int64_t>(warp) * kIntermediate, 0,
                                         kIntermediate / RoutedCodec::kGroupK, dot);
        if (lane == 0) {
#pragma unroll
            for (int row = 0; row < Rows; ++row) { paths[warp][row] = alpha[warp] * dot[row]; }
        }
    } else {
        float dot[Rows];
        dot_fp32_rows<Q8Codec, Rows>(shared_codes, nullptr, shared_scales, row_base,
                                     act + static_cast<std::int64_t>(kTopK) * kIntermediate, 0,
                                     kIntermediate / Q8Codec::kGroupK, dot);
        if (lane == 0) {
#pragma unroll
            for (int row = 0; row < Rows; ++row) { paths[kTopK][row] = *shared_scale * dot[row]; }
        }
    }
    if (prefetch_data != nullptr) {
        // Fire-and-forget L2 warmup of the next consumer's weight codes,
        // issued BEFORE the block barrier so it hides behind the slowest warp instead
        // of extending the grid tail. Pure cache hint; no values, no addition order.
        const unsigned long long offset =
            (static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x) * 128ull;
        if (offset < prefetch_bytes) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(prefetch_data + offset));
        }
    }
    __syncthreads();
    if (warp == 0 && lane < Rows) {
        float value = __bfloat162float(destination[row_base + lane]);
#pragma unroll
        for (int path = 0; path < kTopK + 1; ++path) { value += paths[path][lane]; }
        destination[row_base + lane] = __float2bfloat16_rn(value);
    }
}

template <class RoutedCodec, int Rows, bool Adaptive>
__global__ void sparse_moe_d4_token_kernel(
    const int* __restrict__ token_ids, const float* __restrict__ token_alpha,
    const float* __restrict__ shared_scale, const float* __restrict__ token_activations,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales, const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, __nv_bfloat16* __restrict__ destination,
    int tokens, const int* __restrict__ adaptive_route_jobs,
    const std::uint8_t* __restrict__ warm_codes, const std::uint8_t* __restrict__ warm_high,
    const std::uint8_t* __restrict__ warm_scales,
    const unsigned* __restrict__ expert_residency) {
    // Token is a grid dimension rather than an in-CTA serial loop. Rows lets one routed-weight
    // stream serve adjacent outputs while retaining the deterministic rank-order FP32 epilogue.
    __shared__ float paths[kTopK + 1][Rows];
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if constexpr (Adaptive) {
        if (*adaptive_route_jobs >= 0) { return; }
    }
    constexpr int kRowBlocks = kHidden / Rows;
    const int total_work     = tokens * kRowBlocks;
    const int first_work =
        Adaptive ? static_cast<int>(blockIdx.x)
                 : static_cast<int>(blockIdx.y) * kRowBlocks + static_cast<int>(blockIdx.x);
    const int work_stride = Adaptive ? static_cast<int>(gridDim.x) : total_work;
    for (int work = first_work; work < total_work; work += work_stride) {
        const int token     = work / kRowBlocks;
        const int row_block = work - token * kRowBlocks;
        const int row_base  = row_block * Rows;
        const float* act =
            token_activations + static_cast<std::int64_t>(token) * (kTopK + 1) * kIntermediate;
        if (warp < kTopK) {
            const int expert = token_ids[token * kTopK + warp];
            bool warm = false;
            int tier_row = expert;
            sparse_moe_residency(expert_residency, expert, warm, tier_row);
            float dot[Rows];
            dot_fp32_rows<RoutedCodec, Rows>(warm ? warm_codes : routed_codes,
                                             warm ? warm_high : routed_high,
                                             warm ? warm_scales : routed_scales,
                                             tier_row * kHidden + row_base,
                                             act + static_cast<std::int64_t>(warp) * kIntermediate,
                                             0, kIntermediate / RoutedCodec::kGroupK, dot);
            if (lane == 0) {
#pragma unroll
                for (int row = 0; row < Rows; ++row) {
                    paths[warp][row] = token_alpha[token * kTopK + warp] * dot[row];
                }
            }
        } else {
            float dot[Rows];
            dot_fp32_rows<Q8Codec, Rows>(shared_codes, nullptr, shared_scales, row_base,
                                         act + static_cast<std::int64_t>(kTopK) * kIntermediate, 0,
                                         kIntermediate / Q8Codec::kGroupK, dot);
            if (lane == 0) {
#pragma unroll
                for (int row = 0; row < Rows; ++row) {
                    paths[kTopK][row] = shared_scale[token] * dot[row];
                }
            }
        }
        __syncthreads();
        if (warp == 0 && lane < Rows) {
            __nv_bfloat16* output =
                destination + static_cast<std::int64_t>(token) * kHidden + row_base + lane;
            float value = __bfloat162float(*output);
#pragma unroll
            for (int path = 0; path < kTopK + 1; ++path) { value += paths[path][lane]; }
            *output = __float2bfloat16_rn(value);
        }
        if constexpr (Adaptive) { __syncthreads(); }
    }
}

void launch_d1(const Tensor& x, const SparseMoeWeights& weights,
               const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    sparse_moe_d1_kernel<<<kRouterRows, kD1Warps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weights.router_shared_gate.qdata),
        static_cast<float*>(workspace.scratch.data),
        static_cast<const char*>(weights.shared_down.qdata),
        static_cast<unsigned long long>(kHidden) * kIntermediate);
    CUDA_CHECK(cudaGetLastError());
}

template <class Codec>
void launch_d3_dependent_codec(const Tensor& x, const SparseMoeWeights& weights,
                               const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    const auto* input         = static_cast<const __nv_bfloat16*>(x.data);
    const auto* ids           = static_cast<const int*>(workspace.ids.data);
    auto* act                 = static_cast<float*>(workspace.scratch.data);
    const auto* routed_codes  = static_cast<const std::uint8_t*>(weights.routed_gate_up.qdata);
    const auto* routed_high   = static_cast<const std::uint8_t*>(weights.routed_gate_up.qhigh);
    const auto* routed_scales = static_cast<const std::uint8_t*>(weights.routed_gate_up.scales);
    const auto* warm_codes    = static_cast<const std::uint8_t*>(weights.warm_routed_gate_up.qdata);
    const auto* warm_high     = static_cast<const std::uint8_t*>(weights.warm_routed_gate_up.qhigh);
    const auto* warm_scales   = static_cast<const std::uint8_t*>(weights.warm_routed_gate_up.scales);
    const unsigned* expert_residency = weights.expert_residency;
    const auto* shared_codes  = static_cast<const std::uint8_t*>(weights.shared_gate_up.qdata);
    const auto* shared_scales = static_cast<const std::uint8_t*>(weights.shared_gate_up.scales);
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(kIntermediate), dim3(9 * 32), 0, stream}, sparse_moe_d3_nine_warp_kernel<Codec>,
        input, ids, routed_codes, routed_high, routed_scales, shared_codes, shared_scales, act,
        warm_codes, warm_high, warm_scales, expert_residency));
}

void launch_d2_d3(const Tensor& x, const SparseMoeWeights& weights,
                  const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    const auto* scores = static_cast<const float*>(workspace.scratch.data);
    auto* ids          = static_cast<int*>(workspace.ids.data);
    auto* alpha        = static_cast<float*>(workspace.alpha.data);
    auto* shared_scale = static_cast<float*>(workspace.shared_scale.data);
    sparse_moe_d2_warp_kernel<<<1, 32, 0, stream>>>(
        scores, ids, alpha, shared_scale, weights.expert_activations);
    CUDA_CHECK(cudaGetLastError());

    switch (weights.routed_gate_up.qtype) {
    case QType::Q4_G64_FP16:
        launch_d3_dependent_codec<Q4Codec>(x, weights, workspace, stream);
        return;
    case QType::Q8_G32_FP16:
        launch_d3_dependent_codec<Q8Codec>(x, weights, workspace, stream);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported D3 codec");
    }
}

template <class Codec>
void launch_d4_dependent_codec(const SparseMoeWeights& weights, Tensor& destination,
                               const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream,
                               const void* prefetch_data, std::size_t prefetch_bytes) {
    const auto* ids           = static_cast<const int*>(workspace.ids.data);
    const auto* alpha         = static_cast<const float*>(workspace.alpha.data);
    const auto* shared_scale  = static_cast<const float*>(workspace.shared_scale.data);
    const auto* act           = static_cast<const float*>(workspace.scratch.data);
    const auto* routed_codes  = static_cast<const std::uint8_t*>(weights.routed_down.qdata);
    const auto* routed_high   = static_cast<const std::uint8_t*>(weights.routed_down.qhigh);
    const auto* routed_scales = static_cast<const std::uint8_t*>(weights.routed_down.scales);
    const auto* warm_codes    = static_cast<const std::uint8_t*>(weights.warm_routed_down.qdata);
    const auto* warm_high     = static_cast<const std::uint8_t*>(weights.warm_routed_down.qhigh);
    const auto* warm_scales   = static_cast<const std::uint8_t*>(weights.warm_routed_down.scales);
    const unsigned* expert_residency = weights.expert_residency;
    const auto* shared_codes  = static_cast<const std::uint8_t*>(weights.shared_down.qdata);
    const auto* shared_scales = static_cast<const std::uint8_t*>(weights.shared_down.scales);
    auto* output              = static_cast<__nv_bfloat16*>(destination.data);
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(kHidden), dim3(9 * 32), 0, stream}, sparse_moe_d4_nine_warp_kernel<Codec, 1>, ids,
        alpha, shared_scale, act, routed_codes, routed_high, routed_scales, shared_codes,
        shared_scales, output, static_cast<const char*>(prefetch_data),
        static_cast<unsigned long long>(prefetch_bytes), warm_codes, warm_high, warm_scales,
        expert_residency));
}

void launch_d4_dependent(const SparseMoeWeights& weights, Tensor& destination,
                         const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream,
                         const void* prefetch_data, std::size_t prefetch_bytes) {
    switch (weights.routed_down.qtype) {
    case QType::Q5_G64_FP16:
        launch_d4_dependent_codec<Q5Codec>(weights, destination, workspace, stream, prefetch_data,
                                           prefetch_bytes);
        return;
    case QType::Q6_G64_FP16:
        launch_d4_dependent_codec<Q6Codec>(weights, destination, workspace, stream, prefetch_data,
                                           prefetch_bytes);
        return;
    case QType::Q8_G32_FP16:
        launch_d4_dependent_codec<Q8Codec>(weights, destination, workspace, stream, prefetch_data,
                                           prefetch_bytes);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported D4 codec");
    }
}

template <class Codec, int PathsPerBlock, bool Adaptive>
void launch_d3_small_t_paths(const Tensor& x, const SparseMoeWeights& weights, const int* token_ids,
                             float* token_activations, std::int32_t tokens, cudaStream_t stream,
                             const int* adaptive_route_jobs) {
    constexpr int kPathBlocks = (kTopK + 1) / PathsPerBlock;
    const auto* input         = static_cast<const __nv_bfloat16*>(x.data);
    const auto* routed_codes  = static_cast<const std::uint8_t*>(weights.routed_gate_up.qdata);
    const auto* routed_high   = static_cast<const std::uint8_t*>(weights.routed_gate_up.qhigh);
    const auto* routed_scales = static_cast<const std::uint8_t*>(weights.routed_gate_up.scales);
    const auto* warm_codes    = static_cast<const std::uint8_t*>(weights.warm_routed_gate_up.qdata);
    const auto* warm_high     = static_cast<const std::uint8_t*>(weights.warm_routed_gate_up.qhigh);
    const auto* warm_scales   = static_cast<const std::uint8_t*>(weights.warm_routed_gate_up.scales);
    const unsigned* expert_residency = weights.expert_residency;
    const auto* shared_codes  = static_cast<const std::uint8_t*>(weights.shared_gate_up.qdata);
    const auto* shared_scales = static_cast<const std::uint8_t*>(weights.shared_gate_up.scales);
    if constexpr (Adaptive) {
        sparse_moe_d3_path_tiled_kernel<Codec, PathsPerBlock, true>
            <<<kAdaptiveD3Blocks, PathsPerBlock * 32, 0, stream>>>(
                input, token_ids, routed_codes, routed_high, routed_scales, shared_codes,
                shared_scales, token_activations, tokens, adaptive_route_jobs, warm_codes,
                warm_high, warm_scales, expert_residency);
        CUDA_CHECK(cudaGetLastError());
    } else {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(kIntermediate, tokens * kPathBlocks), dim3(PathsPerBlock * 32), 0, stream},
            sparse_moe_d3_path_tiled_kernel<Codec, PathsPerBlock, false>, input, token_ids,
            routed_codes, routed_high, routed_scales, shared_codes, shared_scales,
            token_activations, tokens, nullptr, warm_codes, warm_high, warm_scales,
            expert_residency));
    }
}

template <class Codec, bool Adaptive>
void launch_d3_small_t_codec(const Tensor& x, const SparseMoeWeights& weights, const int* token_ids,
                             float* token_activations, std::int32_t tokens,
                             SparseMoeSmallTD3Schedule schedule, cudaStream_t stream,
                             const int* adaptive_route_jobs) {
    switch (schedule) {
    case SparseMoeSmallTD3Schedule::Paths1:
        launch_d3_small_t_paths<Codec, 1, Adaptive>(x, weights, token_ids, token_activations,
                                                    tokens, stream, adaptive_route_jobs);
        return;
    case SparseMoeSmallTD3Schedule::Paths3:
        launch_d3_small_t_paths<Codec, 3, Adaptive>(x, weights, token_ids, token_activations,
                                                    tokens, stream, adaptive_route_jobs);
        return;
    case SparseMoeSmallTD3Schedule::Paths9:
        launch_d3_small_t_paths<Codec, 9, Adaptive>(x, weights, token_ids, token_activations,
                                                    tokens, stream, adaptive_route_jobs);
        return;
    }
    throw std::logic_error("sparse_moe: unknown small-T D3 schedule");
}

template <class Codec, int Rows, bool Adaptive>
void launch_d4_small_t_rows(const SparseMoeWeights& weights, Tensor& destination,
                            const int* token_ids, const float* token_alpha,
                            const float* shared_scale, const float* token_activations,
                            std::int32_t tokens, cudaStream_t stream,
                            const int* adaptive_route_jobs) {
    const dim3 grid = Adaptive ? dim3(kAdaptiveD4Blocks) : dim3(kHidden / Rows, tokens);
    sparse_moe_d4_token_kernel<Codec, Rows, Adaptive><<<grid, 9 * 32, 0, stream>>>(
        token_ids, token_alpha, shared_scale, token_activations,
        static_cast<const std::uint8_t*>(weights.routed_down.qdata),
        static_cast<const std::uint8_t*>(weights.routed_down.qhigh),
        static_cast<const std::uint8_t*>(weights.routed_down.scales),
        static_cast<const std::uint8_t*>(weights.shared_down.qdata),
        static_cast<const std::uint8_t*>(weights.shared_down.scales),
        static_cast<__nv_bfloat16*>(destination.data), tokens, adaptive_route_jobs,
        static_cast<const std::uint8_t*>(weights.warm_routed_down.qdata),
        static_cast<const std::uint8_t*>(weights.warm_routed_down.qhigh),
        static_cast<const std::uint8_t*>(weights.warm_routed_down.scales),
        weights.expert_residency);
    CUDA_CHECK(cudaGetLastError());
}

template <class Codec, bool Adaptive>
void launch_d4_small_t_codec(const SparseMoeWeights& weights, Tensor& destination,
                             const int* token_ids, const float* token_alpha,
                             const float* shared_scale, const float* token_activations,
                             std::int32_t tokens, SparseMoeSmallTD4Schedule schedule,
                             cudaStream_t stream, const int* adaptive_route_jobs) {
    switch (schedule) {
    case SparseMoeSmallTD4Schedule::Rows1:
        launch_d4_small_t_rows<Codec, 1, Adaptive>(weights, destination, token_ids, token_alpha,
                                                   shared_scale, token_activations, tokens, stream,
                                                   adaptive_route_jobs);
        return;
    case SparseMoeSmallTD4Schedule::Rows2:
        launch_d4_small_t_rows<Codec, 2, Adaptive>(weights, destination, token_ids, token_alpha,
                                                   shared_scale, token_activations, tokens, stream,
                                                   adaptive_route_jobs);
        return;
    case SparseMoeSmallTD4Schedule::Rows4:
        launch_d4_small_t_rows<Codec, 4, Adaptive>(weights, destination, token_ids, token_alpha,
                                                   shared_scale, token_activations, tokens, stream,
                                                   adaptive_route_jobs);
        return;
    }
    throw std::logic_error("sparse_moe: unknown small-T D4 schedule");
}

// ---- SP-7 EXL3 decode kernels ----
// The three-kernel EXL3 tail of the single-token decode route. D3a computes the gate and up
// inner products (the per-expert input transforms fused per block, the trellis decoded inline
// per weight); D4a applies the gate/up output transforms and SiLU, the down input transform
// and the down inner products; D4b folds the block-diagonal output transform, the per-expert
// svh, the route weights, the shared expert and the residual into the BF16 destination. Each
// expert's directory entry resolves its trellis extent and bitrate at runtime, so mixed-width
// banks stay on this one route (contract section 7).

constexpr int kExl3GateRowGroups = 4;    // 512 gate/up rows as 4 groups of 128
constexpr int kExl3DownRowGroups = 8;    // 2048 down rows per group of 256
constexpr int kExl3MaxTileWords  = 64;   // 4 * k2 at the k2 = 16 (K = 8) maximum

struct Exl3Entry {
    const std::uint32_t* words;  // the expert's ring words (little-endian int16 pairs)
    std::uint32_t k2;           // 2K: 2..16, or 3/5/7 with the mul1 codebook
    std::uint32_t words_per_tile;
};

// One per-expert directory entry: {u64 payload-relative trellis offset, u32 extent, u8 2K,
// three zero bytes} (contract section 4).
__device__ __forceinline__ Exl3Entry exl3_entry(const std::uint8_t* payload, int expert) {
    const std::uint8_t* raw = payload + static_cast<std::size_t>(expert) * 16;
    const std::uint64_t offset = *reinterpret_cast<const std::uint64_t*>(raw);
    const std::uint32_t k2 = static_cast<std::uint32_t>(raw[12]);
    return {reinterpret_cast<const std::uint32_t*>(payload + offset), k2, 4 * k2};
}

// One 128-value input-transform block: u = H128/sqrt(128) applied to (x ⊙ suh), staged to
// `destination` (contract section 2.1's input side). The D3a route reads represented BF16
// activations; the D4a down transform reads its FP32 SiLU product.
__device__ __forceinline__ void exl3_input_transform_block(const __nv_bfloat16* x,
                                                          const __half* suh, float* destination,
                                                          int block) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int base = block * 128 + 4 * lane;
    hadamard::warp_transform_128(
        [&](float (&v)[4]) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                v[q] = __bfloat162float(x[base + q]) * __half2float(suh[base + q]);
            }
        },
        [&](const float (&v)[4]) {
#pragma unroll
            for (int q = 0; q < 4; ++q) { destination[base + q] = v[q]; }
        });
}

__device__ __forceinline__ void exl3_input_transform_block(const float* x, const __half* suh,
                                                          float* destination, int block) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int base = block * 128 + 4 * lane;
    hadamard::warp_transform_128(
        [&](float (&v)[4]) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                v[q] = x[base + q] * __half2float(suh[base + q]);
            }
        },
        [&](const float (&v)[4]) {
#pragma unroll
            for (int q = 0; q < 4; ++q) { destination[base + q] = v[q]; }
        });
}

// D3a: the gate and up inner products. Grid (9 paths, 8 row-groups), 256 threads. Paths
// 0..7 are the selected EXL3 experts (their own suh and trellis extents); path 8 is the
// shared expert on the existing groupwise Q8 route, writing its activation slot directly.
__global__ void sparse_moe_d3a_exl3_kernel(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
    float* __restrict__ exl3_z, float* __restrict__ scratch,
    const std::uint8_t* __restrict__ gate_payload, const std::uint8_t* __restrict__ up_payload,
    std::uint64_t gate_suh_offset, std::uint64_t up_suh_offset, std::uint32_t gate_codebook,
    std::uint32_t up_codebook, const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales) {
    __shared__ __align__(16) __nv_bfloat16 x_shared[kHidden];
    __shared__ float u_gate[kHidden];
    __shared__ float u_up[kHidden];
    __shared__ std::uint32_t tile_words[8][2][kExl3MaxTileWords];
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5, lane = tid & 31;
    const int path = static_cast<int>(blockIdx.x);
    const int group = static_cast<int>(blockIdx.y);
    if (tid == 0) { pdl::trigger_dependents(); }
    for (int vector = tid; vector < kHidden / 8; vector += 256) {
        store_vec(x_shared + vector * 8, load_vec<uint4>(x + vector * 8));
    }
    __syncthreads();
    if (path == kTopK) {
        pdl::wait_for_dependencies();
        for (int j = group * 128 + warp * 16, end = j + 16; j < end; ++j) {
            float gate = 0.0f, up = 0.0f;
            dot_two_rows<Q8Codec, kHidden>(shared_codes, nullptr, shared_scales, j,
                                            kIntermediate + j, x_shared, 0, kHidden, gate, up);
            scratch[8 * kIntermediate + j] = silu(gate) * up;
        }
        return;
    }
    pdl::wait_for_dependencies();
    const int expert = ids[path];
    const Exl3Entry gate_entry = exl3_entry(gate_payload, expert);
    const Exl3Entry up_entry = exl3_entry(up_payload, expert);
    const __half* gate_suh =
        reinterpret_cast<const __half*>(gate_payload + gate_suh_offset) +
        static_cast<std::int64_t>(expert) * kHidden;
    const __half* up_suh =
        reinterpret_cast<const __half*>(up_payload + up_suh_offset) +
        static_cast<std::int64_t>(expert) * kHidden;
    // The input transforms: sixteen gate blocks then sixteen up blocks, four per warp.
    for (int block = warp; block < 2 * (kHidden / 128); block += 8) {
        if (block < kHidden / 128) {
            exl3_input_transform_block(x, gate_suh, u_gate, block);
        } else {
            exl3_input_transform_block(x, up_suh, u_up, block - kHidden / 128);
        }
    }
    __syncthreads();
    // The dot sweep: warp w owns the output row-tile (group * 8 + w) - sixteen outputs of
    // sixteen input columns per input tile, two lanes per output (the column halves). The 4
    // groups x 8 warps x one row-tile cover the 512 rows exactly once. The trellis is
    // [K/16, N/16] input-major, so tile (tc, row_tile) sits at tc * (N/16) + row_tile;
    // inside the tile, the element order maps the input-intra (the lane's c) to the trellis
    // row role and the output-intra (r) to its column role (contract 2.3).
    const int row_tile = group * 8 + warp;
    const int r = lane >> 1, half = lane & 1;
    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    for (int tc = 0; tc < kHidden / 16; ++tc) {
        const std::int64_t gate_tile =
            (static_cast<std::int64_t>(tc) * (kIntermediate / 16) + row_tile) *
            gate_entry.words_per_tile;
        const std::int64_t up_tile =
            (static_cast<std::int64_t>(tc) * (kIntermediate / 16) + row_tile) *
            up_entry.words_per_tile;
        // Staged bit-reversed: the ring's bit order is MSB-first within each stored word.
        for (int word = lane; word < gate_entry.words_per_tile; word += 32) {
            tile_words[warp][0][word] = __brev(gate_entry.words[gate_tile + word]);
        }
        for (int word = lane; word < up_entry.words_per_tile; word += 32) {
            tile_words[warp][1][word] = __brev(up_entry.words[up_tile + word]);
        }
        __syncwarp();
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            const int c = half * 8 + d;
            const int position =
                (d & 1) + (half << 1) + (((r >> 3) & 1) << 2) + (((d >> 1) & 3) << 3) +
                ((r & 7) << 5);
            const int column = tc * 16 + c;
            const std::uint32_t state_gate =
                exl3::state(tile_words[warp][0], gate_entry.words_per_tile, gate_entry.k2,
                           position);
            const std::uint32_t state_up =
                exl3::state(tile_words[warp][1], up_entry.words_per_tile, up_entry.k2, position);
            acc_gate = fmaf(exl3::weight(state_gate, gate_codebook), u_gate[column], acc_gate);
            acc_up = fmaf(exl3::weight(state_up, up_codebook), u_up[column], acc_up);
        }
        __syncwarp();
    }
    const float gate_row = acc_gate + __shfl_xor_sync(0xffffffffu, acc_gate, 1);
    const float up_row = acc_up + __shfl_xor_sync(0xffffffffu, acc_up, 1);
    if (half == 0) {
        const int row = row_tile * 16 + r;
        exl3_z[(static_cast<std::int64_t>(path) * 2 + 0) * kIntermediate + row] = gate_row;
        exl3_z[(static_cast<std::int64_t>(path) * 2 + 1) * kIntermediate + row] = up_row;
    }
}

// D4a: the gate/up output transforms and SiLU, the down input transform and the down inner
// products. Grid (9 paths, 8 down row-groups), 256 threads. Paths 0..7 stage their raw down
// inner products (D4b applies the output transform); path 8 is the shared expert's groupwise
// Q8 down product, already scaled, into its staging slot.
__global__ void sparse_moe_d4a_exl3_kernel(
    const int* __restrict__ ids, const float* __restrict__ alpha,
    const float* __restrict__ shared_scale, const float* __restrict__ exl3_z,
    const float* __restrict__ scratch, float* __restrict__ exl3_down,
    const std::uint8_t* __restrict__ gate_payload, const std::uint8_t* __restrict__ up_payload,
    const std::uint8_t* __restrict__ down_payload, std::uint64_t gate_svh_offset,
    std::uint64_t up_svh_offset, std::uint64_t down_suh_offset, std::uint32_t down_codebook,
    const std::uint8_t* __restrict__ shared_codes, const std::uint8_t* __restrict__ shared_scales) {
    __shared__ __align__(16) float gate[kIntermediate];
    __shared__ __align__(16) float up[kIntermediate];
    __shared__ float activation[kIntermediate];
    __shared__ float u_down[kIntermediate];
    __shared__ std::uint32_t tile_words[8][kExl3MaxTileWords];
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5, lane = tid & 31;
    const int path = static_cast<int>(blockIdx.x);
    const int group = static_cast<int>(blockIdx.y);
    if (tid == 0) { pdl::trigger_dependents(); }
    if (path == kTopK) {
        pdl::wait_for_dependencies();
        // The shared expert: the groupwise Q8 down rows, scaled, into staging slot 8.
        for (int row = group * 256 + warp * 32, end = row + 32; row < end; ++row) {
            float dot[1] = {0.0f};
            dot_fp32_rows<Q8Codec, 1>(shared_codes, nullptr, shared_scales, row,
                                      scratch + 8 * kIntermediate, 0,
                                      kIntermediate / Q8Codec::kGroupK, dot);
            exl3_down[8 * kHidden + row] = *shared_scale * dot[0];
        }
        return;
    }
    pdl::wait_for_dependencies();
    const int expert = ids[path];
    // The gate and up output transforms: z ⊙ svh per projection (four 128-blocks each), then
    // the SiLU product into the activation staging.
    {
        const float* z_gate = exl3_z + (static_cast<std::int64_t>(path) * 2 + 0) * kIntermediate;
        const float* z_up = exl3_z + (static_cast<std::int64_t>(path) * 2 + 1) * kIntermediate;
        const __half* gate_svh =
            reinterpret_cast<const __half*>(gate_payload + gate_svh_offset) +
            static_cast<std::int64_t>(expert) * kIntermediate;
        const __half* up_svh =
            reinterpret_cast<const __half*>(up_payload + up_svh_offset) +
            static_cast<std::int64_t>(expert) * kIntermediate;
        for (int block = warp; block < kIntermediate / 128; block += 8) {
            const int base = block * 128 + 4 * lane;
            hadamard::warp_transform_128(
                [&](float (&v)[4]) {
#pragma unroll
                    for (int q = 0; q < 4; ++q) { v[q] = z_gate[base + q]; }
                },
                [&](const float (&v)[4]) {
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        gate[base + q] = v[q] * __half2float(gate_svh[base + q]);
                    }
                });
            hadamard::warp_transform_128(
                [&](float (&v)[4]) {
#pragma unroll
                    for (int q = 0; q < 4; ++q) { v[q] = z_up[base + q]; }
                },
                [&](const float (&v)[4]) {
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        up[base + q] = v[q] * __half2float(up_svh[base + q]);
                    }
                });
        }
        __syncthreads();
        for (int c = tid; c < kIntermediate; c += 256) {
            activation[c] = silu(gate[c]) * up[c];
        }
        __syncthreads();
    }
    const Exl3Entry down_entry = exl3_entry(down_payload, expert);
    const __half* down_suh =
        reinterpret_cast<const __half*>(down_payload + down_suh_offset) +
        static_cast<std::int64_t>(expert) * kIntermediate;
    for (int block = warp; block < kIntermediate / 128; block += 8) {
        exl3_input_transform_block(activation, down_suh, u_down, block);
    }
    __syncthreads();
    // The down dot sweep: warp w owns the output row-tiles (group * 16 + 2w, +1) - two tiles
    // of sixteen outputs each, sweeping the 512-wide input's 32 tiles. The trellis is
    // [K/16, N/16] with K = 512 and N = 2048, so tile (tc, row_tile) sits at tc * (N/16) +
    // row_tile; the element order maps the input-intra (the lane's c) to the trellis row
    // role and the output-intra (r) to its column role.
    const int r = lane >> 1, half = lane & 1;
    float accumulators[2][16] = {};
#pragma unroll
    for (int tile_pair = 0; tile_pair < 2; ++tile_pair) {
        const int row_tile = group * 16 + warp * 2 + tile_pair;
        float acc = 0.0f;
        for (int tc = 0; tc < kIntermediate / 16; ++tc) {
            const std::int64_t tile =
                (static_cast<std::int64_t>(tc) * (kHidden / 16) + row_tile) *
                down_entry.words_per_tile;
            // Staged bit-reversed: the ring's bit order is MSB-first within each stored word.
            for (int word = lane; word < down_entry.words_per_tile; word += 32) {
                tile_words[warp][word] = __brev(down_entry.words[tile + word]);
            }
            __syncwarp();
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                const int c = half * 8 + d;
                const int position =
                    (d & 1) + (half << 1) + (((r >> 3) & 1) << 2) + (((d >> 1) & 3) << 3) +
                    ((r & 7) << 5);
                const std::uint32_t state =
                    exl3::state(tile_words[warp], down_entry.words_per_tile, down_entry.k2,
                               position);
                acc = fmaf(exl3::weight(state, down_codebook), u_down[tc * 16 + c], acc);
            }
            __syncwarp();
        }
        accumulators[tile_pair][r] = acc + __shfl_xor_sync(0xffffffffu, acc, 1);
    }
#pragma unroll
    for (int tile_pair = 0; tile_pair < 2; ++tile_pair) {
        const int row_tile = group * 16 + warp * 2 + tile_pair;
        if (half == 0) {
            const int row = row_tile * 16 + r;
            exl3_down[static_cast<std::int64_t>(path) * kHidden + row] =
                accumulators[tile_pair][r];
        }
    }
}

// D4b: the block-diagonal output transform, the per-expert svh and route weights, the shared
// expert and the residual, folded into the BF16 destination. Grid (16 output blocks), 256
// threads: warp p applies path p's transform to the staged down products of its block.
__global__ void sparse_moe_d4b_exl3_kernel(
    const int* __restrict__ ids, const float* __restrict__ alpha,
    const float* __restrict__ exl3_down, const std::uint8_t* __restrict__ down_payload,
    std::uint64_t down_svh_offset, const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ destination, const char* __restrict__ prefetch_data,
    unsigned long long prefetch_bytes) {
    __shared__ float block_accumulator[8][128];
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5, lane = tid & 31;
    const int block = static_cast<int>(blockIdx.x);
    if (tid == 0) { pdl::trigger_dependents(); }
    pdl::wait_for_dependencies();
    const int expert = ids[warp];
    const int base = block * 128 + 4 * lane;
    hadamard::warp_transform_128(
        [&](float (&v)[4]) {
            const float* staged =
                exl3_down + static_cast<std::int64_t>(warp) * kHidden + block * 128 + 4 * lane;
#pragma unroll
            for (int q = 0; q < 4; ++q) { v[q] = staged[q]; }
        },
        [&](const float (&v)[4]) {
            const int expert_base = block * 128 + 4 * lane;
            const __half* svh = reinterpret_cast<const __half*>(down_payload + down_svh_offset) +
                                 static_cast<std::int64_t>(expert) * kHidden;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                block_accumulator[warp][4 * lane + q] =
                    alpha[warp] * __half2float(svh[expert_base + q]) * v[q];
            }
        });
    __syncthreads();
    for (int j = tid; j < 128; j += 256) {
        float value = static_cast<float>(residual[block * 128 + j]);
        for (int path = 0; path < kTopK; ++path) { value += block_accumulator[path][j]; }
        value += exl3_down[8 * kHidden + block * 128 + j];
        destination[block * 128 + j] = __float2bfloat16(value);
    }
    if (prefetch_data != nullptr) {
        // Fire-and-forget L2 warmup of the next consumer's weights, issued before the block
        // completes so it hides behind the tail. Pure cache hint; no values, no order.
        const unsigned long long offset =
            (static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x) * 128ull;
        if (offset < prefetch_bytes) {
            asm volatile("prefetch.global.L2 [%0];" :: "l"(prefetch_data + offset));
        }
    }
}

void launch_d3a_exl3(const Tensor& x, const SparseMoeWeights& weights,
                     const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream);

void launch_d2_exl3_d3a(const Tensor& x, const SparseMoeWeights& weights,
                         const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    const auto* scores = static_cast<const float*>(workspace.scratch.data);
    auto* ids = static_cast<int*>(workspace.ids.data);
    auto* alpha = static_cast<float*>(workspace.alpha.data);
    auto* shared_scale = static_cast<float*>(workspace.shared_scale.data);
    sparse_moe_d2_warp_kernel<<<1, 32, 0, stream>>>(scores, ids, alpha, shared_scale,
                                                    weights.expert_activations);
    CUDA_CHECK(cudaGetLastError());
    launch_d3a_exl3(x, weights, workspace, stream);
}

void launch_d3a_exl3(const Tensor& x, const SparseMoeWeights& weights,
                     const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    const auto* input = static_cast<const __nv_bfloat16*>(x.data);
    const auto* ids = static_cast<const int*>(workspace.ids.data);
    auto* exl3_z = static_cast<float*>(workspace.exl3_z.data);
    auto* scratch = static_cast<float*>(workspace.scratch.data);
    const auto* gate_payload = static_cast<const std::uint8_t*>(weights.exl3_routed_gate.payload);
    const auto* up_payload = static_cast<const std::uint8_t*>(weights.exl3_routed_up.payload);
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(kTopK + 1, kExl3GateRowGroups), dim3(256), 0, stream},
        sparse_moe_d3a_exl3_kernel, input, ids, exl3_z, scratch, gate_payload, up_payload,
        weights.exl3_routed_gate.exl3_suh_offset, weights.exl3_routed_up.exl3_suh_offset,
        weights.exl3_routed_gate.exl3_codebook, weights.exl3_routed_up.exl3_codebook,
        static_cast<const std::uint8_t*>(weights.shared_gate_up.qdata),
        static_cast<const std::uint8_t*>(weights.shared_gate_up.scales)));
}

void launch_d4_exl3(const SparseMoeWeights& weights, Tensor& destination,
                    const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream,
                    const void* prefetch_data, std::size_t prefetch_bytes) {
    const auto* ids = static_cast<const int*>(workspace.ids.data);
    const auto* alpha = static_cast<const float*>(workspace.alpha.data);
    const auto* shared_scale = static_cast<const float*>(workspace.shared_scale.data);
    const auto* exl3_z = static_cast<const float*>(workspace.exl3_z.data);
    const auto* scratch = static_cast<const float*>(workspace.scratch.data);
    auto* exl3_down = static_cast<float*>(workspace.exl3_down.data);
    const auto* gate_payload = static_cast<const std::uint8_t*>(weights.exl3_routed_gate.payload);
    const auto* up_payload = static_cast<const std::uint8_t*>(weights.exl3_routed_up.payload);
    const auto* down_payload = static_cast<const std::uint8_t*>(weights.routed_down.payload);
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(kTopK + 1, kExl3DownRowGroups), dim3(256), 0, stream}, sparse_moe_d4a_exl3_kernel,
        ids, alpha, shared_scale, exl3_z, scratch, exl3_down, gate_payload, up_payload,
        down_payload, weights.exl3_routed_gate.exl3_svh_offset, weights.exl3_routed_up.exl3_svh_offset,
        weights.routed_down.exl3_suh_offset, weights.routed_down.exl3_codebook,
        static_cast<const std::uint8_t*>(weights.shared_down.qdata),
        static_cast<const std::uint8_t*>(weights.shared_down.scales)));
    const auto* residual = static_cast<const __nv_bfloat16*>(destination.data);
    auto* output = static_cast<__nv_bfloat16*>(destination.data);
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(kHidden / 128), dim3(kTopK * 32), 0, stream}, sparse_moe_d4b_exl3_kernel,
        static_cast<const int*>(workspace.ids.data), alpha, exl3_down, down_payload,
        weights.routed_down.exl3_svh_offset, residual, output,
        static_cast<const char*>(prefetch_data), static_cast<unsigned long long>(prefetch_bytes)));
}

} // namespace

void sparse_moe_decode_launch_d3_small_t(const Tensor& x, const SparseMoeWeights& weights,
                                         const int* token_ids, float* token_activations,
                                         std::int32_t tokens, SparseMoeSmallTD3Schedule schedule,
                                         cudaStream_t stream, const int* adaptive_route_jobs) {
    switch (weights.routed_gate_up.qtype) {
    case QType::Q4_G64_FP16:
        if (adaptive_route_jobs == nullptr) {
            launch_d3_small_t_codec<Q4Codec, false>(x, weights, token_ids, token_activations,
                                                    tokens, schedule, stream, nullptr);
        } else {
            launch_d3_small_t_codec<Q4Codec, true>(x, weights, token_ids, token_activations, tokens,
                                                   schedule, stream, adaptive_route_jobs);
        }
        return;
    case QType::Q8_G32_FP16:
        launch_d3_small_t_codec<Q8Codec, false>(x, weights, token_ids, token_activations, tokens,
                                                schedule, stream, nullptr);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported small-T D3 codec");
    }
}

void sparse_moe_decode_launch_d4_small_t(const SparseMoeWeights& weights, Tensor& destination,
                                         const int* token_ids, const float* token_alpha,
                                         const float* shared_scale, const float* token_activations,
                                         std::int32_t tokens, SparseMoeSmallTD4Schedule schedule,
                                         cudaStream_t stream, const int* adaptive_route_jobs) {
    switch (weights.routed_down.qtype) {
    case QType::Q5_G64_FP16:
        if (adaptive_route_jobs == nullptr) {
            launch_d4_small_t_codec<Q5Codec, false>(weights, destination, token_ids, token_alpha,
                                                    shared_scale, token_activations, tokens,
                                                    schedule, stream, nullptr);
        } else {
            launch_d4_small_t_codec<Q5Codec, true>(weights, destination, token_ids, token_alpha,
                                                   shared_scale, token_activations, tokens,
                                                   schedule, stream, adaptive_route_jobs);
        }
        return;
    case QType::Q6_G64_FP16:
        if (adaptive_route_jobs == nullptr) {
            launch_d4_small_t_codec<Q6Codec, false>(weights, destination, token_ids, token_alpha,
                                                    shared_scale, token_activations, tokens,
                                                    schedule, stream, nullptr);
        } else {
            launch_d4_small_t_codec<Q6Codec, true>(weights, destination, token_ids, token_alpha,
                                                   shared_scale, token_activations, tokens,
                                                   schedule, stream, adaptive_route_jobs);
        }
        return;
    case QType::Q8_G32_FP16:
        launch_d4_small_t_codec<Q8Codec, false>(weights, destination, token_ids, token_alpha,
                                                shared_scale, token_activations, tokens, schedule,
                                                stream, nullptr);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported small-T D4 codec");
    }
}

void sparse_moe_decode_launch(const Tensor& x, const SparseMoeWeights& weights, Tensor& destination,
                              const SparseMoeDecodePlan& plan,
                              const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    if (weights.exl3_routed_gate.payload != nullptr) {
        launch_d1(x, weights, workspace, stream);
        launch_d2_exl3_d3a(x, weights, workspace, stream);
        launch_d4_exl3(weights, destination, workspace, stream, plan.next_weight_prefetch,
                       plan.next_weight_prefetch_bytes);
        return;
    }
    launch_d1(x, weights, workspace, stream);
    launch_d2_d3(x, weights, workspace, stream);
    launch_d4_dependent(weights, destination, workspace, stream, plan.next_weight_prefetch,
                        plan.next_weight_prefetch_bytes);
}

} // namespace ninfer::ops::detail
