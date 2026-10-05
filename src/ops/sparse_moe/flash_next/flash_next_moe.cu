#include "ninfer/ops/flash_next_moe.h"

#include "ops/flash_next_work.h"

#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_gemm_mma.cuh"
#include "ops/linear/bf16/flash_next/bf16_gemv.cuh"
#include "ops/linear/bf16/flash_next/bf16_launch.h"
#include "ops/linear/fp8/flash_next_launch.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr int kHidden                 = 2560;
constexpr int kExperts                = 512;
constexpr int kTop                    = 10;
constexpr int kIntermediate           = 640;
constexpr int kWarps                  = 8;
constexpr int kGroupedFirstToken      = 1;
constexpr int kGroupedTokenTile       = 32;
constexpr int kLargeGroupedTokenTile  = 128;
constexpr int kDecodeGroupedTokenTile = 16;

using GroupedGateGeometry = detail::Nvfp4Geometry<2 * kIntermediate, kHidden>;
using GroupedDownGeometry = detail::Nvfp4Geometry<kHidden, kIntermediate>;
using GroupedGateSchedule = detail::Nvfp4W4a4MmaSchedule<kGroupedTokenTile, 256, 128, 2, 4, 2, 1>;
using GroupedDownSchedule = detail::Nvfp4W4a4MmaSchedule<kGroupedTokenTile, 256, 128, 2, 4, 2, 1>;
using LargeGroupedSchedule =
    detail::Nvfp4W4a4MmaSchedule<kLargeGroupedTokenTile, 256, 128, 4, 4, 3, 1>;
using DecodeGroupedSchedule =
    detail::Nvfp4W4a4MmaSchedule<kDecodeGroupedTokenTile, 128, 128, 1, 8, 2, 1>;
// Narrow gate tiles with 256-wide K steps keep more weight bytes in flight per expert.
using DecodeGroupedGateSchedule =
    detail::Nvfp4W4a4MmaSchedule<kDecodeGroupedTokenTile, 32, 256, 1, 4, 3, 1>;
using Bf16GroupedSchedule = detail::Bf16MmaSchedule<64, 64, 64, 32, 32, 3, 2, Cache::cg, Cache::cg,
                                                    detail::Bf16MmaFragmentPipeline::PingPong,
                                                    detail::Bf16MmaRaster::TokenFast>;
using Bf16GroupedGateGeometry = detail::Bf16Geometry<2 * kIntermediate, kHidden>;
using Bf16GroupedDownGeometry = detail::Bf16Geometry<kHidden, kIntermediate>;

constexpr int kRouteThreads = 256;
static_assert(kExperts == 2 * kRouteThreads && kHidden % kRouteThreads == 0);

__device__ __forceinline__ void route_token(const __nv_bfloat16* scores,
                                            const __nv_bfloat16* input,
                                            const __nv_bfloat16* shared_scale_weight, int* ids,
                                            float* alpha, float* shared_alpha, int token) {
    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const int warp    = static_cast<int>(threadIdx.x) >> 5;
    const int expert0 = static_cast<int>(threadIdx.x);
    const int expert1 = expert0 + kRouteThreads;
    __shared__ float warp_values[8];
    __shared__ float top_values[kTop];
    __shared__ int top_ids[kTop];
    // Compile-time trip count: all loads issue before the first use.
    float shared_value = 0.0F;
#pragma unroll
    for (int k = static_cast<int>(threadIdx.x); k < kHidden; k += kRouteThreads) {
        shared_value = fmaf(__bfloat162float(input[k + static_cast<std::int64_t>(kHidden) * token]),
                            __bfloat162float(shared_scale_weight[k]), shared_value);
    }
    for (int offset = 16; offset != 0; offset >>= 1) {
        shared_value += __shfl_down_sync(0xffffffffU, shared_value, offset);
    }
    if (lane == 0) { warp_values[warp] = shared_value; }
    __syncthreads();
    if (warp == 0) {
        shared_value = lane < 8 ? warp_values[lane] : 0.0F;
        for (int offset = 16; offset != 0; offset >>= 1) {
            shared_value += __shfl_down_sync(0xffffffffU, shared_value, offset);
        }
        if (lane == 0) { shared_alpha[token] = 1.0F / (1.0F + expf(-shared_value)); }
    }
    __syncthreads();
    // BF16 scores order exactly as 32-bit keys (order-preserving value bits, then the inverted
    // expert id), so (value descending, id ascending) is one unsigned maximum per round. The
    // global top-k lies within the union of the per-warp top-k sets: each warp selects locally
    // without block barriers and warp 0 selects from the 8 * kTop candidates.
    const auto key_of = [](__nv_bfloat16 value, int expert) {
        // -0 and +0 compare equal; canonicalize so their tie is broken by expert id.
        unsigned bits = __bfloat16_as_ushort(value);
        bits = bits == 0x8000U ? 0U : bits;
        const unsigned ordered = (bits & 0x8000U) != 0 ? (~bits & 0xffffU) : (bits | 0x8000U);
        return (ordered << 16) | static_cast<unsigned>(0xffff - expert);
    };
    const auto value_of = [](unsigned key) {
        const unsigned ordered = key >> 16;
        const unsigned bits = (ordered & 0x8000U) != 0 ? (ordered & 0x7fffU) : (~ordered & 0xffffU);
        return __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(bits)));
    };
    __shared__ unsigned candidates[8 * kTop];
    unsigned key0 = key_of(scores[expert0 + static_cast<std::int64_t>(kExperts) * token], expert0);
    unsigned key1 = key_of(scores[expert1 + static_cast<std::int64_t>(kExperts) * token], expert1);
#pragma unroll
    for (int rank = 0; rank < kTop; ++rank) {
        const unsigned best = __reduce_max_sync(0xffffffffU, max(key0, key1));
        if (lane == 0) { candidates[warp * kTop + rank] = best; }
        key0 = key0 == best ? 0U : key0;
        key1 = key1 == best ? 0U : key1;
    }
    __syncthreads();
    if (warp == 0) {
        constexpr int kSlots = (8 * kTop + 31) / 32;
        unsigned keys[kSlots];
#pragma unroll
        for (int slot = 0; slot < kSlots; ++slot) {
            const int index = lane + 32 * slot;
            keys[slot] = index < 8 * kTop ? candidates[index] : 0U;
        }
#pragma unroll
        for (int rank = 0; rank < kTop; ++rank) {
            unsigned local = keys[0];
#pragma unroll
            for (int slot = 1; slot < kSlots; ++slot) { local = max(local, keys[slot]); }
            const unsigned best = __reduce_max_sync(0xffffffffU, local);
#pragma unroll
            for (int slot = 0; slot < kSlots; ++slot) { keys[slot] = keys[slot] == best ? 0U : keys[slot]; }
            if (lane == 0) {
                top_values[rank] = value_of(best);
                top_ids[rank]    = 0xffff - static_cast<int>(best & 0xffffU);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float maximum = top_values[0];
        float denominator   = 0.0F;
#pragma unroll
        for (int rank = 0; rank < kTop; ++rank) { denominator += expf(top_values[rank] - maximum); }
#pragma unroll
        for (int rank = 0; rank < kTop; ++rank) {
            const int offset = rank + kTop * token;
            ids[offset]      = top_ids[rank];
            alpha[offset]    = expf(top_values[rank] - maximum) / denominator;
        }
    }
}

__global__ void __launch_bounds__(kRouteThreads)
    route_kernel(const __nv_bfloat16* scores, const __nv_bfloat16* input,
                 const __nv_bfloat16* shared_scale_weight, int* ids, float* alpha,
                 float* shared_alpha, int) {
    route_token(scores, input, shared_scale_weight, ids, alpha, shared_alpha,
                static_cast<int>(blockIdx.x));
}

// One-token decode: CTA 0 routes while the other CTAs compute the shared-expert Down projection
// [2560,640] from the already available shared activation. Each warp runs one block of the
// production single-warp BF16 decode schedule, so the output equals that GEMV bit for bit.
using SharedDownGeometry = detail::flash_next::Bf16GemvGeometry<kHidden, kIntermediate>;
using SharedDownSchedule = detail::flash_next::Bf16LinearDecodeSchedule<SharedDownGeometry>;
static_assert(SharedDownSchedule::kWarpsPerCta == 1 && SharedDownSchedule::kWarpsPerRow == 1 &&
              SharedDownSchedule::kActivationAccess ==
                  detail::flash_next::Bf16ActivationAccess::Direct);
constexpr int kSharedDownWarps  = kRouteThreads / 32;
constexpr int kSharedDownBlocks =
    kHidden / (SharedDownSchedule::kRowsPerCta * kSharedDownWarps);

__global__ void __launch_bounds__(kRouteThreads)
    route_shared_down_decode_kernel(const __nv_bfloat16* scores, const __nv_bfloat16* input,
                                    const __nv_bfloat16* shared_scale_weight, int* ids,
                                    float* alpha, float* shared_alpha,
                                    const __nv_bfloat16* __restrict__ shared_activation,
                                    const __nv_bfloat16* __restrict__ shared_down,
                                    __nv_bfloat16* __restrict__ destination) {
    if (blockIdx.x == 0) {
        route_token(scores, input, shared_scale_weight, ids, alpha, shared_alpha, 0);
        return;
    }
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row0 = ((static_cast<int>(blockIdx.x) - 1) * kSharedDownWarps + warp) *
                     SharedDownSchedule::kRowsPerCta;
    float accumulators[SharedDownSchedule::kRowsPerWarp][SharedDownSchedule::kAccumulatorChains] =
        {};
    detail::flash_next::compute_bf16_gemv_rows<SharedDownGeometry, SharedDownSchedule>(
        shared_activation, shared_down, row0, 0, lane, accumulators);
#pragma unroll
    for (int local_row = 0; local_row < SharedDownSchedule::kRowsPerWarp; ++local_row) {
        float total = 0.0F;
#pragma unroll
        for (int chain = 0; chain < SharedDownSchedule::kAccumulatorChains; ++chain) {
            total += accumulators[local_row][chain];
        }
        total = warp_reduce_sum(total);
        if (lane == 0) { destination[row0 + local_row] = __float2bfloat16_rn(total); }
    }
}

// Pack each expert in token-major order. Besides making the packed representation reproducible,
// scanning tokens (whose top-k expert ids are unique) avoids contended global atomics.
__global__ void count_routes_kernel(const int* ids, int* local_rank, int* counts, int tokens) {
    const int expert = static_cast<int>(blockIdx.x);
    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    __shared__ int warp_counts[8];
    __shared__ int running;
    if (tid == 0) { running = 0; }
    __syncthreads();

    for (int begin = 0; begin < tokens; begin += static_cast<int>(blockDim.x)) {
        const int token = begin + tid;
        int path        = -1;
        if (token < tokens) {
#pragma unroll
            for (int candidate = 0; candidate < kTop; ++candidate) {
                if (ids[candidate + kTop * token] == expert) { path = candidate; }
            }
        }
        const int selected = path >= 0 ? 1 : 0;
        int inclusive      = selected;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int add = __shfl_up_sync(0xffffffffU, inclusive, offset);
            if (lane >= offset) { inclusive += add; }
        }
        if (lane == 31) { warp_counts[warp] = inclusive; }
        __syncthreads();
        int warp_base = 0;
        if (warp == 0) {
            int value = lane < 8 ? warp_counts[lane] : 0;
#pragma unroll
            for (int offset = 1; offset < 8; offset <<= 1) {
                const int add = __shfl_up_sync(0xffffffffU, value, offset);
                if (lane >= offset) { value += add; }
            }
            if (lane < 8) { warp_counts[lane] = value; }
        }
        __syncthreads();
        if (warp != 0) { warp_base = warp_counts[warp - 1]; }
        const int chunk_count = warp_counts[7];
        if (path >= 0) { local_rank[path + kTop * token] = running + warp_base + inclusive - 1; }
        __syncthreads();
        if (tid == 0) { running += chunk_count; }
        __syncthreads();
    }
    if (tid == 0) { counts[expert] = running; }
}

__global__ void scan_routes_kernel(const int* counts, int* offsets) {
    __shared__ int scan[kExperts];
    const int expert = static_cast<int>(threadIdx.x);
    scan[expert]     = counts[expert];
    __syncthreads();
    for (int distance = 1; distance < kExperts; distance <<= 1) {
        const int add = expert >= distance ? scan[expert - distance] : 0;
        __syncthreads();
        scan[expert] += add;
        __syncthreads();
    }
    offsets[expert] = expert == 0 ? 0 : scan[expert - 1];
    if (expert == kExperts - 1) { offsets[kExperts] = scan[expert]; }
}

__global__ void make_route_jobs_kernel(const int* counts, int* job_experts, int* job_columns,
                                       int* job_count, int token_tile) {
    const int expert = static_cast<int>(threadIdx.x);
    __shared__ int prefix[kExperts];
    const int jobs = (counts[expert] + token_tile - 1) / token_tile;
    prefix[expert] = jobs;
    __syncthreads();
    for (int distance = 1; distance < kExperts; distance <<= 1) {
        const int add = expert >= distance ? prefix[expert - distance] : 0;
        __syncthreads();
        prefix[expert] += add;
        __syncthreads();
    }
    const int base = expert == 0 ? 0 : prefix[expert - 1];
    for (int local = 0; local < jobs; ++local) {
        job_experts[base + local] = expert;
        job_columns[base + local] = local * token_tile;
    }
    if (expert == kExperts - 1) { job_count[0] = prefix[expert]; }
}

__global__ void gather_quantize_routes_kernel(const __nv_bfloat16* input, const int* ids,
                                              const int* local_rank, const int* offsets,
                                              const float* input_divisors, std::uint8_t* codes,
                                              std::uint8_t* scales, int* packed_index,
                                              int* packed_expert, int assignments, int columns) {
    const int groups = columns / 16;
    const int task =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (task >= assignments * groups) { return; }
    const int assignment                      = task / groups;
    const int group                           = task - assignment * groups;
    const int expert                          = ids[assignment];
    const int packed                          = offsets[expert] + local_rank[assignment];
    const int token                           = assignment / kTop;
    const detail::Nvfp4QuantizedK16 quantized = detail::quantize_nvfp4_k16(
        input + static_cast<std::int64_t>(token) * columns + group * 16, input_divisors[expert]);
    auto* destination = codes + static_cast<std::int64_t>(packed) * (columns / 2) + group * 8;
    *reinterpret_cast<uint2*>(destination) = make_uint2(quantized.codes_lo, quantized.codes_hi);
    scales[static_cast<std::int64_t>(packed) * groups + group] = quantized.scale;
    if (group == 0) {
        packed_index[assignment] = packed;
        packed_expert[packed]    = expert;
    }
}

__global__ void quantize_decode_routes_kernel(const __nv_bfloat16* input, const int* ids,
                                              const float* input_divisors, std::uint8_t* codes,
                                              std::uint8_t* scales, int assignments, int columns) {
    const int groups = columns / 16;
    const int task =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (task >= assignments * groups) { return; }
    const int assignment                      = task / groups;
    const int group                           = task - assignment * groups;
    const int expert                          = ids[assignment];
    const int token                           = assignment / kTop;
    const detail::Nvfp4QuantizedK16 quantized = detail::quantize_nvfp4_k16(
        input + static_cast<std::int64_t>(token) * columns + group * 16, input_divisors[expert]);
    auto* destination = codes + static_cast<std::int64_t>(assignment) * (columns / 2) + group * 8;
    *reinterpret_cast<uint2*>(destination) = make_uint2(quantized.codes_lo, quantized.codes_hi);
    scales[static_cast<std::int64_t>(assignment) * groups + group] = quantized.scale;
}

__global__ void gather_routes_bf16_kernel(const __nv_bfloat16* input, const int* ids,
                                          const int* local_rank, const int* offsets,
                                          __nv_bfloat16* packed, int* packed_index,
                                          int assignments) {
    const int assignment  = static_cast<int>(blockIdx.x);
    const int expert      = ids[assignment];
    const int destination = offsets[expert] + local_rank[assignment];
    const int token       = assignment / kTop;
    for (int column = static_cast<int>(threadIdx.x); column < kHidden;
         column += static_cast<int>(blockDim.x)) {
        packed[column + static_cast<std::int64_t>(kHidden) * destination] =
            input[column + static_cast<std::int64_t>(kHidden) * token];
    }
    if (threadIdx.x == 0) { packed_index[assignment] = destination; }
}

__global__ void grouped_silu_kernel(const __nv_bfloat16* gate_up, __nv_bfloat16* activation,
                                    int assignments) {
    const int task =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (task >= assignments * kIntermediate) { return; }
    const int assignment = task / kIntermediate;
    const int row        = task - assignment * kIntermediate;
    const float gate =
        __bfloat162float(gate_up[row + static_cast<std::int64_t>(2 * kIntermediate) * assignment]);
    const float up = __bfloat162float(
        gate_up[kIntermediate + row + static_cast<std::int64_t>(2 * kIntermediate) * assignment]);
    activation[task] = __float2bfloat16_rn((gate / (1.0F + expf(-gate))) * up);
}

__global__ void quantize_grouped_kernel(const __nv_bfloat16* input, const int* packed_expert,
                                        const float* input_divisors, std::uint8_t* codes,
                                        std::uint8_t* scales, int rows, int columns) {
    const int groups = columns / 16;
    const int task =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (task >= rows * groups) { return; }
    const int row   = task / groups;
    const int group = task - row * groups;
    const detail::Nvfp4QuantizedK16 quantized =
        detail::quantize_nvfp4_k16(input + static_cast<std::int64_t>(row) * columns + group * 16,
                                   input_divisors[packed_expert[row]]);
    auto* destination = codes + static_cast<std::int64_t>(row) * (columns / 2) + group * 8;
    *reinterpret_cast<uint2*>(destination) = make_uint2(quantized.codes_lo, quantized.codes_hi);
    scales[static_cast<std::int64_t>(row) * groups + group] = quantized.scale;
}

struct GroupedGateRows {
    static constexpr bool kContiguous = false;
    int expert                        = 0;
    int rows_per_branch               = 0;

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        const int branch = local_row >= rows_per_branch ? 1 : 0;
        const int logical =
            row_begin + local_row - branch * rows_per_branch + branch * kIntermediate;
        return expert * (2 * kIntermediate) + logical;
    }
};

struct GroupedDownRows {
    static constexpr bool kContiguous = false;
    int expert                        = 0;

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return expert * kHidden + row_begin + local_row;
    }
};

template <class RowPolicy>
struct GroupedWork {
    static constexpr bool kPersistent = true;
    const int* job_count;
    const int* job_experts;
    const int* job_columns;
    const int* offsets;
    const float* weight_divisors;
    const float* input_divisors;
    int output_rows;

    __device__ __forceinline__ int work_count(int rows_per_block) const {
        return *job_count * (output_rows / rows_per_block);
    }

    __device__ __forceinline__ void configure(int work, int rows_per_block, int& token_begin,
                                              int& active_tokens, int& row_begin, float& alpha,
                                              RowPolicy& rows) const {
        const int row_blocks = output_rows / rows_per_block;
        const int job        = work / row_blocks;
        const int row_block  = work - job * row_blocks;
        const int expert     = job_experts[job];
        token_begin          = offsets[expert] + job_columns[job];
        active_tokens        = offsets[expert + 1];
        row_begin            = row_block * rows_per_block;
        alpha                = 1.0F / (weight_divisors[expert] * input_divisors[expert]);
        rows.expert          = expert;
    }
};

struct DecodeRouteWork {
    static constexpr bool kPersistent = true;
    const int* ids;
    const float* weight_divisors;
    const float* input_divisors;
    int assignments;
    int output_rows;

    __device__ __forceinline__ int work_count(int rows_per_block) const {
        return assignments * (output_rows / rows_per_block);
    }

    template <class RowPolicy>
    __device__ __forceinline__ void configure(int work, int rows_per_block, int& token_begin,
                                              int& active_tokens, int& row_begin, float& alpha,
                                              RowPolicy& rows) const {
        const int row_blocks = output_rows / rows_per_block;
        const int assignment = work / row_blocks;
        const int expert     = ids[assignment];
        token_begin          = assignment;
        active_tokens        = assignment + 1;
        row_begin            = (work - assignment * row_blocks) * rows_per_block;
        alpha                = 1.0F / (weight_divisors[expert] * input_divisors[expert]);
        rows.expert          = expert;
    }
};

struct GroupedSiluOutput {
    __nv_bfloat16* data;

    __device__ __forceinline__ void store_pair_vector(int row, int token, uint4 gate_raw,
                                                      uint4 up_raw) const {
        const auto* gate  = reinterpret_cast<const __nv_bfloat162*>(&gate_raw);
        const auto* up    = reinterpret_cast<const __nv_bfloat162*>(&up_raw);
        auto* destination = reinterpret_cast<__nv_bfloat162*>(
            data + static_cast<std::int64_t>(token) * kIntermediate + row);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 g    = __bfloat1622float2(gate[pair]);
            const float2 u    = __bfloat1622float2(up[pair]);
            destination[pair] = __floats2bfloat162_rn((g.x / (1.0F + expf(-g.x))) * u.x,
                                                      (g.y / (1.0F + expf(-g.y))) * u.y);
        }
    }
};

struct GroupedSiluQuantizedOutput {
    GroupedSiluOutput activation;
    std::uint8_t* down_codes;
    std::uint8_t* down_scales;
    const int* ids;
    const float* input_divisors;

    __device__ __forceinline__ void store_pair_vector(int row, int token, uint4 gate_raw,
                                                      uint4 up_raw) const {
        activation.store_pair_vector(row, token, gate_raw, up_raw);
    }

    __device__ __forceinline__ void finish_block(int row_begin, int token_begin, int active_tokens,
                                                 int stored_rows) const {
        const int local_group = static_cast<int>(threadIdx.x);
        if (token_begin >= active_tokens || local_group >= stored_rows / 16) { return; }
        const int group                           = row_begin / 16 + local_group;
        const int expert                          = ids[token_begin];
        const detail::Nvfp4QuantizedK16 quantized = detail::quantize_nvfp4_k16(
            activation.data + static_cast<std::int64_t>(token_begin) * kIntermediate + group * 16,
            input_divisors[expert]);
        auto* destination =
            down_codes + static_cast<std::int64_t>(token_begin) * (kIntermediate / 2) + group * 8;
        *reinterpret_cast<uint2*>(destination) = make_uint2(quantized.codes_lo, quantized.codes_hi);
        down_scales[static_cast<std::int64_t>(token_begin) * (kIntermediate / 16) + group] =
            quantized.scale;
    }
};

struct GroupedOutput {
    __nv_bfloat16* data;
    int rows;

    __device__ __forceinline__ void store_vector(int row, int token, uint4 values) const {
        *reinterpret_cast<uint4*>(data + static_cast<std::int64_t>(token) * rows + row) = values;
    }
};

__global__ void reduce_grouped_kernel(const __nv_bfloat16* grouped, const int* packed_index,
                                      const float* alpha, const float* shared_alpha,
                                      __nv_bfloat16* destination, int tokens) {
    const int token = static_cast<int>(blockIdx.y);
    const int row =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (row >= kHidden) { return; }
    float sum = 0.0F;
#pragma unroll
    for (int path = 0; path < kTop; ++path) {
        const int assignment = path + kTop * token;
        const int packed     = packed_index[assignment];
        sum =
            fmaf(alpha[assignment],
                 __bfloat162float(grouped[row + static_cast<std::int64_t>(kHidden) * packed]), sum);
    }
    const std::int64_t offset = row + static_cast<std::int64_t>(kHidden) * token;
    const float shared        = __bfloat162float(
        __float2bfloat16_rn(__bfloat162float(destination[offset]) * shared_alpha[token]));
    destination[offset] = __float2bfloat16_rn(shared + sum);
}

__global__ void reduce_decode_routes_kernel(const __nv_bfloat16* grouped, const float* alpha,
                                            const float* shared_alpha, __nv_bfloat16* destination,
                                            int tokens) {
    const int token = static_cast<int>(blockIdx.y);
    const int row =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (row >= kHidden || token >= tokens) { return; }
    float sum = 0.0F;
#pragma unroll
    for (int path = 0; path < kTop; ++path) {
        const int assignment = path + kTop * token;
        sum = fmaf(alpha[assignment],
                   __bfloat162float(grouped[row + static_cast<std::int64_t>(kHidden) * assignment]),
                   sum);
    }
    const std::int64_t output = row + static_cast<std::int64_t>(kHidden) * token;
    const float shared        = __bfloat162float(
        __float2bfloat16_rn(__bfloat162float(destination[output]) * shared_alpha[token]));
    destination[output] = __float2bfloat16_rn(shared + sum);
}

__device__ __forceinline__ std::int64_t scale_offset(int rows, int columns, int expert, int row,
                                                     int group) {
    const std::int64_t expert_stride = static_cast<std::int64_t>(rows) * columns / 16;
    const int groups_per_row         = columns / 16;
    const int m_tile                 = row / 128;
    const int row_inner              = row - m_tile * 128;
    const int scale_tile             = group / 4;
    const int scale_lane             = group & 3;
    const int row_mod32              = row_inner & 31;
    const int row_quartile           = row_inner >> 5;
    return static_cast<std::int64_t>(expert) * expert_stride +
           static_cast<std::int64_t>(m_tile * (groups_per_row / 4) + scale_tile) * 512 +
           row_mod32 * 16 + row_quartile * 4 + scale_lane;
}

__device__ __forceinline__ float2 nvfp4_pair(const std::uint8_t* codes, const std::uint8_t* scales,
                                             const float* divisors, int rows, int columns,
                                             int expert, int row, int pair) {
    const std::int64_t row_index = static_cast<std::int64_t>(expert) * rows + row;
    const std::uint8_t packed    = codes[row_index * (columns / 2) + pair];
    const std::uint8_t scale     = scales[scale_offset(rows, columns, expert, row, pair / 8)];
    const float coefficient      = detail::decode_nvfp4_e4m3(scale) / divisors[expert];
    const float2 code            = detail::decode_nvfp4_e2m1x2(packed);
    return make_float2(code.x * coefficient, code.y * coefficient);
}

template <bool Bf16>
__device__ __forceinline__ float2 expert_pair(const void* data, const std::uint8_t* scales,
                                              const float* divisors, int rows, int columns,
                                              int expert, int row, int pair) {
    if constexpr (Bf16) {
        const auto* values = static_cast<const __nv_bfloat16*>(data);
        const auto* pairs  = reinterpret_cast<const __nv_bfloat162*>(values);
        return __bfloat1622float2(
            pairs[pair + static_cast<std::int64_t>(columns / 2) *
                             (row + static_cast<std::int64_t>(rows) * expert)]);
    }
    return nvfp4_pair(static_cast<const std::uint8_t*>(data), scales, divisors, rows, columns,
                      expert, row, pair);
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset != 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

template <bool Bf16>
__global__ void routed_gate_up_kernel(const __nv_bfloat16* input, const int* ids,
                                      const std::uint8_t* codes, const std::uint8_t* scales,
                                      const float* divisors, __nv_bfloat16* activations,
                                      int tokens) {
    const int token = static_cast<int>(blockIdx.z);
    const int path  = static_cast<int>(blockIdx.y);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int row   = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (row >= kIntermediate) { return; }
    const int expert        = ids[path + kTop * token];
    float gate              = 0.0F;
    float up                = 0.0F;
    const auto* input_pairs = reinterpret_cast<const __nv_bfloat162*>(input);
    for (int pair = lane; pair < kHidden / 2; pair += 32) {
        const float2 x =
            __bfloat1622float2(input_pairs[pair + static_cast<std::int64_t>(kHidden / 2) * token]);
        const float2 gate_weight = expert_pair<Bf16>(codes, scales, divisors, 2 * kIntermediate,
                                                     kHidden, expert, row, pair);
        const float2 up_weight   = expert_pair<Bf16>(codes, scales, divisors, 2 * kIntermediate,
                                                     kHidden, expert, kIntermediate + row, pair);
        gate                     = fmaf(gate_weight.x, x.x, gate);
        gate                     = fmaf(gate_weight.y, x.y, gate);
        up                       = fmaf(up_weight.x, x.x, up);
        up                       = fmaf(up_weight.y, x.y, up);
    }
    gate = warp_sum(gate);
    up   = warp_sum(up);
    if (lane == 0) {
        const float silu = gate / (1.0F + expf(-gate));
        activations[row + static_cast<std::int64_t>(kIntermediate) * (path + kTop * token)] =
            __float2bfloat16_rn(silu * up);
    }
}

template <bool Bf16>
__global__ void
routed_down_kernel(const __nv_bfloat16* activations, const int* ids, const float* alpha,
                   const std::uint8_t* codes, const std::uint8_t* scales, const float* divisors,
                   const float* shared_alpha, __nv_bfloat16* destination, int tokens) {
    const int token = static_cast<int>(blockIdx.y);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int row   = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (row >= kHidden) { return; }
    float total = 0.0F;
    for (int path = 0; path < kTop; ++path) {
        const int expert             = ids[path + kTop * token];
        float value                  = 0.0F;
        const auto* activation_pairs = reinterpret_cast<const __nv_bfloat162*>(activations);
        for (int pair = lane; pair < kIntermediate / 2; pair += 32) {
            const float2 x = __bfloat1622float2(
                activation_pairs[pair + static_cast<std::int64_t>(kIntermediate / 2) *
                                            (path + kTop * token)]);
            const float2 weight = expert_pair<Bf16>(codes, scales, divisors, kHidden, kIntermediate,
                                                    expert, row, pair);
            value               = fmaf(weight.x, x.x, value);
            value               = fmaf(weight.y, x.y, value);
        }
        value = warp_sum(value);
        if (lane == 0) { total += alpha[path + kTop * token] * value; }
    }
    if (lane == 0) {
        const std::int64_t offset = row + static_cast<std::int64_t>(kHidden) * token;
        const float shared        = __bfloat162float(
            __float2bfloat16_rn(__bfloat162float(destination[offset]) * shared_alpha[token]));
        destination[offset] = __float2bfloat16_rn(shared + total);
    }
}

void require_bank(const FlashNextExpertBank& bank, int rows, int columns, const char* label) {
    const bool nvfp4 = bank.qtype == QType::NVFP4;
    const bool bf16  = bank.qtype == QType::BF16;
    if (bank.codes == nullptr || (!nvfp4 && !bf16) ||
        (nvfp4 && (bank.scales == nullptr || bank.weight_scale_divisors == nullptr ||
                   bank.input_scale_divisors == nullptr)) ||
        bank.experts != kExperts || bank.rows != rows || bank.columns != columns) {
        throw std::invalid_argument(label);
    }
}

void run_nvfp4_decode_routes(const Tensor& input, const FlashNextMoeWeights& weights,
                             const Tensor& ids, const Tensor& alpha, const Tensor& shared_alpha,
                             Tensor& destination, Tensor& routed_activation,
                             WorkspaceArena& workspace, cudaStream_t stream, int tokens) {
    const int assignments = tokens * kTop;
    Tensor gate_codes     = workspace.alloc(DType::U8, {kHidden / 2, assignments});
    Tensor gate_scales    = workspace.alloc(DType::U8, {kHidden / 16, assignments});
    quantize_decode_routes_kernel<<<(assignments * (kHidden / 16) + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data), static_cast<const int*>(ids.data),
        weights.routed_gate_up.input_scale_divisors, static_cast<std::uint8_t*>(gate_codes.data),
        static_cast<std::uint8_t*>(gate_scales.data), assignments, kHidden);

    const detail::Nvfp4W4a4MaterializedActivation gate_input{
        static_cast<const std::uint8_t*>(gate_codes.data),
        static_cast<const std::uint8_t*>(gate_scales.data)};
    Tensor down_codes  = workspace.alloc(DType::U8, {kIntermediate / 2, assignments});
    Tensor down_scales = workspace.alloc(DType::U8, {kIntermediate / 16, assignments});
    const DecodeRouteWork gate_work{
        static_cast<const int*>(ids.data), weights.routed_gate_up.weight_scale_divisors,
        weights.routed_gate_up.input_scale_divisors, assignments, kIntermediate};
    const auto gate_output = GroupedSiluQuantizedOutput{
        GroupedSiluOutput{static_cast<__nv_bfloat16*>(routed_activation.data)},
        static_cast<std::uint8_t*>(down_codes.data), static_cast<std::uint8_t*>(down_scales.data),
        static_cast<const int*>(ids.data), weights.routed_down.input_scale_divisors};
    constexpr int kGateRowsPerBlock = DecodeGroupedGateSchedule::kBlockN / 2;
    const int gate_blocks           = assignments * (kIntermediate / kGateRowsPerBlock);
    detail::nvfp4_w4a4_mma_kernel<GroupedGateGeometry, DecodeGroupedGateSchedule,
                                  detail::Nvfp4IdentityEpilogue, GroupedSiluQuantizedOutput,
                                  GroupedGateRows, true, DecodeRouteWork>
        <<<gate_blocks, DecodeGroupedGateSchedule::kThreads, 0, stream>>>(
            gate_input, static_cast<const std::uint8_t*>(weights.routed_gate_up.codes),
            static_cast<const std::uint8_t*>(weights.routed_gate_up.scales), assignments, 1.0F,
            detail::Nvfp4IdentityEpilogue{}, gate_output, GroupedGateRows{0, kGateRowsPerBlock},
            gate_work);
    Tensor grouped_output = workspace.alloc(DType::BF16, {kHidden, assignments});
    const detail::Nvfp4W4a4MaterializedActivation down_input{
        static_cast<const std::uint8_t*>(down_codes.data),
        static_cast<const std::uint8_t*>(down_scales.data)};
    const DecodeRouteWork down_work{static_cast<const int*>(ids.data),
                                    weights.routed_down.weight_scale_divisors,
                                    weights.routed_down.input_scale_divisors, assignments, kHidden};
    constexpr int kDownRowsPerBlock = DecodeGroupedSchedule::kBlockN;
    const int down_blocks           = assignments * (kHidden / kDownRowsPerBlock);
    detail::nvfp4_w4a4_mma_kernel<GroupedDownGeometry, DecodeGroupedSchedule,
                                  detail::Nvfp4IdentityEpilogue, GroupedOutput, GroupedDownRows,
                                  false, DecodeRouteWork>
        <<<down_blocks, DecodeGroupedSchedule::kThreads, 0, stream>>>(
            down_input, static_cast<const std::uint8_t*>(weights.routed_down.codes),
            static_cast<const std::uint8_t*>(weights.routed_down.scales), assignments, 1.0F,
            detail::Nvfp4IdentityEpilogue{},
            GroupedOutput{static_cast<__nv_bfloat16*>(grouped_output.data), kHidden},
            GroupedDownRows{}, down_work);
    reduce_decode_routes_kernel<<<dim3((kHidden + 255) / 256, tokens), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(grouped_output.data),
        static_cast<const float*>(alpha.data), static_cast<const float*>(shared_alpha.data),
        static_cast<__nv_bfloat16*>(destination.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::size_t flash_next_moe_workspace_capacity_bytes(std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("Flash-Next MoE token count must be positive"); }
    const std::uint64_t bf16        = static_cast<std::uint64_t>(tokens) *
                                      (kExperts + 3 * kIntermediate + kTop * kIntermediate) *
                                      sizeof(__nv_bfloat16);
    const std::uint64_t assignments = static_cast<std::uint64_t>(tokens) * kTop;
    const std::uint64_t other       = assignments * (sizeof(std::int32_t) + sizeof(float)) +
                                      static_cast<std::uint64_t>(tokens) * sizeof(float);
    const std::uint64_t grouped =
        assignments *
            ((2 * kHidden + 2 * kIntermediate) * sizeof(__nv_bfloat16) + 4 * sizeof(std::int32_t)) +
        (2 * kExperts + 2) * sizeof(std::int32_t);
    if (bf16 + other + grouped > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("Flash-Next MoE workspace size overflow");
    }
    return static_cast<std::size_t>(bf16 + other + grouped) + 24 * 256;
}

void flash_next_moe(const Tensor& input, const FlashNextMoeWeights& weights, Tensor& destination,
                    WorkspaceArena& workspace, cudaStream_t stream, Bf16GemmContext* bf16_gemm) {
    NINFER_PERF_SCOPE(weights.routed_gate_up.qtype == QType::NVFP4 ? "moe.nvfp4" : "moe.bf16",
                       input.ne[1], 0, 0,
                       flash_next_work::moe(input.ne[1], weights.routed_gate_up.qtype == QType::NVFP4));

    const int tokens = input.ne[1];
    if (input.dtype != DType::BF16 || !input.is_contiguous() || input.ne[0] != kHidden ||
        tokens <= 0 || destination.dtype != DType::BF16 || !destination.is_contiguous() ||
        destination.ne[0] != kHidden || destination.ne[1] != tokens ||
        weights.router.n != kExperts || weights.router.k != kHidden ||
        weights.shared_gate.n != kIntermediate || weights.shared_gate.k != kHidden ||
        weights.shared_up.n != kIntermediate || weights.shared_up.k != kHidden ||
        weights.shared_down.n != kHidden || weights.shared_down.k != kIntermediate ||
        weights.shared_scale.n != 1 || weights.shared_scale.k != kHidden) {
        throw std::invalid_argument("flash_next_moe: invalid exact geometry");
    }
    require_bank(weights.routed_gate_up, 2 * kIntermediate, kHidden,
                 "flash_next_moe: invalid routed gate/up bank");
    require_bank(weights.routed_down, kHidden, kIntermediate,
                 "flash_next_moe: invalid routed down bank");
    auto scope    = workspace.scope();
    Tensor scores = workspace.alloc(DType::BF16, {kExperts, tokens});
    Tensor shared_activation = workspace.alloc(DType::BF16, {kIntermediate, tokens});
    // Compact FP8 batches compute router scores and the shared SwiGLU in one grid.
    const bool fp8_entry = tokens <= detail::flash_next::kFp8MoeEntryMaxTokens &&
                           weights.router.qtype == QType::FP8_E4M3FN_ROW_BF16 &&
                           weights.shared_gate.qtype == QType::FP8_E4M3FN_ROW_BF16 &&
                           weights.shared_up.qtype == QType::FP8_E4M3FN_ROW_BF16;
    if (fp8_entry) {
        detail::flash_next::launch_fp8_moe_entry_decode(input, weights.router, weights.shared_gate,
                                                        weights.shared_up, scores,
                                                        shared_activation, stream);
    } else {
        linear(input, weights.router, scores, stream, bf16_gemm);
    }
    Tensor ids          = workspace.alloc(DType::I32, {kTop, tokens});
    Tensor alpha        = workspace.alloc(DType::FP32, {kTop, tokens});
    Tensor shared_alpha = workspace.alloc(DType::FP32, {tokens});
    if (tokens == 1 && fp8_entry && weights.shared_down.qtype == QType::BF16) {
        // The entry kernel produced the shared activation; route and shared Down run together.
        route_shared_down_decode_kernel<<<1 + kSharedDownBlocks, kRouteThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(scores.data),
            static_cast<const __nv_bfloat16*>(input.data),
            static_cast<const __nv_bfloat16*>(weights.shared_scale.qdata),
            static_cast<int*>(ids.data), static_cast<float*>(alpha.data),
            static_cast<float*>(shared_alpha.data),
            static_cast<const __nv_bfloat16*>(shared_activation.data),
            static_cast<const __nv_bfloat16*>(weights.shared_down.qdata),
            static_cast<__nv_bfloat16*>(destination.data));
        CUDA_CHECK(cudaGetLastError());
    } else {
        route_kernel<<<tokens, kRouteThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(scores.data),
            static_cast<const __nv_bfloat16*>(input.data),
            static_cast<const __nv_bfloat16*>(weights.shared_scale.qdata),
            static_cast<int*>(ids.data), static_cast<float*>(alpha.data),
            static_cast<float*>(shared_alpha.data), tokens);
        if (fp8_entry) {
            // Already produced by the MoE entry kernel.
        } else if (tokens == 1 && weights.shared_gate.qtype == QType::BF16 &&
                   weights.shared_up.qtype == QType::BF16) {
            detail::flash_next::launch_bf16_shared_swiglu_decode(
                input, weights.shared_gate, weights.shared_up, shared_activation, stream);
        } else {
            Tensor shared_gate = workspace.alloc(DType::BF16, {kIntermediate, tokens});
            Tensor shared_up   = workspace.alloc(DType::BF16, {kIntermediate, tokens});
            linear(input, weights.shared_gate, shared_gate, stream, bf16_gemm);
            linear(input, weights.shared_up, shared_up, stream, bf16_gemm);
            silu_mul(shared_gate, shared_up, shared_activation, stream);
        }
        linear(shared_activation, weights.shared_down, destination, stream, bf16_gemm);
    }
    Tensor routed_activation = workspace.alloc(DType::BF16, {kIntermediate, kTop, tokens});
    if (weights.routed_gate_up.qtype == QType::NVFP4 && tokens >= kGroupedFirstToken) {
        if (tokens <= kDecodeGroupedTokenTile) {
            run_nvfp4_decode_routes(input, weights, ids, alpha, shared_alpha, destination,
                                    routed_activation, workspace, stream, tokens);
            return;
        }
        const int assignments = tokens * kTop;
        Tensor local_rank     = workspace.alloc(DType::I32, {assignments});
        Tensor counts         = workspace.alloc(DType::I32, {kExperts});
        Tensor offsets        = workspace.alloc(DType::I32, {kExperts + 1});
        Tensor packed_index   = workspace.alloc(DType::I32, {assignments});
        Tensor packed_expert  = workspace.alloc(DType::I32, {assignments});
        Tensor job_experts    = workspace.alloc(DType::I32, {assignments});
        Tensor job_columns    = workspace.alloc(DType::I32, {assignments});
        Tensor job_count      = workspace.alloc(DType::I32, {1});
        CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), stream));
        CUDA_CHECK(cudaMemsetAsync(job_count.data, 0, job_count.bytes(), stream));
        count_routes_kernel<<<kExperts, 256, 0, stream>>>(static_cast<const int*>(ids.data),
                                                          static_cast<int*>(local_rank.data),
                                                          static_cast<int*>(counts.data), tokens);
        scan_routes_kernel<<<1, kExperts, 0, stream>>>(static_cast<const int*>(counts.data),
                                                       static_cast<int*>(offsets.data));
        const bool decode_grouped = tokens <= kDecodeGroupedTokenTile;
        const bool large_grouped  = tokens >= 4096;
        const int grouped_token_tile =
            decode_grouped ? kDecodeGroupedTokenTile
                           : (large_grouped ? kLargeGroupedTokenTile : kGroupedTokenTile);
        make_route_jobs_kernel<<<1, kExperts, 0, stream>>>(
            static_cast<const int*>(counts.data), static_cast<int*>(job_experts.data),
            static_cast<int*>(job_columns.data), static_cast<int*>(job_count.data),
            grouped_token_tile);

        Tensor gate_codes  = workspace.alloc(DType::U8, {kHidden / 2, assignments});
        Tensor gate_scales = workspace.alloc(DType::U8, {kHidden / 16, assignments});
        gather_quantize_routes_kernel<<<(assignments * (kHidden / 16) + 255) / 256, 256, 0,
                                        stream>>>(
            static_cast<const __nv_bfloat16*>(input.data), static_cast<const int*>(ids.data),
            static_cast<const int*>(local_rank.data), static_cast<const int*>(offsets.data),
            weights.routed_gate_up.input_scale_divisors,
            static_cast<std::uint8_t*>(gate_codes.data),
            static_cast<std::uint8_t*>(gate_scales.data), static_cast<int*>(packed_index.data),
            static_cast<int*>(packed_expert.data), assignments, kHidden);
        detail::Nvfp4W4a4MaterializedActivation gate_input{
            static_cast<const std::uint8_t*>(gate_codes.data),
            static_cast<const std::uint8_t*>(gate_scales.data)};
        const GroupedWork<GroupedGateRows> gate_work{static_cast<const int*>(job_count.data),
                                                     static_cast<const int*>(job_experts.data),
                                                     static_cast<const int*>(job_columns.data),
                                                     static_cast<const int*>(offsets.data),
                                                     weights.routed_gate_up.weight_scale_divisors,
                                                     weights.routed_gate_up.input_scale_divisors,
                                                     kIntermediate};
        if (decode_grouped) {
            constexpr int kGateRowsPerBlock = DecodeGroupedSchedule::kBlockN / 2;
            const int gate_blocks =
                std::min(510, assignments * (kIntermediate / kGateRowsPerBlock));
            detail::nvfp4_w4a4_mma_kernel<GroupedGateGeometry, DecodeGroupedSchedule,
                                          detail::Nvfp4IdentityEpilogue, GroupedSiluOutput,
                                          GroupedGateRows, true, GroupedWork<GroupedGateRows>>
                <<<gate_blocks, DecodeGroupedSchedule::kThreads, 0, stream>>>(
                    gate_input, static_cast<const std::uint8_t*>(weights.routed_gate_up.codes),
                    static_cast<const std::uint8_t*>(weights.routed_gate_up.scales), assignments,
                    1.0F, detail::Nvfp4IdentityEpilogue{},
                    GroupedSiluOutput{static_cast<__nv_bfloat16*>(routed_activation.data)},
                    GroupedGateRows{0, kGateRowsPerBlock}, gate_work);
        } else if (large_grouped) {
            detail::nvfp4_w4a4_mma_kernel<GroupedGateGeometry, LargeGroupedSchedule,
                                          detail::Nvfp4IdentityEpilogue, GroupedSiluOutput,
                                          GroupedGateRows, true, GroupedWork<GroupedGateRows>>
                <<<510, LargeGroupedSchedule::kThreads, 0, stream>>>(
                    gate_input, static_cast<const std::uint8_t*>(weights.routed_gate_up.codes),
                    static_cast<const std::uint8_t*>(weights.routed_gate_up.scales), assignments,
                    1.0F, detail::Nvfp4IdentityEpilogue{},
                    GroupedSiluOutput{static_cast<__nv_bfloat16*>(routed_activation.data)},
                    GroupedGateRows{0, LargeGroupedSchedule::kBlockN / 2}, gate_work);
        } else {
            detail::nvfp4_w4a4_mma_kernel<GroupedGateGeometry, GroupedGateSchedule,
                                          detail::Nvfp4IdentityEpilogue, GroupedSiluOutput,
                                          GroupedGateRows, true, GroupedWork<GroupedGateRows>>
                <<<510, GroupedGateSchedule::kThreads, 0, stream>>>(
                    gate_input, static_cast<const std::uint8_t*>(weights.routed_gate_up.codes),
                    static_cast<const std::uint8_t*>(weights.routed_gate_up.scales), assignments,
                    1.0F, detail::Nvfp4IdentityEpilogue{},
                    GroupedSiluOutput{static_cast<__nv_bfloat16*>(routed_activation.data)},
                    GroupedGateRows{0, GroupedGateSchedule::kBlockN / 2}, gate_work);
        }
        Tensor down_codes  = workspace.alloc(DType::U8, {kIntermediate / 2, assignments});
        Tensor down_scales = workspace.alloc(DType::U8, {kIntermediate / 16, assignments});
        quantize_grouped_kernel<<<(assignments * (kIntermediate / 16) + 255) / 256, 256, 0,
                                  stream>>>(
            static_cast<const __nv_bfloat16*>(routed_activation.data),
            static_cast<const int*>(packed_expert.data), weights.routed_down.input_scale_divisors,
            static_cast<std::uint8_t*>(down_codes.data),
            static_cast<std::uint8_t*>(down_scales.data), assignments, kIntermediate);
        Tensor grouped_output = workspace.alloc(DType::BF16, {kHidden, assignments});
        detail::Nvfp4W4a4MaterializedActivation down_input{
            static_cast<const std::uint8_t*>(down_codes.data),
            static_cast<const std::uint8_t*>(down_scales.data)};
        const GroupedWork<GroupedDownRows> down_work{static_cast<const int*>(job_count.data),
                                                     static_cast<const int*>(job_experts.data),
                                                     static_cast<const int*>(job_columns.data),
                                                     static_cast<const int*>(offsets.data),
                                                     weights.routed_down.weight_scale_divisors,
                                                     weights.routed_down.input_scale_divisors,
                                                     kHidden};
        if (decode_grouped) {
            constexpr int kDownRowsPerBlock = DecodeGroupedSchedule::kBlockN;
            const int down_blocks = std::min(510, assignments * (kHidden / kDownRowsPerBlock));
            detail::nvfp4_w4a4_mma_kernel<GroupedDownGeometry, DecodeGroupedSchedule,
                                          detail::Nvfp4IdentityEpilogue, GroupedOutput,
                                          GroupedDownRows, false, GroupedWork<GroupedDownRows>>
                <<<down_blocks, DecodeGroupedSchedule::kThreads, 0, stream>>>(
                    down_input, static_cast<const std::uint8_t*>(weights.routed_down.codes),
                    static_cast<const std::uint8_t*>(weights.routed_down.scales), assignments, 1.0F,
                    detail::Nvfp4IdentityEpilogue{},
                    GroupedOutput{static_cast<__nv_bfloat16*>(grouped_output.data), kHidden},
                    GroupedDownRows{}, down_work);
        } else if (large_grouped) {
            detail::nvfp4_w4a4_mma_kernel<GroupedDownGeometry, LargeGroupedSchedule,
                                          detail::Nvfp4IdentityEpilogue, GroupedOutput,
                                          GroupedDownRows, false, GroupedWork<GroupedDownRows>>
                <<<510, LargeGroupedSchedule::kThreads, 0, stream>>>(
                    down_input, static_cast<const std::uint8_t*>(weights.routed_down.codes),
                    static_cast<const std::uint8_t*>(weights.routed_down.scales), assignments, 1.0F,
                    detail::Nvfp4IdentityEpilogue{},
                    GroupedOutput{static_cast<__nv_bfloat16*>(grouped_output.data), kHidden},
                    GroupedDownRows{}, down_work);
        } else {
            detail::nvfp4_w4a4_mma_kernel<GroupedDownGeometry, GroupedDownSchedule,
                                          detail::Nvfp4IdentityEpilogue, GroupedOutput,
                                          GroupedDownRows, false, GroupedWork<GroupedDownRows>>
                <<<510, GroupedDownSchedule::kThreads, 0, stream>>>(
                    down_input, static_cast<const std::uint8_t*>(weights.routed_down.codes),
                    static_cast<const std::uint8_t*>(weights.routed_down.scales), assignments, 1.0F,
                    detail::Nvfp4IdentityEpilogue{},
                    GroupedOutput{static_cast<__nv_bfloat16*>(grouped_output.data), kHidden},
                    GroupedDownRows{}, down_work);
        }
        reduce_grouped_kernel<<<dim3((kHidden + 255) / 256, tokens), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(grouped_output.data),
            static_cast<const int*>(packed_index.data), static_cast<const float*>(alpha.data),
            static_cast<const float*>(shared_alpha.data),
            static_cast<__nv_bfloat16*>(destination.data), tokens);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (weights.routed_gate_up.qtype == QType::BF16 && tokens > 16) {
        const int assignments = tokens * kTop;
        Tensor local_rank     = workspace.alloc(DType::I32, {assignments});
        Tensor counts         = workspace.alloc(DType::I32, {kExperts});
        Tensor offsets        = workspace.alloc(DType::I32, {kExperts + 1});
        Tensor packed_index   = workspace.alloc(DType::I32, {assignments});
        Tensor job_experts    = workspace.alloc(DType::I32, {assignments});
        Tensor job_columns    = workspace.alloc(DType::I32, {assignments});
        Tensor job_count      = workspace.alloc(DType::I32, {1});
        CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), stream));
        CUDA_CHECK(cudaMemsetAsync(job_count.data, 0, job_count.bytes(), stream));
        count_routes_kernel<<<kExperts, 256, 0, stream>>>(static_cast<const int*>(ids.data),
                                                          static_cast<int*>(local_rank.data),
                                                          static_cast<int*>(counts.data), tokens);
        scan_routes_kernel<<<1, kExperts, 0, stream>>>(static_cast<const int*>(counts.data),
                                                       static_cast<int*>(offsets.data));
        make_route_jobs_kernel<<<1, kExperts, 0, stream>>>(
            static_cast<const int*>(counts.data), static_cast<int*>(job_experts.data),
            static_cast<int*>(job_columns.data), static_cast<int*>(job_count.data),
            Bf16GroupedSchedule::kBlockCols);
        Tensor packed_input = workspace.alloc(DType::BF16, {kHidden, assignments});
        gather_routes_bf16_kernel<<<assignments, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(input.data), static_cast<const int*>(ids.data),
            static_cast<const int*>(local_rank.data), static_cast<const int*>(offsets.data),
            static_cast<__nv_bfloat16*>(packed_input.data), static_cast<int*>(packed_index.data),
            assignments);
        Tensor gate_up = workspace.alloc(DType::BF16, {2 * kIntermediate, assignments});
        detail::bf16_grouped_gemm_mma_kernel<Bf16GroupedGateGeometry, Bf16GroupedSchedule>
            <<<510, Bf16GroupedSchedule::kThreads, Bf16GroupedSchedule::kSharedBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(packed_input.data),
                static_cast<const __nv_bfloat16*>(weights.routed_gate_up.codes),
                static_cast<const int*>(offsets.data), static_cast<const int*>(job_experts.data),
                static_cast<const int*>(job_columns.data), static_cast<const int*>(job_count.data),
                static_cast<__nv_bfloat16*>(gate_up.data));
        grouped_silu_kernel<<<(assignments * kIntermediate + 255) / 256, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate_up.data),
            static_cast<__nv_bfloat16*>(routed_activation.data), assignments);
        Tensor grouped_output = workspace.alloc(DType::BF16, {kHidden, assignments});
        detail::bf16_grouped_gemm_mma_kernel<Bf16GroupedDownGeometry, Bf16GroupedSchedule>
            <<<510, Bf16GroupedSchedule::kThreads, Bf16GroupedSchedule::kSharedBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(routed_activation.data),
                static_cast<const __nv_bfloat16*>(weights.routed_down.codes),
                static_cast<const int*>(offsets.data), static_cast<const int*>(job_experts.data),
                static_cast<const int*>(job_columns.data), static_cast<const int*>(job_count.data),
                static_cast<__nv_bfloat16*>(grouped_output.data));
        reduce_grouped_kernel<<<dim3((kHidden + 255) / 256, tokens), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(grouped_output.data),
            static_cast<const int*>(packed_index.data), static_cast<const float*>(alpha.data),
            static_cast<const float*>(shared_alpha.data),
            static_cast<__nv_bfloat16*>(destination.data), tokens);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const dim3 gate_grid((kIntermediate + kWarps - 1) / kWarps, kTop, tokens);
    const dim3 down_grid((kHidden + kWarps - 1) / kWarps, tokens);
    if (weights.routed_gate_up.qtype == QType::BF16) {
        routed_gate_up_kernel<true><<<gate_grid, kWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(input.data), static_cast<const int*>(ids.data),
            static_cast<const std::uint8_t*>(weights.routed_gate_up.codes), nullptr, nullptr,
            static_cast<__nv_bfloat16*>(routed_activation.data), tokens);
        routed_down_kernel<true><<<down_grid, kWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(routed_activation.data),
            static_cast<const int*>(ids.data), static_cast<const float*>(alpha.data),
            static_cast<const std::uint8_t*>(weights.routed_down.codes), nullptr, nullptr,
            static_cast<const float*>(shared_alpha.data),
            static_cast<__nv_bfloat16*>(destination.data), tokens);
    } else {
        routed_gate_up_kernel<false><<<gate_grid, kWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(input.data), static_cast<const int*>(ids.data),
            static_cast<const std::uint8_t*>(weights.routed_gate_up.codes),
            static_cast<const std::uint8_t*>(weights.routed_gate_up.scales),
            weights.routed_gate_up.weight_scale_divisors,
            static_cast<__nv_bfloat16*>(routed_activation.data), tokens);
        routed_down_kernel<false><<<down_grid, kWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(routed_activation.data),
            static_cast<const int*>(ids.data), static_cast<const float*>(alpha.data),
            static_cast<const std::uint8_t*>(weights.routed_down.codes),
            static_cast<const std::uint8_t*>(weights.routed_down.scales),
            weights.routed_down.weight_scale_divisors, static_cast<const float*>(shared_alpha.data),
            static_cast<__nv_bfloat16*>(destination.data), tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
