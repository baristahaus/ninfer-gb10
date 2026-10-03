#include "ninfer/ops/flash_next_moe.h"

#include "ops/flash_next_work.h"

#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/common/device_info.h"
#include "ops/linear/bf16/flash_next/bf16_config.h"
#include "ops/linear/bf16/flash_next/bf16_gemm_mma.cuh"
#include "ops/linear/bf16/flash_next/bf16_launch.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/sparse_moe/flash_next/flash_next_nvfp4_w4a4.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <variant>

#ifdef NINFER_PERFORMANCE_TRACE
#    include <chrono>
#    include <cstdio>
#    include <cstdlib>
#    include <cstring>
#    include <string>
#    include <thread>
#endif

namespace ninfer::ops {
namespace {

constexpr int kHidden                 = 2560;
constexpr int kExperts                = 512;
constexpr int kTop                    = 10;
constexpr int kIntermediate           = 640;
constexpr int kWarps                  = 8;
constexpr int kGroupedTokenTile       = 32;
constexpr int kLargeGroupedTokenTile  = 128;
constexpr int kDecodeGroupedTokenTile = 16;
// Below this many rows the per-assignment decode route is used. The per-assignment route streams
// a selected expert once per (row, path); the expert-grouped route streams each distinct expert
// once for all of its rows but adds route counting, packing and a larger graph. On GB10 decode,
// where a round's rows come from independent requests and share few experts, grouping measured
// -1.8% end to end at 2 rows, flat at 4 and +3.1% at 8 (2026-10-02).
constexpr int kGroupedDecodeMinTokens = 8;
constexpr int kPrefillBlocksPerSm     = 3;

using GroupedGateGeometry = detail::Nvfp4Geometry<2 * kIntermediate, kHidden>;
using GroupedDownGeometry = detail::Nvfp4Geometry<kHidden, kIntermediate>;
using GroupedGateSchedule = detail::flash_next::Nvfp4W4a4MmaSchedule<kGroupedTokenTile, 256, 128, 2, 4, 2, 1>;
using GroupedDownSchedule = detail::flash_next::Nvfp4W4a4MmaSchedule<kGroupedTokenTile, 256, 128, 2, 4, 2, 1>;
using LargeGroupedSchedule =
    detail::flash_next::Nvfp4W4a4MmaSchedule<kLargeGroupedTokenTile, 256, 128, 4, 4, 3, 1>;
// Decode (both decode routes): 64-row items. Gate/up reads 256 bytes of every row per stage
// (BK512) and down keeps four 64-byte stages in flight. On GB10 at 8 rows these stream at 96-97%
// (gate/up) and 95-97% (down) of a plain read of the same expert bytes, against 85-87% and
// 90-93% for the former BN128 BK128 S2 (bench --probe, 2026-10-02). The k64 accumulation order of
// every output is unchanged, so results are bitwise those of the former schedules.
using DecodeGateSchedule =
    detail::flash_next::Nvfp4W4a4MmaSchedule<kDecodeGroupedTokenTile, 64, 512, 1, 8, 2, 1>;
using DecodeDownSchedule =
    detail::flash_next::Nvfp4W4a4MmaSchedule<kDecodeGroupedTokenTile, 64, 128, 1, 8, 4, 1>;
using Bf16GroupedSchedule =
    detail::flash_next::Bf16MmaSchedule<64, 64, 64, 32, 32, 3, 2, Cache::cg, Cache::cg,
                                        detail::flash_next::Bf16MmaFragmentPipeline::PingPong,
                                        detail::flash_next::Bf16MmaRaster::TokenFast>;
using Bf16GroupedGateGeometry = detail::flash_next::Bf16GemvGeometry<2 * kIntermediate, kHidden>;
using Bf16GroupedDownGeometry = detail::flash_next::Bf16GemvGeometry<kHidden, kIntermediate>;

// Routing, one warp per token, 32 tokens per CTA: the shared-expert gate, the top-10 experts and
// their normalized weights. The results are bitwise those of the 256-thread-per-token kernel this
// replaced: the gate's dot product is evaluated as the same eight 32-thread fmaf chains and
// shuffle trees, and the top-10 order (score descending, then expert ascending) is unique.
//
// With grouping (`offsets` non-null; one CTA, tokens <= kFusedRouteTokens) the CTA also packs the
// assignments by expert in token order and emits one grouped job per selected expert, as
// count_routes/scan_routes/make_route_jobs do at any token tile >= tokens.
constexpr int kRouteWarps       = 32;
constexpr int kFusedRouteTokens = kRouteWarps;

struct RouteOutputs {
    int* ids;
    float* alpha;
    float* shared_alpha;
    int* offsets;       // [kExperts + 1], null without grouping
    int* packed_index;  // [assignments]: assignment -> packed row
    int* packed_expert; // [assignments]: packed row -> expert
    int* job_experts;   // [assignments]
    int* job_columns;   // [assignments]
    int* job_count;     // [1]
};

__device__ __forceinline__ bool route_better(float lhs, int lhs_id, float rhs, int rhs_id) {
    return lhs > rhs || (lhs == rhs && lhs_id < rhs_id);
}

__global__ void __launch_bounds__(kRouteWarps * 32)
    route_kernel(const __nv_bfloat16* scores, const __nv_bfloat16* input,
                 const __nv_bfloat16* shared_scale_weight, RouteOutputs out, int tokens) {
    __shared__ int group_ids[kFusedRouteTokens * kTop];
    __shared__ int group_offsets[kExperts];
    __shared__ int scan_totals[kRouteWarps];
    constexpr unsigned kFull = 0xffffffffU;
    const int lane           = static_cast<int>(threadIdx.x) & 31;
    const int warp           = static_cast<int>(threadIdx.x) >> 5;
    const int token          = static_cast<int>(blockIdx.x) * kRouteWarps + warp;
    if (token < tokens) {
        // Shared gate: virtual thread v*32+lane of the former 256-thread block accumulates
        // k = v*32+lane, +256, ... in order; each virtual warp reduces by shuffle-down and the
        // eight warp sums reduce the same way from lanes 0..7.
        // The loads of a virtual warp are issued before its chain of fmaf.
        constexpr int kSharedSteps = kHidden / 256;
        static_assert(kHidden % 256 == 0);
        const __nv_bfloat16* row = input + static_cast<std::int64_t>(kHidden) * token;
        float gathered           = 0.0F;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            __nv_bfloat16 inputs[kSharedSteps];
            __nv_bfloat16 weights[kSharedSteps];
#pragma unroll
            for (int step = 0; step < kSharedSteps; ++step) {
                inputs[step]  = row[v * 32 + lane + step * 256];
                weights[step] = shared_scale_weight[v * 32 + lane + step * 256];
            }
            float value = 0.0F;
#pragma unroll
            for (int step = 0; step < kSharedSteps; ++step) {
                value = fmaf(__bfloat162float(inputs[step]), __bfloat162float(weights[step]),
                             value);
            }
            for (int offset = 16; offset != 0; offset >>= 1) {
                value += __shfl_down_sync(kFull, value, offset);
            }
            const float sum = __shfl_sync(kFull, value, 0);
            if (lane == v) { gathered = sum; }
        }
        for (int offset = 16; offset != 0; offset >>= 1) {
            gathered += __shfl_down_sync(kFull, gathered, offset);
        }
        if (lane == 0) { out.shared_alpha[token] = 1.0F / (1.0F + expf(-gathered)); }

        // Top-10: lane holds experts lane + 32 j.
        float values[kExperts / 32];
#pragma unroll
        for (int j = 0; j < kExperts / 32; ++j) {
            values[j] = __bfloat162float(
                scores[lane + 32 * j + static_cast<std::int64_t>(kExperts) * token]);
        }
        unsigned taken = 0;
        float top_values[kTop];
        int top_ids[kTop];
#pragma unroll
        for (int rank = 0; rank < kTop; ++rank) {
            float best  = -__int_as_float(0x7f800000);
            int best_id = kExperts;
#pragma unroll
            for (int j = 0; j < kExperts / 32; ++j) {
                if (((taken >> j) & 1U) == 0U &&
                    route_better(values[j], lane + 32 * j, best, best_id)) {
                    best    = values[j];
                    best_id = lane + 32 * j;
                }
            }
            for (int offset = 16; offset != 0; offset >>= 1) {
                const float other  = __shfl_xor_sync(kFull, best, offset);
                const int other_id = __shfl_xor_sync(kFull, best_id, offset);
                if (route_better(other, other_id, best, best_id)) {
                    best    = other;
                    best_id = other_id;
                }
            }
            top_values[rank] = best;
            top_ids[rank]    = best_id;
            if ((best_id & 31) == lane) { taken |= 1U << (best_id >> 5); }
        }
        if (lane == 0) {
            const float maximum = top_values[0];
            float denominator   = 0.0F;
#pragma unroll
            for (int rank = 0; rank < kTop; ++rank) {
                denominator += expf(top_values[rank] - maximum);
            }
#pragma unroll
            for (int rank = 0; rank < kTop; ++rank) {
                const int offset  = rank + kTop * token;
                out.ids[offset]   = top_ids[rank];
                out.alpha[offset] = expf(top_values[rank] - maximum) / denominator;
            }
        }
        if (out.offsets != nullptr) {
#pragma unroll
            for (int rank = 0; rank < kTop; ++rank) {
                if (lane == rank) { group_ids[rank + kTop * token] = top_ids[rank]; }
            }
        }
    }
    if (out.offsets == nullptr) { return; }

    // Grouping (one CTA): per-expert counts and one job per selected expert, exclusive-scanned
    // together as count | job << 16 (both totals are at most kFusedRouteTokens * kTop).
    __syncthreads();
    const int assignments = tokens * kTop;
    const int tid         = static_cast<int>(threadIdx.x);
    int packed_count      = 0;
    if (tid < kExperts) {
        int count = 0;
        for (int a = 0; a < assignments; ++a) { count += group_ids[a] == tid ? 1 : 0; }
        packed_count = count | (count > 0 ? 1 << 16 : 0);
    }
    int inclusive = packed_count;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int add = __shfl_up_sync(kFull, inclusive, offset);
        if (lane >= offset) { inclusive += add; }
    }
    if (lane == 31) { scan_totals[warp] = inclusive; }
    __syncthreads();
    if (warp == 0) {
        int total = scan_totals[lane];
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int add = __shfl_up_sync(kFull, total, offset);
            if (lane >= offset) { total += add; }
        }
        scan_totals[lane] = total;
    }
    __syncthreads();
    if (tid < kExperts) {
        const int exclusive = (warp == 0 ? 0 : scan_totals[warp - 1]) + inclusive - packed_count;
        const int count     = packed_count & 0xffff;
        const int offset    = exclusive & 0xffff;
        const int job       = exclusive >> 16;
        group_offsets[tid]  = offset;
        out.offsets[tid]    = offset;
        if (count > 0) {
            out.job_experts[job] = tid;
            out.job_columns[job] = 0;
        }
        if (tid == kExperts - 1) {
            out.offsets[kExperts] = offset + count;
            out.job_count[0]      = job + (count > 0 ? 1 : 0);
        }
    }
    __syncthreads();
    if (tid < assignments) {
        // Rank among the earlier tokens that chose the same expert (each token's ten are distinct).
        const int expert = group_ids[tid];
        const int before = (tid / kTop) * kTop;
        int rank         = 0;
        for (int a = 0; a < before; ++a) { rank += group_ids[a] == expert ? 1 : 0; }
        const int packed          = group_offsets[expert] + rank;
        out.packed_index[tid]     = packed;
        out.packed_expert[packed] = expert;
    }
}

#ifdef NINFER_PERFORMANCE_TRACE
// Trace builds only: how many distinct experts each grouped decode/verify call selects, the
// quantity its routed weight bytes scale with (the work envelope cannot know it). With
// NINFER_MOE_ROUTE_STATS=<path>, a one-thread kernel after the routing tallies (tokens, distinct
// experts) into host-mapped memory, and a host thread rewrites <path> every two seconds as
// "tokens distinct calls" lines. Increments are not atomic; calls on concurrent streams may
// rarely lose a count.
struct RouteStats {
    unsigned long long calls[kFusedRouteTokens + 1][kFusedRouteTokens * kTop + 1];
};

__global__ void route_stats_kernel(const int* job_count, int tokens, RouteStats* stats) {
    volatile unsigned long long* slot = &stats->calls[tokens][*job_count];
    *slot                             = *slot + 1;
}

RouteStats* route_stats() {
    static RouteStats* const device_stats = []() -> RouteStats* {
        const char* path = std::getenv("NINFER_MOE_ROUTE_STATS");
        if (path == nullptr || *path == '\0') return nullptr;
        // The first call may come during graph capture; the mapped allocation is not a stream op.
        cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
        CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        void* host = nullptr;
        CUDA_CHECK(cudaHostAlloc(&host, sizeof(RouteStats), cudaHostAllocMapped));
        std::memset(host, 0, sizeof(RouteStats));
        void* device = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&device, host, 0));
        CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        std::thread([output = std::string(path), host] {
            const auto* stats = static_cast<const volatile RouteStats*>(host);
            const std::string partial = output + ".tmp";
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
                std::FILE* file = std::fopen(partial.c_str(), "w");
                if (file == nullptr) continue;
                std::fprintf(file, "tokens distinct calls\n");
                for (int tokens = 0; tokens <= kFusedRouteTokens; ++tokens) {
                    for (int distinct = 0; distinct <= kFusedRouteTokens * kTop; ++distinct) {
                        const unsigned long long calls = stats->calls[tokens][distinct];
                        if (calls != 0)
                            std::fprintf(file, "%d %d %llu\n", tokens, distinct, calls);
                    }
                }
                std::fclose(file);
                std::rename(partial.c_str(), output.c_str());
            }
        }).detach();
        return static_cast<RouteStats*>(device);
    }();
    return device_stats;
}
#endif

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

// Packed rows for the multi-kernel grouping: assignment -> offsets[expert] + rank and back.
__global__ void pack_routes_kernel(const int* ids, const int* local_rank, const int* offsets,
                                   int* packed_index, int* packed_expert, int assignments) {
    const int assignment =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (assignment >= assignments) { return; }
    const int expert         = ids[assignment];
    const int packed         = offsets[expert] + local_rank[assignment];
    packed_index[assignment] = packed;
    packed_expert[packed]    = expert;
}

__global__ void gather_quantize_routes_kernel(const __nv_bfloat16* input, const int* ids,
                                              const int* packed_index, const float* input_divisors,
                                              std::uint8_t* codes, std::uint8_t* scales,
                                              int assignments, int columns) {
    const int groups = columns / 16;
    const int task =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (task >= assignments * groups) { return; }
    const int assignment                      = task / groups;
    const int group                           = task - assignment * groups;
    const int expert                          = ids[assignment];
    const int packed                          = packed_index[assignment];
    const int token                           = assignment / kTop;
    const detail::Nvfp4QuantizedK16 quantized = detail::quantize_nvfp4_k16(
        input + static_cast<std::int64_t>(token) * columns + group * 16, input_divisors[expert]);
    auto* destination = codes + static_cast<std::int64_t>(packed) * (columns / 2) + group * 8;
    *reinterpret_cast<uint2*>(destination) = make_uint2(quantized.codes_lo, quantized.codes_hi);
    scales[static_cast<std::int64_t>(packed) * groups + group] = quantized.scale;
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

    const detail::flash_next::Nvfp4W4a4MaterializedActivation gate_input{
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
    {
        constexpr int kGateRowsPerBlock = DecodeGateSchedule::kBlockN / 2;
        const int gate_blocks           = assignments * (kIntermediate / kGateRowsPerBlock);
        detail::flash_next::nvfp4_w4a4_mma_kernel<
            GroupedGateGeometry, DecodeGateSchedule, detail::flash_next::Nvfp4IdentityEpilogue,
            GroupedSiluQuantizedOutput, GroupedGateRows, true, DecodeRouteWork>
            <<<gate_blocks, DecodeGateSchedule::kThreads, 0, stream>>>(
                gate_input, static_cast<const std::uint8_t*>(weights.routed_gate_up.codes),
                static_cast<const std::uint8_t*>(weights.routed_gate_up.scales), assignments, 1.0F,
                detail::flash_next::Nvfp4IdentityEpilogue{}, gate_output,
                GroupedGateRows{0, kGateRowsPerBlock}, gate_work);
    }
    Tensor grouped_output = workspace.alloc(DType::BF16, {kHidden, assignments});
    const detail::flash_next::Nvfp4W4a4MaterializedActivation down_input{
        static_cast<const std::uint8_t*>(down_codes.data),
        static_cast<const std::uint8_t*>(down_scales.data)};
    const DecodeRouteWork down_work{static_cast<const int*>(ids.data),
                                    weights.routed_down.weight_scale_divisors,
                                    weights.routed_down.input_scale_divisors, assignments, kHidden};
    constexpr int kDownRowsPerBlock = DecodeDownSchedule::kBlockN;
    const int down_blocks           = assignments * (kHidden / kDownRowsPerBlock);
    detail::flash_next::nvfp4_w4a4_mma_kernel<
        GroupedDownGeometry, DecodeDownSchedule, detail::flash_next::Nvfp4IdentityEpilogue,
        GroupedOutput, GroupedDownRows, false, DecodeRouteWork>
        <<<down_blocks, DecodeDownSchedule::kThreads, 0, stream>>>(
            down_input, static_cast<const std::uint8_t*>(weights.routed_down.codes),
            static_cast<const std::uint8_t*>(weights.routed_down.scales), assignments, 1.0F,
            detail::flash_next::Nvfp4IdentityEpilogue{},
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
    const auto* shared_pair   = std::get_if<FlashNextSharedGateUpPair>(&weights.shared_gate_up);
    const auto* shared_packed = std::get_if<Weight>(&weights.shared_gate_up);
    NINFER_PERF_SCOPE(
        weights.routed_gate_up.qtype == QType::NVFP4 ? "moe.nvfp4" : "moe.bf16", input.ne[1], 0, 0,
        flash_next_work::moe(input.ne[1], weights.routed_gate_up.qtype == QType::NVFP4,
                             shared_packed != nullptr,
                             weights.shared_down.qtype == QType::FP8_E4M3FN_ROW_BF16));

    const int tokens = input.ne[1];
    // One resident persistent wave sized for the active device, not the
    // reference part: 3 blocks per SM (device_sm_count() falls back to the
    // 5090's 170 SMs). The grouped GEMMs stride their work list by gridDim.x,
    // so any grid is correct.
    const int persistent_blocks = kPrefillBlocksPerSm * device_sm_count();
    if (input.dtype != DType::BF16 || !input.is_contiguous() || input.ne[0] != kHidden ||
        tokens <= 0 || destination.dtype != DType::BF16 || !destination.is_contiguous() ||
        destination.ne[0] != kHidden || destination.ne[1] != tokens ||
        weights.router.n != kExperts || weights.router.k != kHidden ||
        weights.shared_down.n != kHidden || weights.shared_down.k != kIntermediate ||
        weights.shared_scale.n != 1 || weights.shared_scale.k != kHidden) {
        throw std::invalid_argument("flash_next_moe: invalid exact geometry");
    }
    const auto bf16 = [](const Weight& weight, int rows) {
        return weight.qtype == QType::BF16 && weight.layout == QuantLayout::Contiguous &&
               weight.n == rows && weight.k == kHidden;
    };
    const auto fp8 = [](const Weight& weight) {
        return weight.qtype == QType::FP8_E4M3FN_ROW_BF16 && weight.layout == QuantLayout::RowScale;
    };
    if ((shared_pair != nullptr &&
         (!bf16(shared_pair->gate, kIntermediate) || !bf16(shared_pair->up, kIntermediate))) ||
        (shared_packed != nullptr &&
         (!fp8(*shared_packed) || shared_packed->n != 2 * kIntermediate ||
          shared_packed->k != kHidden)) ||
        (!fp8(weights.shared_down) && (weights.shared_down.qtype != QType::BF16 ||
                                       weights.shared_down.layout != QuantLayout::Contiguous))) {
        throw std::invalid_argument("flash_next_moe: unsupported shared-expert representation");
    }
    require_bank(weights.routed_gate_up, 2 * kIntermediate, kHidden,
                 "flash_next_moe: invalid routed gate/up bank");
    require_bank(weights.routed_down, kHidden, kIntermediate,
                 "flash_next_moe: invalid routed down bank");
    auto scope    = workspace.scope();
    Tensor scores = workspace.alloc(DType::BF16, {kExperts, tokens});
    linear(input, weights.router, scores, stream, bf16_gemm);
    Tensor ids          = workspace.alloc(DType::I32, {kTop, tokens});
    Tensor alpha        = workspace.alloc(DType::FP32, {kTop, tokens});
    Tensor shared_alpha = workspace.alloc(DType::FP32, {tokens});
    // The NVFP4 expert-grouped route packs its assignments by expert. Up to kFusedRouteTokens rows
    // (every decode and verify shape) the routing kernel does it in the same launch; one job per
    // expert then matches every token tile the route uses (16 or 32 >= tokens).
    const int assignments     = tokens * kTop;
    const bool nvfp4          = weights.routed_gate_up.qtype == QType::NVFP4;
    const bool grouped        = nvfp4 && tokens >= kGroupedDecodeMinTokens;
    const bool fused_grouping = grouped && tokens <= kFusedRouteTokens;
    static_assert(kFusedRouteTokens <= kGroupedTokenTile);
    Tensor offsets;
    Tensor packed_index;
    Tensor packed_expert;
    Tensor job_experts;
    Tensor job_columns;
    Tensor job_count;
    if (grouped) {
        offsets       = workspace.alloc(DType::I32, {kExperts + 1});
        packed_index  = workspace.alloc(DType::I32, {assignments});
        packed_expert = workspace.alloc(DType::I32, {assignments});
        job_experts   = workspace.alloc(DType::I32, {assignments});
        job_columns   = workspace.alloc(DType::I32, {assignments});
        job_count     = workspace.alloc(DType::I32, {1});
    }
    const auto i32 = [](Tensor& tensor) { return static_cast<int*>(tensor.data); };
    route_kernel<<<(tokens + kRouteWarps - 1) / kRouteWarps, kRouteWarps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scores.data),
        static_cast<const __nv_bfloat16*>(input.data),
        static_cast<const __nv_bfloat16*>(weights.shared_scale.qdata),
        RouteOutputs{
            i32(ids), static_cast<float*>(alpha.data), static_cast<float*>(shared_alpha.data),
            fused_grouping ? i32(offsets) : nullptr, fused_grouping ? i32(packed_index) : nullptr,
            fused_grouping ? i32(packed_expert) : nullptr,
            fused_grouping ? i32(job_experts) : nullptr,
            fused_grouping ? i32(job_columns) : nullptr, fused_grouping ? i32(job_count) : nullptr},
        tokens);
    Tensor shared_activation = workspace.alloc(DType::BF16, {kIntermediate, tokens});
    if (shared_packed != nullptr) {
        linear_swiglu(input, *shared_packed, shared_activation, workspace, stream);
    } else if (tokens == 1) {
        detail::flash_next::launch_bf16_shared_swiglu_decode(
            input, shared_pair->gate, shared_pair->up, shared_activation, stream);
    } else {
        Tensor shared_gate = workspace.alloc(DType::BF16, {kIntermediate, tokens});
        Tensor shared_up   = workspace.alloc(DType::BF16, {kIntermediate, tokens});
        linear(input, shared_pair->gate, shared_gate, stream, bf16_gemm);
        linear(input, shared_pair->up, shared_up, stream, bf16_gemm);
        silu_mul(shared_gate, shared_up, shared_activation, stream);
    }
    linear(shared_activation, weights.shared_down, destination, stream, bf16_gemm);
    Tensor routed_activation = workspace.alloc(DType::BF16, {kIntermediate, kTop, tokens});
    if (nvfp4) {
        if (!grouped) {
            run_nvfp4_decode_routes(input, weights, ids, alpha, shared_alpha, destination,
                                    routed_activation, workspace, stream, tokens);
            return;
        }
        const bool decode_grouped = tokens <= kDecodeGroupedTokenTile;
        const bool large_grouped  = tokens >= 4096;
        if (!fused_grouping) {
            const int grouped_token_tile =
                large_grouped ? kLargeGroupedTokenTile : kGroupedTokenTile;
            Tensor local_rank = workspace.alloc(DType::I32, {assignments});
            Tensor counts     = workspace.alloc(DType::I32, {kExperts});
            CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), stream));
            CUDA_CHECK(cudaMemsetAsync(job_count.data, 0, job_count.bytes(), stream));
            count_routes_kernel<<<kExperts, 256, 0, stream>>>(static_cast<const int*>(ids.data),
                                                              i32(local_rank), i32(counts), tokens);
            scan_routes_kernel<<<1, kExperts, 0, stream>>>(static_cast<const int*>(counts.data),
                                                           i32(offsets));
            make_route_jobs_kernel<<<1, kExperts, 0, stream>>>(static_cast<const int*>(counts.data),
                                                               i32(job_experts), i32(job_columns),
                                                               i32(job_count), grouped_token_tile);
            pack_routes_kernel<<<(assignments + 255) / 256, 256, 0, stream>>>(
                static_cast<const int*>(ids.data), static_cast<const int*>(local_rank.data),
                static_cast<const int*>(offsets.data), i32(packed_index), i32(packed_expert),
                assignments);
        }

#ifdef NINFER_PERFORMANCE_TRACE
        if (RouteStats* stats = route_stats(); stats != nullptr && tokens <= kFusedRouteTokens) {
            route_stats_kernel<<<1, 1, 0, stream>>>(static_cast<const int*>(job_count.data),
                                                    tokens, stats);
        }
#endif
        Tensor gate_codes  = workspace.alloc(DType::U8, {kHidden / 2, assignments});
        Tensor gate_scales = workspace.alloc(DType::U8, {kHidden / 16, assignments});
        gather_quantize_routes_kernel<<<(assignments * (kHidden / 16) + 255) / 256, 256, 0,
                                        stream>>>(
            static_cast<const __nv_bfloat16*>(input.data), static_cast<const int*>(ids.data),
            static_cast<const int*>(packed_index.data), weights.routed_gate_up.input_scale_divisors,
            static_cast<std::uint8_t*>(gate_codes.data),
            static_cast<std::uint8_t*>(gate_scales.data), assignments, kHidden);
        detail::flash_next::Nvfp4W4a4MaterializedActivation gate_input{
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
            constexpr int kGateRowsPerBlock = DecodeGateSchedule::kBlockN / 2;
            const int gate_blocks =
                std::min(persistent_blocks, assignments * (kIntermediate / kGateRowsPerBlock));
            detail::flash_next::nvfp4_w4a4_mma_kernel<
                GroupedGateGeometry, DecodeGateSchedule, detail::flash_next::Nvfp4IdentityEpilogue,
                GroupedSiluOutput, GroupedGateRows, true, GroupedWork<GroupedGateRows>>
                <<<gate_blocks, DecodeGateSchedule::kThreads, 0, stream>>>(
                    gate_input, static_cast<const std::uint8_t*>(weights.routed_gate_up.codes),
                    static_cast<const std::uint8_t*>(weights.routed_gate_up.scales), assignments,
                    1.0F, detail::flash_next::Nvfp4IdentityEpilogue{},
                    GroupedSiluOutput{static_cast<__nv_bfloat16*>(routed_activation.data)},
                    GroupedGateRows{0, kGateRowsPerBlock}, gate_work);
        } else if (large_grouped) {
            detail::flash_next::nvfp4_w4a4_mma_kernel<GroupedGateGeometry, LargeGroupedSchedule,
                                          detail::flash_next::Nvfp4IdentityEpilogue, GroupedSiluOutput,
                                          GroupedGateRows, true, GroupedWork<GroupedGateRows>>
                <<<persistent_blocks, LargeGroupedSchedule::kThreads, 0, stream>>>(
                    gate_input, static_cast<const std::uint8_t*>(weights.routed_gate_up.codes),
                    static_cast<const std::uint8_t*>(weights.routed_gate_up.scales), assignments,
                    1.0F, detail::flash_next::Nvfp4IdentityEpilogue{},
                    GroupedSiluOutput{static_cast<__nv_bfloat16*>(routed_activation.data)},
                    GroupedGateRows{0, LargeGroupedSchedule::kBlockN / 2}, gate_work);
        } else {
            detail::flash_next::nvfp4_w4a4_mma_kernel<GroupedGateGeometry, GroupedGateSchedule,
                                          detail::flash_next::Nvfp4IdentityEpilogue, GroupedSiluOutput,
                                          GroupedGateRows, true, GroupedWork<GroupedGateRows>>
                <<<persistent_blocks, GroupedGateSchedule::kThreads, 0, stream>>>(
                    gate_input, static_cast<const std::uint8_t*>(weights.routed_gate_up.codes),
                    static_cast<const std::uint8_t*>(weights.routed_gate_up.scales), assignments,
                    1.0F, detail::flash_next::Nvfp4IdentityEpilogue{},
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
        detail::flash_next::Nvfp4W4a4MaterializedActivation down_input{
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
            constexpr int kDownRowsPerBlock = DecodeDownSchedule::kBlockN;
            const int down_blocks = std::min(persistent_blocks, assignments * (kHidden / kDownRowsPerBlock));
            detail::flash_next::nvfp4_w4a4_mma_kernel<
                GroupedDownGeometry, DecodeDownSchedule, detail::flash_next::Nvfp4IdentityEpilogue,
                GroupedOutput, GroupedDownRows, false, GroupedWork<GroupedDownRows>>
                <<<down_blocks, DecodeDownSchedule::kThreads, 0, stream>>>(
                    down_input, static_cast<const std::uint8_t*>(weights.routed_down.codes),
                    static_cast<const std::uint8_t*>(weights.routed_down.scales), assignments, 1.0F,
                    detail::flash_next::Nvfp4IdentityEpilogue{},
                    GroupedOutput{static_cast<__nv_bfloat16*>(grouped_output.data), kHidden},
                    GroupedDownRows{}, down_work);
        } else if (large_grouped) {
            detail::flash_next::nvfp4_w4a4_mma_kernel<GroupedDownGeometry, LargeGroupedSchedule,
                                          detail::flash_next::Nvfp4IdentityEpilogue, GroupedOutput,
                                          GroupedDownRows, false, GroupedWork<GroupedDownRows>>
                <<<persistent_blocks, LargeGroupedSchedule::kThreads, 0, stream>>>(
                    down_input, static_cast<const std::uint8_t*>(weights.routed_down.codes),
                    static_cast<const std::uint8_t*>(weights.routed_down.scales), assignments, 1.0F,
                    detail::flash_next::Nvfp4IdentityEpilogue{},
                    GroupedOutput{static_cast<__nv_bfloat16*>(grouped_output.data), kHidden},
                    GroupedDownRows{}, down_work);
        } else {
            detail::flash_next::nvfp4_w4a4_mma_kernel<GroupedDownGeometry, GroupedDownSchedule,
                                          detail::flash_next::Nvfp4IdentityEpilogue, GroupedOutput,
                                          GroupedDownRows, false, GroupedWork<GroupedDownRows>>
                <<<persistent_blocks, GroupedDownSchedule::kThreads, 0, stream>>>(
                    down_input, static_cast<const std::uint8_t*>(weights.routed_down.codes),
                    static_cast<const std::uint8_t*>(weights.routed_down.scales), assignments, 1.0F,
                    detail::flash_next::Nvfp4IdentityEpilogue{},
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
        Tensor local_rank = workspace.alloc(DType::I32, {assignments});
        Tensor counts     = workspace.alloc(DType::I32, {kExperts});
        offsets           = workspace.alloc(DType::I32, {kExperts + 1});
        packed_index      = workspace.alloc(DType::I32, {assignments});
        job_experts       = workspace.alloc(DType::I32, {assignments});
        job_columns       = workspace.alloc(DType::I32, {assignments});
        job_count         = workspace.alloc(DType::I32, {1});
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
        detail::flash_next::bf16_grouped_gemm_mma_kernel<Bf16GroupedGateGeometry, Bf16GroupedSchedule>
            <<<persistent_blocks, Bf16GroupedSchedule::kThreads, Bf16GroupedSchedule::kSharedBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(packed_input.data),
                static_cast<const __nv_bfloat16*>(weights.routed_gate_up.codes),
                static_cast<const int*>(offsets.data), static_cast<const int*>(job_experts.data),
                static_cast<const int*>(job_columns.data), static_cast<const int*>(job_count.data),
                static_cast<__nv_bfloat16*>(gate_up.data));
        grouped_silu_kernel<<<(assignments * kIntermediate + 255) / 256, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate_up.data),
            static_cast<__nv_bfloat16*>(routed_activation.data), assignments);
        Tensor grouped_output = workspace.alloc(DType::BF16, {kHidden, assignments});
        detail::flash_next::bf16_grouped_gemm_mma_kernel<Bf16GroupedDownGeometry, Bf16GroupedSchedule>
            <<<persistent_blocks, Bf16GroupedSchedule::kThreads, Bf16GroupedSchedule::kSharedBytes, stream>>>(
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
