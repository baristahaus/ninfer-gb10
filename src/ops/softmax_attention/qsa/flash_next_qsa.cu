#include "ninfer/ops/flash_next_qsa.h"

#include "ops/flash_next_work.h"

#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"
#include "ops/linear/bf16/flash_next/bf16_launch.h"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops {
namespace {

constexpr int kHeadDim = 256;
constexpr int kQueryHeads = 24;
constexpr int kKvHeads = 2;
constexpr int kIndexDim = 128;
constexpr int kIndexHeads = 4;
constexpr int kTopGroups = 512;
constexpr int kRatio = 4;
constexpr int kOutputWidth = kTopGroups * kRatio + kRatio - 1;
static_assert(kOutputWidth == kFlashNextQsaSelectedTokens);
constexpr float kTheta = 1.0e7F;
constexpr int kTopkBlockThreads = 256;
constexpr int kTopkItemsPerThread = 8;
constexpr int kTopkBlockItems = kTopkBlockThreads * kTopkItemsPerThread;

struct FlashQsaKVGeometry {
    static constexpr int KVHeads = kKvHeads;
};

__global__ void split_query_gate_kernel(const __nv_bfloat16* packed,
                                        __nv_bfloat16* query,
                                        __nv_bfloat16* gate,
                                        int tokens) {
    const std::int64_t count = static_cast<std::int64_t>(kQueryHeads) * kHeadDim * tokens;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const int d = static_cast<int>(index % kHeadDim);
        const std::int64_t row = index / kHeadDim;
        const int head = static_cast<int>(row % kQueryHeads);
        const int token = static_cast<int>(row / kQueryHeads);
        const std::int64_t packed_base = static_cast<std::int64_t>(token) *
                                         (2 * kQueryHeads * kHeadDim) +
                                         static_cast<std::int64_t>(head) * 2 * kHeadDim;
        query[index] = packed[packed_base + d];
        gate[index] = packed[packed_base + kHeadDim + d];
    }
}

__global__ void expand_text_positions_kernel(const int* input, int* output, int tokens) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < 3 * tokens) { output[index] = input[index % tokens]; }
}

__global__ void select_shared_indices_kernel(const int* input, const int* selectors,
                                             int* output, int width, int batch) {
    const int item = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int lane = static_cast<int>(blockIdx.y);
    if (item >= kOutputWidth || lane >= batch) { return; }
    const int column = selectors[lane];
    if (column < 0 || column >= width) { return; }
    output[item + static_cast<std::int64_t>(kOutputWidth) * lane] =
        input[item + static_cast<std::int64_t>(kOutputWidth) *
                         (column + width * lane)];
}

__device__ float block_sum(float value, float* partial) {
    for (int offset = 16; offset != 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { partial[warp] = value; }
    __syncthreads();
    if (warp == 0) {
        value = lane < static_cast<int>(blockDim.x) / 32 ? partial[lane] : 0.0F;
        for (int offset = 16; offset != 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        if (lane == 0) { partial[0] = value; }
    }
    __syncthreads();
    return partial[0];
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset != 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ int physical_page(const int* tables, int logical_pages, int table_row,
                             int position) {
    return tables[position / kPagedKVPageSize + logical_pages * table_row];
}

__global__ void prepare_index_query_kernel(const __nv_bfloat16* projected,
                                           const __nv_bfloat16* norm_weight,
                                           const int* positions, int tokens,
                                           __nv_bfloat16* output) {
    const int head = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int d = static_cast<int>(threadIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(token) * kIndexHeads * kIndexDim +
                              head * kIndexDim;
    const float x = __bfloat162float(projected[base + d]);
    __shared__ float partial[4];
    const float square = block_sum(x * x, partial);
    const float normalized = x * rsqrtf(square / kIndexDim + 1.0e-6F) *
                             (1.0F + __bfloat162float(norm_weight[d]));
    __shared__ float values[kIndexDim];
    values[d] = normalized;
    __syncthreads();
    float result = normalized;
    if (d < 64) {
        const int pair = d & 31;
        const int other = d < 32 ? d + 32 : d - 32;
        const int axis = pair % 3;
        const float angle = static_cast<float>(positions[token + tokens * axis]) *
                            powf(kTheta, -2.0F * pair / 64.0F);
        const float first = d < 32 ? values[d] : values[other];
        const float second = d < 32 ? values[other] : values[d];
        result = d < 32 ? first * cosf(angle) - second * sinf(angle)
                        : second * cosf(angle) + first * sinf(angle);
    }
    output[base + d] = __float2bfloat16_rn(result);
}

__global__ void append_cache_kernel(const __nv_bfloat16* key, const __nv_bfloat16* value,
                                    const __nv_bfloat16* raw_index_key,
                                    const int* cache_positions, const int* rope_positions,
                                    const int* valid_columns, const int* table_rows,
                                    int width, int batch, int logical_pages,
                                    const int* tables, __nv_bfloat16* key_pages,
                                    __nv_bfloat16* value_pages, __nv_bfloat16* raw_pages,
                                    int* position_pages) {
    const int token = static_cast<int>(blockIdx.x);
    const int lane = token / width;
    const int column = token - lane * width;
    if (lane >= batch || column >= valid_columns[lane]) { return; }
    const int position = cache_positions[token];
    const int page = physical_page(tables, logical_pages, table_rows[lane], position);
    const int page_offset = position % kPagedKVPageSize;
    for (int index = static_cast<int>(threadIdx.x); index < 2 * kKvHeads * kHeadDim;
         index += static_cast<int>(blockDim.x)) {
        const int is_value = index >= kKvHeads * kHeadDim;
        const int local = index - is_value * kKvHeads * kHeadDim;
        const int head = local / kHeadDim;
        const int d = local - head * kHeadDim;
        const std::int64_t cache_offset = d + static_cast<std::int64_t>(kHeadDim) *
            (page_offset + kPagedKVPageSize * (head + kKvHeads * page));
        const std::int64_t source = d + static_cast<std::int64_t>(kHeadDim) *
            (head + kKvHeads * token);
        if (is_value) {
            value_pages[cache_offset] = value[source];
        } else {
            key_pages[cache_offset] = key[source];
        }
    }
    for (int d = static_cast<int>(threadIdx.x); d < kIndexDim;
         d += static_cast<int>(blockDim.x)) {
        raw_pages[d + static_cast<std::int64_t>(kIndexDim) *
                         (page_offset + kPagedKVPageSize * page)] =
            raw_index_key[d + static_cast<std::int64_t>(kIndexDim) * token];
    }
    if (threadIdx.x < 3) {
        position_pages[threadIdx.x + 3LL * (page_offset + kPagedKVPageSize * page)] =
            rope_positions[token + static_cast<std::int64_t>(width * batch) * threadIdx.x];
    }
}

__global__ void append_cache_fp8_kernel(
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    const __nv_bfloat16* raw_index_key, const int* cache_positions,
    const int* rope_positions, const int* valid_columns, const int* table_rows,
    int width, int batch, int logical_pages, const int* tables,
    std::uint8_t* key_pages, std::uint8_t* value_pages,
    __half* key_scales, __half* value_scales,
    __nv_bfloat16* raw_pages, int* position_pages) {
    const int token = static_cast<int>(blockIdx.x);
    const int lane_index = token / width;
    const int column = token - lane_index * width;
    if (lane_index >= batch || column >= valid_columns[lane_index]) { return; }
    const int position = cache_positions[token];
    const int page = physical_page(tables, logical_pages, table_rows[lane_index], position);
    const int page_offset = position % kPagedKVPageSize;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (warp < kKvHeads) {
        kv_cache_append_full_fp8_row<FlashQsaKVGeometry>(
            key, value, key_pages, value_pages, key_scales, value_scales,
            token, warp, page, page_offset, lane);
    }
    for (int d = static_cast<int>(threadIdx.x); d < kIndexDim;
         d += static_cast<int>(blockDim.x)) {
        raw_pages[d + static_cast<std::int64_t>(kIndexDim) *
                         (page_offset + kPagedKVPageSize * page)] =
            raw_index_key[d + static_cast<std::int64_t>(kIndexDim) * token];
    }
    if (threadIdx.x < 3) {
        position_pages[threadIdx.x + 3LL * (page_offset + kPagedKVPageSize * page)] =
            rope_positions[token + static_cast<std::int64_t>(width * batch) * threadIdx.x];
    }
}

__global__ void hadamard_rows_kernel(__nv_bfloat16* rows, int row_count) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= row_count || lane >= 32) { return; }
    float values[8];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        values[item] = __bfloat162float(rows[static_cast<std::int64_t>(row) * kHeadDim +
                                             lane + 32 * item]);
    }
    normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        rows[static_cast<std::int64_t>(row) * kHeadDim + lane + 32 * item] =
            __float2bfloat16_rn(values[item]);
    }
}

__global__ void compress_index_groups_kernel(__nv_bfloat16* raw_pages,
                                             const int* position_pages,
                                             const __nv_bfloat16* key_norm,
                                             const int* cache_positions,
                                             const int* valid_columns,
                                             const int* table_rows, int width, int batch,
                                             int logical_pages, const int* tables) {
    const int token = static_cast<int>(blockIdx.x);
    const int batch_lane = token / width;
    const int column = token - batch_lane * width;
    if (batch_lane >= batch || column >= valid_columns[batch_lane]) { return; }
    const int end_position = cache_positions[token];
    if ((end_position & (kRatio - 1)) != kRatio - 1) { return; }
    const int d = static_cast<int>(threadIdx.x);
    const int first_position = end_position - (kRatio - 1);
    const int table_row = table_rows[batch_lane];
    float pooled = 0.0F;
#pragma unroll
    for (int member = 0; member < kRatio; ++member) {
        const int position = first_position + member;
        const int page = physical_page(tables, logical_pages, table_row, position);
        pooled += __bfloat162float(raw_pages[d + static_cast<std::int64_t>(kIndexDim) *
            (position % kPagedKVPageSize + kPagedKVPageSize * page)]);
    }
    pooled = __bfloat162float(__float2bfloat16_rn(pooled * 0.25F));
    __shared__ float partial[4];
    const float square = block_sum(pooled * pooled, partial);
    pooled *= rsqrtf(square / kIndexDim + 1.0e-6F) *
              (1.0F + __bfloat162float(key_norm[d]));
    __shared__ float values[kIndexDim];
    values[d] = pooled;
    __syncthreads();
    if (d < 64) {
        const int pair = d & 31;
        const int other = d < 32 ? d + 32 : d - 32;
        const int page = physical_page(tables, logical_pages, table_row, first_position);
        const int axis = pair % 3;
        const int rope_position = position_pages[axis + 3LL *
            (first_position % kPagedKVPageSize + kPagedKVPageSize * page)];
        const float angle = static_cast<float>(rope_position) *
                            powf(kTheta, -2.0F * pair / 64.0F);
        const float first = d < 32 ? values[d] : values[other];
        const float second = d < 32 ? values[other] : values[d];
        pooled = d < 32 ? first * cosf(angle) - second * sinf(angle)
                        : second * cosf(angle) + first * sinf(angle);
    }
    const int page = physical_page(tables, logical_pages, table_row, end_position);
    raw_pages[d + static_cast<std::int64_t>(kIndexDim) *
        (end_position % kPagedKVPageSize + kPagedKVPageSize * page)] =
        __float2bfloat16_rn(pooled);
}

__global__ void score_groups_batched_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* compressed_pages, const int* tables,
    int logical_pages, const int* cache_positions, const int* valid_columns,
    const int* table_rows, int width, int tokens, int group_extent, int score_stride,
    float* scores) {
    constexpr int kGroupsPerBlock = 8;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane_id = static_cast<int>(threadIdx.x) & 31;
    const int query_token = static_cast<int>(blockIdx.y);
    const int batch_lane = query_token / width;
    const int column = query_token - batch_lane * width;
    const int sequence_length = cache_positions[query_token] + 1;
    const int groups = sequence_length / kRatio;
    for (int group = static_cast<int>(blockIdx.x) * kGroupsPerBlock + warp;
         group < group_extent;
         group += static_cast<int>(gridDim.x) * kGroupsPerBlock) {
    if (column >= valid_columns[batch_lane] || group >= groups) {
        if (lane_id == 0) {
            scores[group + static_cast<std::int64_t>(score_stride) * query_token] = -INFINITY;
        }
        continue;
    }
    const int table_row = table_rows[batch_lane];
    float compressed[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d = lane_id + 32 * item;
        const int position = group * kRatio + (kRatio - 1);
        const int page = physical_page(tables, logical_pages, table_row, position);
        compressed[item] = __bfloat162float(compressed_pages[
            d + static_cast<std::int64_t>(kIndexDim) *
                (position % kPagedKVPageSize + kPagedKVPageSize * page)]);
    }
    float score = 0.0F;
#pragma unroll
    for (int head = 0; head < kIndexHeads; ++head) {
        float dot = 0.0F;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int d = lane_id + 32 * item;
            dot = fmaf(compressed[item], __bfloat162float(query[d + kIndexDim *
                (head + static_cast<std::int64_t>(kIndexHeads) * query_token)]), dot);
        }
        dot = warp_sum(dot);
        dot = __shfl_sync(0xffffffffU, dot, 0) * rsqrtf(static_cast<float>(kIndexDim));
        score += fmaxf(dot, 0.0F);
    }
    if (lane_id == 0) {
        scores[group + static_cast<std::int64_t>(score_stride) * query_token] = score;
    }
    }
}

constexpr int kScoreRows = 16;
constexpr int kScoreColumns = 64;
constexpr int kScoreTilesPerBlock = 8;
constexpr int kScoreSharedBytes =
    (kScoreRows + kScoreColumns) * kIndexDim * sizeof(__nv_bfloat16);

// Score one 64-group tile as Q[16,128] x K[128,64]. Only the first four padded
// query rows are live; the warp reduces their ReLU'd MMA outputs into one score
// per compressed group. This is the prefill geometry used by the fast NVIDIA
// implementation and avoids scalar dot products at every query/group pair.
__launch_bounds__(64, 4) __global__ void score_groups_mma_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* compressed_pages,
    const int* tables, int logical_pages, const int* cache_positions,
    const int* valid_columns, const int* table_rows, int width,
    int group_extent, int score_stride, float* scores) {
    constexpr int kNTiles = (kScoreColumns / 2) / 8;
    constexpr int kKSteps = kIndexDim / 16;
    const int token = static_cast<int>(blockIdx.y);
    const int batch_lane = token / width;
    const int column = token - batch_lane * width;
    if (column >= valid_columns[batch_lane]) { return; }
    const int tile_base = static_cast<int>(blockIdx.x) *
                          (kScoreColumns * kScoreTilesPerBlock);
    const int groups = min((cache_positions[token] + 1) / kRatio, group_extent);
    if (tile_base >= groups) { return; }

    extern __shared__ __align__(16) __nv_bfloat16 shared[];
    __nv_bfloat16* query_shared = shared;
    __nv_bfloat16* key_shared = query_shared + kScoreRows * kIndexDim;
    constexpr int kVectorsPerRow = kIndexDim / 8;
    const int tid = static_cast<int>(threadIdx.x);
    for (int vector = tid; vector < kScoreRows * kVectorsPerRow; vector += 64) {
        const int row = vector / kVectorsPerRow;
        const int d = (vector % kVectorsPerRow) * 8;
        __nv_bfloat16* destination = query_shared +
            row * kIndexDim + causal_prompt_swz(row, d);
        if (row < kIndexHeads) {
            const __nv_bfloat16* source = query + d + kIndexDim *
                (row + static_cast<std::int64_t>(kIndexHeads) * token);
            cp_async<16, Cache::cg>(destination, source);
        } else {
            store_vec(destination, make_int4(0, 0, 0, 0));
        }
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int a_matrix = lane >> 3;
    const int a_row = (lane & 7) + ((a_matrix & 1) << 3);
    const int b_row = lane & 7;
    const int b_k = ((lane >> 3) & 1) << 3;
    const unsigned q_base = smem_addr(query_shared) +
                            static_cast<unsigned>(a_row * 256);
    const unsigned q_a = static_cast<unsigned>((a_matrix >> 1) << 4);
    const unsigned q_r = static_cast<unsigned>((lane & 7) << 4);
    const unsigned k_base = smem_addr(key_shared) + static_cast<unsigned>(warp * 8192) +
                            static_cast<unsigned>(b_row * 256) +
                            (static_cast<unsigned>(lane >> 4) << 11);
    const unsigned k_a = static_cast<unsigned>((b_k >> 3) << 4);
    const unsigned k_r = static_cast<unsigned>(b_row << 4);

    const int table_row = table_rows[batch_lane];
#pragma unroll
    for (int tile = 0; tile < kScoreTilesPerBlock; ++tile) {
        const int tile_begin = tile_base + tile * kScoreColumns;
        if (tile_begin >= groups) { break; }
        for (int vector = tid; vector < kScoreColumns * kVectorsPerRow; vector += 64) {
            const int row = vector / kVectorsPerRow;
            const int d = (vector % kVectorsPerRow) * 8;
            __nv_bfloat16* destination = key_shared +
                row * kIndexDim + causal_prompt_swz(row, d);
            const int group = tile_begin + row;
            if (group < groups) {
                const int position = group * kRatio + (kRatio - 1);
                const int page = physical_page(tables, logical_pages, table_row, position);
                const std::int64_t offset = d + static_cast<std::int64_t>(kIndexDim) *
                    (position % kPagedKVPageSize + kPagedKVPageSize * page);
                cp_async<16, Cache::cg>(destination, compressed_pages + offset);
            } else {
                store_vec(destination, make_int4(0, 0, 0, 0));
            }
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();

        float accumulator[kNTiles][4] = {};
        unsigned af[2][4];
        unsigned bf[2][kNTiles][2];
        ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                    causal_prompt_swz_addr(q_base, 0U, q_a, q_r));
#pragma unroll
        for (int nt = 0; nt < kNTiles; nt += 2) {
            ldmatrix_x4(bf[0][nt][0], bf[0][nt][1], bf[0][nt + 1][0], bf[0][nt + 1][1],
                        causal_prompt_swz_addr(k_base + static_cast<unsigned>(nt * 2048),
                                               0U, k_a, k_r));
        }
#pragma unroll
        for (int step = 0; step < kKSteps; ++step) {
            const int current = step & 1;
            const int next = current ^ 1;
            if (step + 1 < kKSteps) {
                const unsigned contraction = static_cast<unsigned>((step + 1) << 5);
                ldmatrix_x4(af[next][0], af[next][1], af[next][2], af[next][3],
                            causal_prompt_swz_addr(q_base, contraction, q_a, q_r));
#pragma unroll
                for (int nt = 0; nt < kNTiles; nt += 2) {
                    ldmatrix_x4(
                        bf[next][nt][0], bf[next][nt][1],
                        bf[next][nt + 1][0], bf[next][nt + 1][1],
                        causal_prompt_swz_addr(k_base + static_cast<unsigned>(nt * 2048),
                                               contraction, k_a, k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < kNTiles; ++nt) {
                mma_bf16(accumulator[nt][0], accumulator[nt][1],
                         accumulator[nt][2], accumulator[nt][3],
                         af[current][0], af[current][1], af[current][2], af[current][3],
                         bf[current][nt][0], bf[current][nt][1]);
            }
        }

#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
            float score0 = gid < kIndexHeads ? fmaxf(accumulator[nt][0], 0.0F) : 0.0F;
            float score1 = gid < kIndexHeads ? fmaxf(accumulator[nt][1], 0.0F) : 0.0F;
            score0 += __shfl_down_sync(0xffffffffU, score0, 4);
            score1 += __shfl_down_sync(0xffffffffU, score1, 4);
            score0 += __shfl_down_sync(0xffffffffU, score0, 8);
            score1 += __shfl_down_sync(0xffffffffU, score1, 8);
            if (gid == 0) {
                const int group0 = tile_begin + warp * 32 + nt * 8 + 2 * lid;
                const int group1 = group0 + 1;
                if (group0 < groups) {
                    scores[group0 + static_cast<std::int64_t>(score_stride) * token] =
                        score0 * rsqrtf(static_cast<float>(kIndexDim));
                }
                if (group1 < groups) {
                    scores[group1 + static_cast<std::int64_t>(score_stride) * token] =
                        score1 * rsqrtf(static_cast<float>(kIndexDim));
                }
            }
        }
        __syncthreads();
    }
}

// Inclusive block scan over the 256 threads of a top-k block; returns the block total.
__device__ __forceinline__ int topk_block_inclusive_scan(int& value, int* warp_totals) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int add = __shfl_up_sync(0xffffffffU, value, offset);
        if (lane >= offset) { value += add; }
    }
    if (lane == 31) { warp_totals[warp] = value; }
    __syncthreads();
    int base = 0;
    int total = 0;
#pragma unroll
    for (int w = 0; w < kTopkBlockThreads / 32; ++w) {
        const int count = warp_totals[w];
        base += w < warp ? count : 0;
        total += count;
    }
    value += base;
    __syncthreads();
    return total;
}

constexpr int kSelectStagedGroups = 8192;

// Exact radix selection of the 512 greatest non-negative QSA scores.  Keeping
// only the threshold lets one block select a token without materializing a
// second score buffer or capturing CUB's size-dependent graph topology. Scores are staged in
// shared memory when they fit; digits are located with a block scan.
__global__ void __launch_bounds__(kTopkBlockThreads) select_top_groups_kernel(
    const float* scores, const int* cache_positions, int score_stride, int group_extent,
    int* selected) {
    const int token = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int groups = min((cache_positions[token] + 1) / kRatio, group_extent);
    int* output = selected + static_cast<std::int64_t>(kTopGroups) * token;
    // Every visible group is selected before the compressed history exceeds the budget.  Emit
    // those groups directly in causal order.  Routing the degenerate case through the atomic
    // threshold compaction below made their order CTA-scheduler dependent; the selected-attention
    // reduction consequently changed between CUDA Graph replays even though the selected set was
    // identical.
    if (groups <= kTopGroups) {
        for (int item = tid; item < kTopGroups; item += kTopkBlockThreads) {
            output[item] = item < groups ? item : -1;
        }
        return;
    }
    __shared__ unsigned staged[kSelectStagedGroups];
    __shared__ int histogram[256];
    __shared__ int warp_totals[kTopkBlockThreads / 32];
    __shared__ unsigned prefix;
    __shared__ int rank;
    const float* row = scores + static_cast<std::int64_t>(score_stride) * token;
    const bool use_staged = groups <= kSelectStagedGroups;
    if (use_staged) {
        for (int group = tid; group < groups; group += kTopkBlockThreads) {
            staged[group] = __float_as_uint(row[group]);
        }
    }
    const auto bits_at = [&](int group) {
        return use_staged ? staged[group] : __float_as_uint(row[group]);
    };
    if (tid == 0) {
        prefix = 0;
        rank = kTopGroups;
    }
    for (int shift = 24; shift >= 0; shift -= 8) {
        histogram[tid] = 0;
        __syncthreads();
        const unsigned mask = shift == 24 ? 0U : (0xffffffffU << (shift + 8));
        const unsigned current_prefix = prefix;
        for (int group = tid; group < groups; group += kTopkBlockThreads) {
            const unsigned bits = bits_at(group);
            if ((bits & mask) == current_prefix) {
                atomicAdd(histogram + ((bits >> shift) & 255U), 1);
            }
        }
        __syncthreads();
        // Thread i owns digit 255 - i, so the inclusive scan counts the groups at or above it.
        const int own = histogram[255 - tid];
        int at_or_above = own;
        const int current_rank = rank;
        topk_block_inclusive_scan(at_or_above, warp_totals);
        if (at_or_above >= current_rank && at_or_above - own < current_rank) {
            prefix = current_prefix | (static_cast<unsigned>(255 - tid) << shift);
            rank = current_rank - (at_or_above - own);
        }
        __syncthreads();
    }
    // Emit every score above the threshold, then the lowest-id equal scores, each in ascending
    // group order: the exact (score descending, group id ascending) top-k contract used by the
    // hierarchical selector as well. Threads own contiguous group ranges.
    const unsigned threshold = prefix;
    const int per_thread = (groups + kTopkBlockThreads - 1) / kTopkBlockThreads;
    const int begin = min(groups, tid * per_thread);
    const int end = min(groups, begin + per_thread);
    int above = 0;
    int equal = 0;
    for (int group = begin; group < end; ++group) {
        const unsigned bits = bits_at(group);
        above += bits > threshold;
        equal += bits == threshold;
    }
    int above_end = above;
    const int above_total = topk_block_inclusive_scan(above_end, warp_totals);
    int equal_end = equal;
    topk_block_inclusive_scan(equal_end, warp_totals);
    int above_slot = above_end - above;
    int equal_slot = above_total + equal_end - equal;
    for (int group = begin; group < end; ++group) {
        const unsigned bits = bits_at(group);
        if (bits > threshold) {
            output[above_slot++] = group;
        } else if (bits == threshold) {
            if (equal_slot < kTopGroups) { output[equal_slot] = group; }
            ++equal_slot;
        }
    }
}

// Exact hierarchical top-k: every global top-512 item must be in the local top-512 of the
// 2048-item partition that contains it. Repeating that reduction exposes enough parallelism for
// long-context decode, where a single-CTA radix scan otherwise dominates all QSA work.
__global__ void hierarchical_top_groups_kernel(
    const float* input_scores, const int* input_indices, int input_count, int input_stride,
    float* output_scores, int* output_indices, int output_stride,
    const int* cache_positions, const int* valid_columns, int width, int reduction_level,
    bool expand_final) {
    // Scores are non-negative.  Encode the global group id into the low key bits so equal-score
    // selection has one exact ordering at every hierarchy level: greater score first, then lower
    // group id.  Sorting float keys alone lets CUB choose different tied groups between graph
    // replays, which changes the subsequent attention reduction and can alter an MTP draft.
    using Sort = cub::BlockRadixSort<std::uint64_t, kTopkBlockThreads,
                                     kTopkItemsPerThread, int>;
    __shared__ typename Sort::TempStorage storage;
    const int token = static_cast<int>(blockIdx.y);
    const int begin = static_cast<int>(blockIdx.x) * kTopkBlockItems;
    int visible = (cache_positions[token] + 1) / kRatio;
    for (int level = 0; level < reduction_level; ++level) {
        visible = ((visible + kTopkBlockItems - 1) / kTopkBlockItems) * kTopGroups;
    }
    if (begin >= visible && !expand_final) {
        for (int rank = static_cast<int>(threadIdx.x); rank < kTopGroups;
             rank += static_cast<int>(blockDim.x)) {
            const int output = static_cast<int>(blockIdx.x) * kTopGroups + rank +
                static_cast<std::int64_t>(output_stride) * token;
            output_scores[output] = -INFINITY;
            output_indices[output] = -1;
        }
        return;
    }
    std::uint64_t keys[kTopkItemsPerThread];
    int values[kTopkItemsPerThread];
#pragma unroll
    for (int item = 0; item < kTopkItemsPerThread; ++item) {
        const int local = static_cast<int>(threadIdx.x) * kTopkItemsPerThread + item;
        const int index = begin + local;
        if (index < min(visible, input_count)) {
            values[item] = input_indices == nullptr
                ? index
                : input_indices[index + static_cast<std::int64_t>(input_stride) * token];
            const float score =
                input_scores[index + static_cast<std::int64_t>(input_stride) * token];
            keys[item] = values[item] < 0
                ? 0ULL
                : (static_cast<std::uint64_t>(__float_as_uint(score)) << 32U) |
                      static_cast<std::uint32_t>(INT_MAX - values[item]);
        } else {
            keys[item] = 0ULL;
            values[item] = -1;
        }
    }
    Sort(storage).SortDescending(keys, values);
#pragma unroll
    for (int item = 0; item < kTopkItemsPerThread; ++item) {
        const int rank = static_cast<int>(threadIdx.x) * kTopkItemsPerThread + item;
        if (rank < kTopGroups) {
            if (expand_final) {
                const int batch_lane = token / width;
                const int column = token - batch_lane * width;
                const int selected_count = min((cache_positions[token] + 1) / kRatio,
                                               kTopGroups);
                const int group = column < valid_columns[batch_lane] && rank < selected_count
                    ? values[item]
                    : -1;
#pragma unroll
                for (int member = 0; member < kRatio; ++member) {
                    output_indices[rank * kRatio + member +
                                   static_cast<std::int64_t>(kOutputWidth) * token] =
                        group < 0 ? -1 : group * kRatio + member;
                }
            } else {
                const int output = static_cast<int>(blockIdx.x) * kTopGroups + rank +
                    static_cast<std::int64_t>(output_stride) * token;
                output_scores[output] = values[item] < 0
                    ? -INFINITY
                    : __uint_as_float(static_cast<unsigned>(keys[item] >> 32U));
                output_indices[output] = values[item];
            }
        }
    }
    if (expand_final && threadIdx.x < kRatio - 1) {
        const int batch_lane = token / width;
        const int column = token - batch_lane * width;
        const int sequence_length = cache_positions[token] + 1;
        const int tail_begin = (sequence_length / kRatio) * kRatio;
        const int selected_count = min(sequence_length / kRatio, kTopGroups);
        const int tail = static_cast<int>(threadIdx.x);
        output_indices[selected_count * kRatio + tail +
                       static_cast<std::int64_t>(kOutputWidth) * token] =
            column < valid_columns[batch_lane] && tail_begin + tail < sequence_length
                ? tail_begin + tail
                : -1;
    }
}

__global__ void expand_indices_batched_kernel(const int* selected_groups,
                                              const int* cache_positions,
                                              const int* valid_columns, int width,
                                              int* output) {
    const int token = static_cast<int>(blockIdx.y);
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kOutputWidth) { return; }
    const int batch_lane = token / width;
    const int column = token - batch_lane * width;
    int* token_output = output + static_cast<std::int64_t>(kOutputWidth) * token;
    if (column >= valid_columns[batch_lane]) {
        token_output[index] = -1;
        return;
    }
    const int sequence_length = cache_positions[token] + 1;
    const int groups = sequence_length / kRatio;
    const int selected_count = min(groups, kTopGroups);
    const int selected_tokens = selected_count * kRatio;
    if (index < selected_tokens) {
        const int group = selected_groups[index / kRatio +
            static_cast<std::int64_t>(kTopGroups) * token];
        token_output[index] = group < 0 ? -1 : group * kRatio + index % kRatio;
    } else {
        const int tail = index - selected_tokens;
        const int tail_begin = groups * kRatio;
        token_output[index] = tail < sequence_length - tail_begin ? tail_begin + tail : -1;
    }
}

// Match the collection order used by vLLM's short-row persistent top-k: its vectorized
// histogram collector visits one element from each float4 before advancing to the next lane.
// The selected set is unchanged; only the deterministic accumulation order is reproduced.
__global__ void order_groups_like_persistent_topk_kernel(int* selected_groups,
                                                          const int* cache_positions,
                                                          int group_extent) {
    using Sort = cub::BlockRadixSort<std::uint64_t, kTopkBlockThreads, 2, int>;
    __shared__ typename Sort::TempStorage storage;
    const int token = static_cast<int>(blockIdx.x);
    const int groups = min((cache_positions[token] + 1) / kRatio, group_extent);
    if (groups <= kTopGroups) { return; }
    std::uint64_t keys[2];
    int values[2];
#pragma unroll
    for (int item = 0; item < 2; ++item) {
        const int rank = static_cast<int>(threadIdx.x) * 2 + item;
        const int group = selected_groups[rank +
            static_cast<std::int64_t>(kTopGroups) * token];
        values[item] = group;
        if (group < 0) {
            keys[item] = UINT64_MAX;
        } else if (group_extent <= 8192) {
            keys[item] = (static_cast<std::uint64_t>(group & 3) << 32U) |
                         static_cast<std::uint32_t>(group >> 2);
        } else if (group_extent <= 32768) {
            // vLLM's medium collector uses one 1024-thread CTA and walks the row at a
            // 1024-element stride.  Preserve its warp/iteration/lane visitation order.
            keys[item] = (static_cast<std::uint64_t>((group & 1023) >> 5) << 48U) |
                         (static_cast<std::uint64_t>(group >> 10) << 16U) |
                         static_cast<std::uint32_t>(group & 31);
        } else {
            // Decode logits have the full 65536-group capacity.  With vLLM's 35968-byte
            // decode shared-memory cap this is eight 8192-element CTAs.  Each CTA reserves
            // one contiguous output span, then its 1024 threads collect at stride 1024.
            constexpr int kRadixChunk = 8192;
            const int local = group & (kRadixChunk - 1);
            keys[item] = (static_cast<std::uint64_t>(group / kRadixChunk) << 52U) |
                         (static_cast<std::uint64_t>((local & 1023) >> 5) << 44U) |
                         (static_cast<std::uint64_t>(local >> 10) << 16U) |
                         static_cast<std::uint32_t>(local & 31);
        }
    }
    Sort(storage).Sort(keys, values);
#pragma unroll
    for (int item = 0; item < 2; ++item) {
        const int rank = static_cast<int>(threadIdx.x) * 2 + item;
        selected_groups[rank + static_cast<std::int64_t>(kTopGroups) * token] = values[item];
    }
}

constexpr int kPrefillRows = 16;
constexpr int kPrefillColumns = 16;
constexpr int kPrefillSharedBytes =
    (kPrefillRows + 2 * kPrefillColumns) * kHeadDim * sizeof(__nv_bfloat16) +
    kPrefillColumns * sizeof(int);

template <bool Fp8>
__device__ __forceinline__ void stage_selected_qsa_tile(
    __nv_bfloat16* destination, int* staged_positions,
    const void* cache, const __half* scales,
    const int* indices, const int* tables,
    int logical_pages, int table_row, int kv_head, int token, int tile_begin,
    int item_count) {
    const int tid = static_cast<int>(threadIdx.x);
    for (int row = tid; row < kPrefillColumns; row += static_cast<int>(blockDim.x)) {
        const int item = tile_begin + row;
        staged_positions[row] = item < item_count
            ? (indices == nullptr
                   ? item
                   : indices[item + static_cast<std::int64_t>(kOutputWidth) * token])
            : -1;
    }
    __syncthreads();
    constexpr int kVectorsPerRow = kHeadDim / 8;
    for (int vector = tid; vector < kPrefillColumns * kVectorsPerRow;
         vector += static_cast<int>(blockDim.x)) {
        const int row = vector / kVectorsPerRow;
        const int d = (vector % kVectorsPerRow) * 8;
        __nv_bfloat16* output = destination +
            row * kHeadDim + causal_prompt_swz(row, d);
        const int position = staged_positions[row];
        if (position >= 0) {
            const int page = physical_page(tables, logical_pages, table_row, position);
            const std::int64_t offset = d + static_cast<std::int64_t>(kHeadDim) *
                (position % kPagedKVPageSize + kPagedKVPageSize *
                    (kv_head + kKvHeads * page));
            if constexpr (Fp8) {
                const auto* codes = static_cast<const std::uint8_t*>(cache);
                const std::int64_t scale_offset =
                    position % kPagedKVPageSize + static_cast<std::int64_t>(kPagedKVPageSize) *
                        (kv_head + kKvHeads * page);
#pragma unroll
                for (int item = 0; item < 8; ++item) {
                    output[item] = __float2bfloat16_rn(kv_cache_fp8_dequant_code_to_float(
                        codes[offset + item], scales[scale_offset]));
                }
            } else {
                cp_async<16, Cache::cg>(output,
                    static_cast<const __nv_bfloat16*>(cache) + offset);
            }
        } else {
            store_vec(output, make_int4(0, 0, 0, 0));
        }
    }
}

// Four warps share QK/softmax work for the twelve query heads of one KV head and split the
// 256-value output dimension. This preserves the compact 16-key tile while increasing resident
// PV warp-level work at the same dynamic shared-memory footprint.
template <bool Fp8>
__device__ __forceinline__ void selected_attention_batched_body(
    const __nv_bfloat16* query, const void* key_pages, const void* value_pages,
    const __half* key_scales, const __half* value_scales,
    const int* indices, const int* tables,
    int logical_pages, const int* cache_positions, const int* valid_columns,
    const int* table_rows, int width, int tokens, int num_splits,
    float* partial_maximum, float* partial_denominator, float* partial_numerator,
    __nv_bfloat16* output) {
    constexpr int kHeadsPerKv = kQueryHeads / kKvHeads;
    constexpr int kQkTiles = kPrefillColumns / 8;
    constexpr int kQkSteps = kHeadDim / 16;
    constexpr int kValueWarps = 4;
    constexpr int kPvTiles = kHeadDim / (8 * kValueWarps);
    constexpr int kPvSteps = kPrefillColumns / 16;
    constexpr float kScaleLog2 = (1.0F / 16.0F) * 1.4426950408889634F;
    const int token = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int split = static_cast<int>(blockIdx.z);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int batch_lane = token / width;
    const int column = token - batch_lane * width;
    if (column >= valid_columns[batch_lane]) {
        if (num_splits > 1) {
            const int gid = lane >> 2;
            const int lid = lane & 3;
            const std::int64_t split_base = static_cast<std::int64_t>(kQueryHeads) *
                (token + static_cast<std::int64_t>(tokens) * split);
            if (warp == 0 && lid == 0) {
                if (gid < kHeadsPerKv) {
                    partial_maximum[split_base + kv_head * kHeadsPerKv + gid] = -INFINITY;
                    partial_denominator[split_base + kv_head * kHeadsPerKv + gid] = 0.0F;
                }
                if (gid + 8 < kHeadsPerKv) {
                    partial_maximum[split_base + kv_head * kHeadsPerKv + gid + 8] = -INFINITY;
                    partial_denominator[split_base + kv_head * kHeadsPerKv + gid + 8] = 0.0F;
                }
            }
            for (int nt = 0; nt < kPvTiles; ++nt) {
                const int d = warp * (kHeadDim / kValueWarps) + nt * 8 + 2 * lid;
                if (gid < kHeadsPerKv) {
                    const int head = kv_head * kHeadsPerKv + gid;
                    const std::int64_t base = kHeadDim * (split_base + head);
                    partial_numerator[base + d] = 0.0F;
                    partial_numerator[base + d + 1] = 0.0F;
                }
                if (gid + 8 < kHeadsPerKv) {
                    const int head = kv_head * kHeadsPerKv + gid + 8;
                    const std::int64_t base = kHeadDim * (split_base + head);
                    partial_numerator[base + d] = 0.0F;
                    partial_numerator[base + d + 1] = 0.0F;
                }
            }
        }
        return;
    }

    extern __shared__ __align__(16) __nv_bfloat16 shared[];
    __nv_bfloat16* query_shared = shared;
    __nv_bfloat16* key_shared = query_shared + kPrefillRows * kHeadDim;
    __nv_bfloat16* value_shared = key_shared + kPrefillColumns * kHeadDim;
    int* staged_positions = reinterpret_cast<int*>(
        value_shared + kPrefillColumns * kHeadDim);

    constexpr int kVectorsPerRow = kHeadDim / 8;
    for (int vector = tid; vector < kPrefillRows * kVectorsPerRow;
         vector += static_cast<int>(blockDim.x)) {
        const int row = vector / kVectorsPerRow;
        const int d = (vector % kVectorsPerRow) * 8;
        __nv_bfloat16* destination = query_shared +
            row * kHeadDim + causal_prompt_swz(row, d);
        if (row < kHeadsPerKv) {
            const int head = kv_head * kHeadsPerKv + row;
            const __nv_bfloat16* source = query + d + kHeadDim *
                (head + static_cast<std::int64_t>(kQueryHeads) * token);
            cp_async<16, Cache::cg>(destination, source);
        } else {
            store_vec(destination, make_int4(0, 0, 0, 0));
        }
    }

    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int a_matrix = lane >> 3;
    const int a_row = (lane & 7) + ((a_matrix & 1) << 3);
    const int b_row = lane & 7;
    const int b_k = ((lane >> 3) & 1) << 3;
    const unsigned q_base = smem_addr(query_shared) +
                            static_cast<unsigned>(a_row * 512);
    const unsigned q_a = static_cast<unsigned>((a_matrix >> 1) << 4);
    const unsigned q_r = static_cast<unsigned>((lane & 7) << 4);
    const unsigned k_base = smem_addr(key_shared) + static_cast<unsigned>(b_row * 512) +
                            (static_cast<unsigned>(lane >> 4) << 12);
    const unsigned k_a = static_cast<unsigned>((b_k >> 3) << 4);
    const unsigned k_r = static_cast<unsigned>(b_row << 4);
    const unsigned v_base = smem_addr(value_shared) +
                            static_cast<unsigned>(((lane >> 3) & 1) * 4096) +
                            static_cast<unsigned>(b_row * 512);
    const unsigned v_a = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r = static_cast<unsigned>(b_row << 4);

    __shared__ unsigned probability_shared[32][kPvSteps][4];
    __shared__ float previous_scale_shared[32][2];
    __shared__ float inverse_shared[32][2];
    float accumulator[kPvTiles][4] = {};
    float maximum0 = -INFINITY;
    float maximum1 = -INFINITY;
    float denominator0 = 0.0F;
    float denominator1 = 0.0F;
    const int sequence_length = cache_positions[token] + 1;
    const int item_count = indices == nullptr ? sequence_length : kOutputWidth;
    const int table_row = table_rows[batch_lane];
    const int tile_count = (item_count + kPrefillColumns - 1) / kPrefillColumns;
    const int tile_begin = split * tile_count / num_splits;
    const int tile_end = (split + 1) * tile_count / num_splits;

    cp_commit();
    stage_selected_qsa_tile<Fp8>(key_shared, staged_positions, key_pages, key_scales,
                            indices, tables,
                            logical_pages, table_row, kv_head, token,
                            tile_begin * kPrefillColumns, item_count);
    cp_commit();
    for (int tile = tile_begin; tile < tile_end; ++tile) {
        cp_wait<0>();
        __syncthreads();
        stage_selected_qsa_tile<Fp8>(value_shared, staged_positions, value_pages, value_scales,
                                indices, tables,
                                logical_pages, table_row, kv_head, token,
                                tile * kPrefillColumns, item_count);
        cp_commit();

        unsigned probabilities[kPvSteps][4] = {};
        float previous_scale0 = 0.0F;
        float previous_scale1 = 0.0F;
        if (warp == 0) {
            float scores[kQkTiles][4] = {};
            unsigned af[2][4];
            unsigned bf[2][kQkTiles][2];
            ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                        causal_prompt_swz_addr(q_base, 0U, q_a, q_r));
#pragma unroll
            for (int nt = 0; nt < kQkTiles; nt += 2) {
                ldmatrix_x4(bf[0][nt][0], bf[0][nt][1], bf[0][nt + 1][0], bf[0][nt + 1][1],
                            causal_prompt_swz_addr(k_base + static_cast<unsigned>(nt * 4096),
                                                   0U, k_a, k_r));
            }
#pragma unroll
            for (int step = 0; step < kQkSteps; ++step) {
                const int current = step & 1;
                const int next = current ^ 1;
                if (step + 1 < kQkSteps) {
                    const unsigned contraction = static_cast<unsigned>((step + 1) << 5);
                    ldmatrix_x4(af[next][0], af[next][1], af[next][2], af[next][3],
                                causal_prompt_swz_addr(q_base, contraction, q_a, q_r));
#pragma unroll
                    for (int nt = 0; nt < kQkTiles; nt += 2) {
                        ldmatrix_x4(
                            bf[next][nt][0], bf[next][nt][1],
                            bf[next][nt + 1][0], bf[next][nt + 1][1],
                            causal_prompt_swz_addr(k_base + static_cast<unsigned>(nt * 4096),
                                                   contraction, k_a, k_r));
                    }
                }
#pragma unroll
                for (int nt = 0; nt < kQkTiles; ++nt) {
                    mma_bf16(scores[nt][0], scores[nt][1], scores[nt][2], scores[nt][3],
                             af[current][0], af[current][1], af[current][2], af[current][3],
                             bf[current][nt][0], bf[current][nt][1]);
                }
            }

            float block_maximum0 = -INFINITY;
            float block_maximum1 = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < kQkTiles; ++nt) {
                const int key0 = nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                const bool valid0 = staged_positions[key0] >= 0;
                const bool valid1 = staged_positions[key1] >= 0;
                scores[nt][0] = gid < kHeadsPerKv && valid0 ? scores[nt][0] : -INFINITY;
                scores[nt][1] = gid < kHeadsPerKv && valid1 ? scores[nt][1] : -INFINITY;
                scores[nt][2] = gid + 8 < kHeadsPerKv && valid0 ? scores[nt][2] : -INFINITY;
                scores[nt][3] = gid + 8 < kHeadsPerKv && valid1 ? scores[nt][3] : -INFINITY;
                block_maximum0 = fmaxf(block_maximum0, fmaxf(scores[nt][0], scores[nt][1]));
                block_maximum1 = fmaxf(block_maximum1, fmaxf(scores[nt][2], scores[nt][3]));
            }
            block_maximum0 = warp_max<4>(block_maximum0, 0xffffffffU);
            block_maximum1 = warp_max<4>(block_maximum1, 0xffffffffU);
            const float next_maximum0 = fmaxf(maximum0, block_maximum0);
            const float next_maximum1 = fmaxf(maximum1, block_maximum1);
            const float scaled_maximum0 = next_maximum0 > -INFINITY
                ? next_maximum0 * kScaleLog2 : 0.0F;
            const float scaled_maximum1 = next_maximum1 > -INFINITY
                ? next_maximum1 * kScaleLog2 : 0.0F;
            previous_scale0 = next_maximum0 > -INFINITY
                ? exp2_approx(__fmaf_rn(maximum0, kScaleLog2, -scaled_maximum0)) : 1.0F;
            previous_scale1 = next_maximum1 > -INFINITY
                ? exp2_approx(__fmaf_rn(maximum1, kScaleLog2, -scaled_maximum1)) : 1.0F;
            float block_sum0 = 0.0F;
            float block_sum1 = 0.0F;
#pragma unroll
            for (int nt = 0; nt < kQkTiles; ++nt) {
                const float p00 = scores[nt][0] > -INFINITY
                    ? exp2_approx(__fmaf_rn(scores[nt][0], kScaleLog2, -scaled_maximum0)) : 0.0F;
                const float p01 = scores[nt][1] > -INFINITY
                    ? exp2_approx(__fmaf_rn(scores[nt][1], kScaleLog2, -scaled_maximum0)) : 0.0F;
                const float p10 = scores[nt][2] > -INFINITY
                    ? exp2_approx(__fmaf_rn(scores[nt][2], kScaleLog2, -scaled_maximum1)) : 0.0F;
                const float p11 = scores[nt][3] > -INFINITY
                    ? exp2_approx(__fmaf_rn(scores[nt][3], kScaleLog2, -scaled_maximum1)) : 0.0F;
                block_sum0 += p00 + p01;
                block_sum1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    probabilities[pk][0] = pack_bf16x2(p00, p01);
                    probabilities[pk][1] = pack_bf16x2(p10, p11);
                } else {
                    probabilities[pk][2] = pack_bf16x2(p00, p01);
                    probabilities[pk][3] = pack_bf16x2(p10, p11);
                }
            }
            denominator0 = __fmaf_rn(denominator0, previous_scale0, block_sum0);
            denominator1 = __fmaf_rn(denominator1, previous_scale1, block_sum1);
            maximum0 = next_maximum0;
            maximum1 = next_maximum1;
#pragma unroll
            for (int step = 0; step < kPvSteps; ++step) {
#pragma unroll
                for (int item = 0; item < 4; ++item) {
                    probability_shared[lane][step][item] = probabilities[step][item];
                }
            }
            previous_scale_shared[lane][0] = previous_scale0;
            previous_scale_shared[lane][1] = previous_scale1;
        }
        __syncthreads();
#pragma unroll
        for (int nt = 0; nt < kPvTiles; ++nt) {
            accumulator[nt][0] *= previous_scale_shared[lane][0];
            accumulator[nt][1] *= previous_scale_shared[lane][0];
            accumulator[nt][2] *= previous_scale_shared[lane][1];
            accumulator[nt][3] *= previous_scale_shared[lane][1];
        }

        cp_wait<0>();
        __syncthreads();
        if (tile + 1 < tile_end) {
            stage_selected_qsa_tile<Fp8>(key_shared, staged_positions, key_pages, key_scales,
                                    indices, tables,
                                    logical_pages, table_row, kv_head, token,
                                    (tile + 1) * kPrefillColumns, item_count);
            cp_commit();
        }
        constexpr int kPvHalf = kPvTiles / 2;
        constexpr int kPvLoads = kPvSteps * kPvHalf;
        unsigned vf[2][4];
        ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                      causal_prompt_swz_addr(
                          v_base, static_cast<unsigned>((warp * kPvTiles) << 4), v_a, v_r));
#pragma unroll
        for (int item = 0; item < kPvLoads; ++item) {
            const int step = item / kPvHalf;
            const int nt = (item % kPvHalf) * 2;
            const int current = item & 1;
            const int next = current ^ 1;
            if (item + 1 < kPvLoads) {
                const int next_step = (item + 1) / kPvHalf;
                const int next_nt = warp * kPvTiles + ((item + 1) % kPvHalf) * 2;
                ldmatrix_x4_t(
                    vf[next][0], vf[next][1], vf[next][2], vf[next][3],
                    causal_prompt_swz_addr(
                        v_base + static_cast<unsigned>(next_step * 8192),
                        static_cast<unsigned>(next_nt << 4), v_a, v_r));
            }
            mma_bf16(accumulator[nt][0], accumulator[nt][1],
                     accumulator[nt][2], accumulator[nt][3],
                     probability_shared[lane][step][0], probability_shared[lane][step][1],
                     probability_shared[lane][step][2], probability_shared[lane][step][3],
                     vf[current][0], vf[current][1]);
            mma_bf16(accumulator[nt + 1][0], accumulator[nt + 1][1],
                     accumulator[nt + 1][2], accumulator[nt + 1][3],
                     probability_shared[lane][step][0], probability_shared[lane][step][1],
                     probability_shared[lane][step][2], probability_shared[lane][step][3],
                     vf[current][2], vf[current][3]);
        }
    }

    if (warp == 0) {
        denominator0 = ninfer::ops::warp_sum<4>(denominator0, 0xffffffffU);
        denominator1 = ninfer::ops::warp_sum<4>(denominator1, 0xffffffffU);
        inverse_shared[lane][0] = denominator0 > 0.0F ? __frcp_rn(denominator0) : 0.0F;
        inverse_shared[lane][1] = denominator1 > 0.0F ? __frcp_rn(denominator1) : 0.0F;
    }
    __syncthreads();
#pragma unroll
    for (int nt = 0; nt < kPvTiles; ++nt) {
        const int d = warp * (kHeadDim / kValueWarps) + nt * 8 + 2 * lid;
        if (gid < kHeadsPerKv) {
            const int head = kv_head * kHeadsPerKv + gid;
            if (num_splits == 1) {
                *reinterpret_cast<unsigned*>(&output[d + kHeadDim *
                    (head + static_cast<std::int64_t>(kQueryHeads) * token)]) =
                    pack_bf16x2(accumulator[nt][0] * inverse_shared[lane][0],
                                accumulator[nt][1] * inverse_shared[lane][0]);
            } else {
                const std::int64_t stats = head + static_cast<std::int64_t>(kQueryHeads) *
                    (token + static_cast<std::int64_t>(tokens) * split);
                const std::int64_t base = kHeadDim * stats;
                partial_numerator[base + d] = accumulator[nt][0];
                partial_numerator[base + d + 1] = accumulator[nt][1];
            }
        }
        if (gid + 8 < kHeadsPerKv) {
            const int head = kv_head * kHeadsPerKv + gid + 8;
            if (num_splits == 1) {
                *reinterpret_cast<unsigned*>(&output[d + kHeadDim *
                    (head + static_cast<std::int64_t>(kQueryHeads) * token)]) =
                    pack_bf16x2(accumulator[nt][2] * inverse_shared[lane][1],
                                accumulator[nt][3] * inverse_shared[lane][1]);
            } else {
                const std::int64_t stats = head + static_cast<std::int64_t>(kQueryHeads) *
                    (token + static_cast<std::int64_t>(tokens) * split);
                const std::int64_t base = kHeadDim * stats;
                partial_numerator[base + d] = accumulator[nt][2];
                partial_numerator[base + d + 1] = accumulator[nt][3];
            }
        }
    }
    if (num_splits > 1 && warp == 0 && lid == 0) {
        const std::int64_t split_base = static_cast<std::int64_t>(kQueryHeads) *
            (token + static_cast<std::int64_t>(tokens) * split);
        if (gid < kHeadsPerKv) {
            const int head = kv_head * kHeadsPerKv + gid;
            partial_maximum[split_base + head] = maximum0 * (1.0F / 16.0F);
            partial_denominator[split_base + head] = denominator0;
        }
        if (gid + 8 < kHeadsPerKv) {
            const int head = kv_head * kHeadsPerKv + gid + 8;
            partial_maximum[split_base + head] = maximum1 * (1.0F / 16.0F);
            partial_denominator[split_base + head] = denominator1;
        }
    }
}

__launch_bounds__(128, 1) __global__ void selected_attention_batched_bf16_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_pages,
    const __nv_bfloat16* value_pages, const int* indices, const int* tables,
    int logical_pages, const int* cache_positions, const int* valid_columns,
    const int* table_rows, int width, int tokens, int num_splits,
    float* partial_maximum, float* partial_denominator, float* partial_numerator,
    __nv_bfloat16* output) {
    selected_attention_batched_body<false>(
        query, key_pages, value_pages, nullptr, nullptr, indices, tables, logical_pages,
        cache_positions, valid_columns, table_rows, width, tokens, num_splits, partial_maximum,
        partial_denominator, partial_numerator, output);
}

__launch_bounds__(128, 1) __global__ void selected_attention_batched_fp8_kernel(
    const __nv_bfloat16* query, const std::uint8_t* key_pages,
    const std::uint8_t* value_pages, const __half* key_scales, const __half* value_scales,
    const int* indices, const int* tables, int logical_pages, const int* cache_positions,
    const int* valid_columns, const int* table_rows, int width, int tokens, int num_splits,
    float* partial_maximum, float* partial_denominator, float* partial_numerator,
    __nv_bfloat16* output) {
    selected_attention_batched_body<true>(
        query, key_pages, value_pages, key_scales, value_scales, indices, tables, logical_pages,
        cache_positions, valid_columns, table_rows, width, tokens, num_splits, partial_maximum,
        partial_denominator, partial_numerator, output);
}

constexpr int kMaxDecodeAttentionSplits = 64;
constexpr int kSplitHeadsPerBlock = 4;
static_assert(kSplitHeadsPerBlock * 32 == 128, "split staging assumes 128 threads");

// Match the split-K profiles used by the reference vLLM Triton kernel for this
// target's two KV heads. In particular, four-row MTP verification uses 32
// splits rather than the single-token decode profile's 64 splits.
int decode_attention_splits(int tokens) {
    const int base_programs = tokens * kKvHeads;
    if (base_programs <= 4) { return 64; }
    if (base_programs < 32) { return 32; }
    if (base_programs <= 256) { return 8; }
    if (base_programs <= 512) { return 4; }
    return 1;
}

template <bool Fp8>
__device__ __forceinline__ void selected_attention_split_body(
    const __nv_bfloat16* query, const void* key_pages, const void* value_pages,
    const __half* key_scales, const __half* value_scales,
    const int* indices, const int* tables,
    int logical_pages, const int* cache_positions, const int* valid_columns,
    const int* table_rows, int width, int tokens, float* partial_maximum,
    float* partial_denominator, float* partial_numerator, int num_splits) {
    constexpr int kTile = 16;
    constexpr float kLog2E = 1.4426950408889634F;
    const int head_group = static_cast<int>(blockIdx.x);
    const int head = head_group * kSplitHeadsPerBlock +
                     (static_cast<int>(threadIdx.x) >> 5);
    const int kv_head = head / (kQueryHeads / kKvHeads);
    const int token = static_cast<int>(blockIdx.y);
    const int split = static_cast<int>(blockIdx.z);
    const int lane_id = static_cast<int>(threadIdx.x) & 31;
    const int batch_lane = token / width;
    const int column = token - batch_lane * width;
    const std::int64_t stats_offset = head + static_cast<std::int64_t>(kQueryHeads) *
        (token + static_cast<std::int64_t>(tokens) * split);
    const std::int64_t numerator_base = kHeadDim * stats_offset;
    if (column >= valid_columns[batch_lane]) {
        if (lane_id == 0) {
            partial_maximum[stats_offset] = -INFINITY;
            partial_denominator[stats_offset] = 0.0F;
        }
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            partial_numerator[numerator_base + lane_id + 32 * item] = 0.0F;
        }
        return;
    }

    float q[8];
    float numerator[8] = {};
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int d = lane_id + 32 * item;
        q[item] = __bfloat162float(query[d + kHeadDim *
            (head + static_cast<std::int64_t>(kQueryHeads) * token)]);
    }
    float maximum = -INFINITY;
    float denominator = 0.0F;
    const int sequence_length = cache_positions[token] + 1;
    const int item_count = indices == nullptr ? sequence_length : kOutputWidth;
    // Partition whole 16-key tiles exactly as vLLM does. Dividing the raw item
    // count starts most splits at a different key and changes the reduction tree.
    const int num_tiles = (item_count + kTile - 1) / kTile;
    const int item_begin = (split * num_tiles / num_splits) * kTile;
    const int item_end = min(item_count,
                             ((split + 1) * num_tiles / num_splits) * kTile);
    const int table_row = table_rows[batch_lane];
    __shared__ __align__(16) __nv_bfloat16 staged_key[kTile][kHeadDim];
    __shared__ __align__(16) __nv_bfloat16 staged_value[kTile][kHeadDim];
    __shared__ int staged_positions[kTile];
    __shared__ int staged_rows[kTile];  // paged KV row of each key, -1 when absent
    for (int tile_begin = item_begin; tile_begin < item_end; tile_begin += kTile) {
        const int tile_size = min(kTile, item_end - tile_begin);
        bool present = false;
        if (threadIdx.x < tile_size) {
            const int position = indices == nullptr
                ? tile_begin + static_cast<int>(threadIdx.x)
                : indices[tile_begin + static_cast<int>(threadIdx.x) +
                          static_cast<std::int64_t>(kOutputWidth) * token];
            present = position >= 0;
            staged_positions[threadIdx.x] = position;
            staged_rows[threadIdx.x] = present
                ? position % kPagedKVPageSize +
                      kPagedKVPageSize *
                          (kv_head + kKvHeads * physical_page(tables, logical_pages, table_row,
                                                              position))
                : -1;
        }
        // A tile without selected keys contributes nothing (its maximum stays -inf).
        if (!__syncthreads_or(present)) { continue; }
        if constexpr (Fp8) {
            for (int element = static_cast<int>(threadIdx.x);
                 element < tile_size * kHeadDim; element += static_cast<int>(blockDim.x)) {
                const int item = element / kHeadDim;
                const int d = element - item * kHeadDim;
                const int row = staged_rows[item];
                if (row >= 0) {
                    const std::int64_t offset = d + static_cast<std::int64_t>(kHeadDim) * row;
                    staged_key[item][d] = __float2bfloat16_rn(
                        kv_cache_fp8_dequant_code_to_float(
                            static_cast<const std::uint8_t*>(key_pages)[offset],
                            key_scales[row]));
                    staged_value[item][d] = __float2bfloat16_rn(
                        kv_cache_fp8_dequant_code_to_float(
                            static_cast<const std::uint8_t*>(value_pages)[offset],
                            value_scales[row]));
                } else {
                    staged_key[item][d] = __float2bfloat16_rn(0.0F);
                    staged_value[item][d] = __float2bfloat16_rn(0.0F);
                }
            }
        } else {
            // Eight BF16 per 16-byte load; all loads of a thread are independent.
            constexpr int kVectors = kHeadDim / 8;
#pragma unroll
            for (int i = 0; i < kTile * kVectors / 128; ++i) {
                const int vector = static_cast<int>(threadIdx.x) + 128 * i;
                const int item = vector / kVectors;
                const int d = 8 * (vector % kVectors);
                const int row = item < tile_size ? staged_rows[item] : -1;
                uint4 key = make_uint4(0, 0, 0, 0);
                uint4 value = make_uint4(0, 0, 0, 0);
                if (row >= 0) {
                    const std::int64_t offset = d + static_cast<std::int64_t>(kHeadDim) * row;
                    key = *reinterpret_cast<const uint4*>(
                        static_cast<const __nv_bfloat16*>(key_pages) + offset);
                    value = *reinterpret_cast<const uint4*>(
                        static_cast<const __nv_bfloat16*>(value_pages) + offset);
                }
                *reinterpret_cast<uint4*>(&staged_key[item][d]) = key;
                *reinterpret_cast<uint4*>(&staged_value[item][d]) = value;
            }
        }
        __syncthreads();
        float scores[kTile];
        float tile_maximum = -INFINITY;
        for (int position = 0; position < tile_size; ++position) {
            float score = 0.0F;
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                score += q[item] *
                         __bfloat162float(staged_key[position][lane_id + 32 * item]);
            }
            for (int offset = 16; offset != 0; offset >>= 1) {
                score += __shfl_down_sync(0xffffffffU, score, offset);
            }
            score = staged_positions[position] >= 0
                ? __shfl_sync(0xffffffffU, score, 0) * (1.0F / 16.0F)
                : -INFINITY;
            scores[position] = score;
            tile_maximum = fmaxf(tile_maximum, score);
        }
        if (tile_maximum == -INFINITY) {
            __syncthreads();
            continue;
        }
        const float next_maximum = fmaxf(maximum, tile_maximum);
        const float scaled_next_maximum = next_maximum * kLog2E;
        const float previous_scale = exp2_approx(
            __fmaf_rn(maximum, kLog2E, -scaled_next_maximum));
        denominator *= previous_scale;
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            numerator[item] *= previous_scale;
        }
        for (int position = 0; position < tile_size; ++position) {
            if (staged_positions[position] < 0) { continue; }
            const float current_scale = exp2_approx(
                __fmaf_rn(scores[position], kLog2E, -scaled_next_maximum));
            const float current_probability = __bfloat162float(
                __float2bfloat16_rn(current_scale));
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                numerator[item] += current_probability *
                    __bfloat162float(staged_value[position][lane_id + 32 * item]);
            }
            denominator += current_scale;
        }
        maximum = next_maximum;
        __syncthreads();
    }
    if (lane_id == 0) {
        partial_maximum[stats_offset] = maximum;
        partial_denominator[stats_offset] = denominator;
    }
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        partial_numerator[numerator_base + lane_id + 32 * item] = numerator[item];
    }
}

__global__ void selected_attention_split_bf16_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_pages,
    const __nv_bfloat16* value_pages, const int* indices, const int* tables,
    int logical_pages, const int* cache_positions, const int* valid_columns,
    const int* table_rows, int width, int tokens, float* partial_maximum,
    float* partial_denominator, float* partial_numerator, int num_splits) {
    selected_attention_split_body<false>(
        query, key_pages, value_pages, nullptr, nullptr, indices, tables, logical_pages,
        cache_positions, valid_columns, table_rows, width, tokens, partial_maximum,
        partial_denominator, partial_numerator, num_splits);
}

__global__ void selected_attention_split_fp8_kernel(
    const __nv_bfloat16* query, const std::uint8_t* key_pages,
    const std::uint8_t* value_pages, const __half* key_scales, const __half* value_scales,
    const int* indices, const int* tables, int logical_pages, const int* cache_positions,
    const int* valid_columns, const int* table_rows, int width, int tokens,
    float* partial_maximum, float* partial_denominator, float* partial_numerator, int num_splits) {
    selected_attention_split_body<true>(
        query, key_pages, value_pages, key_scales, value_scales, indices, tables, logical_pages,
        cache_positions, valid_columns, table_rows, width, tokens, partial_maximum,
        partial_denominator, partial_numerator, num_splits);
}

// One CTA per (query head, token), one thread per head dimension. Every element combines the
// splits in ascending order, so the result does not depend on the launch shape.
__global__ void __launch_bounds__(kHeadDim) reduce_selected_attention_splits_kernel(
    const float* partial_maximum, const float* partial_denominator,
    const float* partial_numerator, const int* valid_columns, int width, int tokens,
    __nv_bfloat16* output, int num_splits) {
    const int head = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int d = static_cast<int>(threadIdx.x);
    const int batch_lane = token / width;
    const int column = token - batch_lane * width;
    const std::int64_t output_index =
        d + kHeadDim * (head + static_cast<std::int64_t>(kQueryHeads) * token);
    if (column >= valid_columns[batch_lane]) {
        output[output_index] = __float2bfloat16_rn(0.0F);
        return;
    }
    constexpr float kLog2E = 1.4426950408889634F;
    const std::int64_t split_stride = static_cast<std::int64_t>(kQueryHeads) * tokens;
    const std::int64_t stats_base = head + static_cast<std::int64_t>(kQueryHeads) * token;
    // Per-split rescale factors are shared by the head's 256 dimensions; empty splits get 0.
    __shared__ float split_scale[kMaxDecodeAttentionSplits];
    __shared__ float shared_denominator;
    if (d < 32) {
        float maximum = -INFINITY;
        for (int split = d; split < num_splits; split += 32) {
            maximum = fmaxf(maximum, partial_maximum[stats_base + split_stride * split]);
        }
        for (int offset = 16; offset != 0; offset >>= 1) {
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, offset));
        }
        for (int split = d; split < num_splits; split += 32) {
            const std::int64_t stats_offset = stats_base + split_stride * split;
            split_scale[split] = partial_denominator[stats_offset] == 0.0F
                ? 0.0F
                : exp2_approx(__fmaf_rn(partial_maximum[stats_offset], kLog2E,
                                        -maximum * kLog2E));
        }
        __syncwarp();
        if (d == 0) {
            float denominator = 0.0F;
            for (int split = 0; split < num_splits; ++split) {
                const float scale = split_scale[split];
                if (scale != 0.0F) {
                    denominator +=
                        partial_denominator[stats_base + split_stride * split] * scale;
                }
            }
            shared_denominator = denominator;
        }
    }
    __syncthreads();
    float numerator = 0.0F;
#pragma unroll 16
    for (int split = 0; split < num_splits; ++split) {
        const float value = partial_numerator[kHeadDim * (stats_base + split_stride * split) + d];
        const float scale = split_scale[split];
        if (scale != 0.0F) { numerator += value * scale; }
    }
    const float denominator = shared_denominator;
    output[output_index] = __float2bfloat16_rn(numerator / denominator);
}

void require_weight(const Weight& weight, int n, int k, const char* label) {
    if (weight.qtype != QType::BF16 || weight.layout != QuantLayout::Contiguous ||
        weight.qdata == nullptr || weight.n != n || weight.k != k) {
        throw std::invalid_argument(label);
    }
}

// Q/K/V/O projections may also be row-scaled FP8 (8-bit projection artifacts); linear dispatches.
void require_projection(const Weight& weight, int n, int k, const char* label) {
    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        if (weight.layout != QuantLayout::RowScale || weight.qdata == nullptr || weight.n != n ||
            weight.k != k) {
            throw std::invalid_argument(label);
        }
        return;
    }
    require_weight(weight, n, k, label);
}

} // namespace

void flash_next_expand_text_positions(const Tensor& positions, Tensor& mrope_positions,
                                      cudaStream_t stream) {
    const int width = positions.ne[0];
    const int batch = positions.ne[1];
    const int tokens = width * batch;
    if (tokens <= 0 || positions.dtype != DType::I32 || !positions.is_contiguous() ||
        positions.ne[2] != 1 || positions.ne[3] != 1 ||
        mrope_positions.dtype != DType::I32 || !mrope_positions.is_contiguous() ||
        mrope_positions.ne[0] != width || mrope_positions.ne[1] != batch ||
        mrope_positions.ne[2] != 3 || mrope_positions.ne[3] != 1) {
        throw std::invalid_argument("flash_next_expand_text_positions: invalid geometry");
    }
    expand_text_positions_kernel<<<(3 * tokens + 255) / 256, 256, 0, stream>>>(
        static_cast<const int*>(positions.data), static_cast<int*>(mrope_positions.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t flash_next_qsa_workspace_capacity_bytes(std::int32_t tokens,
                                                     std::uint32_t max_context) {
    if (tokens <= 0 || max_context == 0) {
        throw std::invalid_argument("Flash-Next QSA dimensions must be positive");
    }
    const std::uint64_t bf16 = static_cast<std::uint64_t>(tokens) * 39552ULL *
                               sizeof(__nv_bfloat16);
    const std::uint64_t groups = (static_cast<std::uint64_t>(max_context) + kRatio - 1) / kRatio;
    const std::uint64_t hierarchy = tokens <= 16
        ? 2 * ((groups + kTopkBlockItems - 1) / kTopkBlockItems) * kTopGroups *
              (sizeof(float) + sizeof(std::int32_t))
        : 0;
    const std::uint64_t selection = static_cast<std::uint64_t>(tokens) *
                                    (groups * sizeof(float) +
                                     kTopGroups * sizeof(std::int32_t) + hierarchy);
    const std::uint64_t indices = static_cast<std::uint64_t>(tokens) * kOutputWidth *
                                  sizeof(std::int32_t);
    const std::uint64_t split_partials = tokens <= 16
        ? static_cast<std::uint64_t>(tokens) * kMaxDecodeAttentionSplits * kQueryHeads *
              (kHeadDim + 2) * sizeof(float)
        : 0;
    const std::uint64_t total = bf16 + indices + selection + split_partials + 16 * 256;
    if (total > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("Flash-Next QSA workspace size overflow");
    }
    return static_cast<std::size_t>(total);
}

std::size_t flash_next_query_gate_workspace_capacity_bytes(std::int32_t tokens) {
    if (tokens <= 0) {
        throw std::invalid_argument("Flash-Next query/gate dimensions must be positive");
    }
    return static_cast<std::size_t>(tokens) * 2 * kQueryHeads * kHeadDim *
           sizeof(__nv_bfloat16) + 256;
}

void flash_next_project_query_gate(const Tensor& input, const Weight& query_gate,
                                   Tensor& query, Tensor& gate,
                                   WorkspaceArena& workspace, cudaStream_t stream,
                                   Bf16GemmContext* bf16_gemm) {
    const int tokens = input.ne[1];
    if (input.dtype != DType::BF16 || !input.is_contiguous() || input.ne[0] != 2560 ||
        tokens <= 0 || query.dtype != DType::BF16 || !query.is_contiguous() ||
        query.numel() != static_cast<std::int64_t>(kQueryHeads) * kHeadDim * tokens ||
        gate.dtype != DType::BF16 || !gate.is_contiguous() || gate.numel() != query.numel()) {
        throw std::invalid_argument("flash_next_project_query_gate: invalid tensor geometry");
    }
    require_projection(query_gate, 12288, 2560,
                       "flash_next_project_query_gate: invalid packed weight");
    if (tokens == 1 && query_gate.qtype == QType::BF16) {
        detail::flash_next::launch_bf16_query_gate_decode(input, query_gate, query, gate, stream);
        return;
    }
    auto scope = workspace.scope();
    Tensor packed = workspace.alloc(DType::BF16, {12288, tokens});
    linear(input, query_gate, packed, stream, bf16_gemm);
    const int blocks = static_cast<int>(std::min<std::int64_t>(
        4096, (query.numel() + 255) / 256));
    split_query_gate_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(packed.data),
        static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(gate.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_qsa(const Tensor& input, const Tensor& cache_positions,
                    const Tensor& rope_positions, const Tensor& valid_columns,
                    const Tensor& table_rows, const FlashNextQsaWeights& weights,
                    PagedKVBatchLayerView cache, CausalAttentionExecutionEnvelope envelope,
                    Tensor& destination,
                    WorkspaceArena& workspace, cudaStream_t stream,
                    Bf16GemmContext* bf16_gemm,
                    FlashNextQsaIndexControl index_control) {
    NINFER_PERF_SCOPE(index_control.reused_indices != nullptr ? "qsa.reuse" : "qsa.select",
                       input.ne[1], cache_positions.ne[1], envelope.max_visible_keys,
                       flash_next_work::qsa(input.ne[1], index_control.reused_indices != nullptr));

    const int width = cache_positions.ne[0];
    const int batch = cache_positions.ne[1];
    const int tokens = width * batch;
    if (input.dtype != DType::BF16 || !input.is_contiguous() || input.ne[0] != 2560 ||
        input.ne[1] != tokens || destination.dtype != DType::BF16 || !destination.is_contiguous() ||
        destination.ne[0] != 2560 || destination.ne[1] != tokens) {
        throw std::invalid_argument("flash_next_qsa: invalid input/output geometry");
    }
    if (cache_positions.dtype != DType::I32 || !cache_positions.is_contiguous() ||
        rope_positions.dtype != DType::I32 || !rope_positions.is_contiguous() ||
        rope_positions.ne[0] != width || rope_positions.ne[1] != batch ||
        rope_positions.ne[2] != 3 || valid_columns.dtype != DType::I32 ||
        valid_columns.ne[0] != batch || table_rows.dtype != DType::I32 ||
        table_rows.ne[0] != batch) {
        throw std::invalid_argument("flash_next_qsa: invalid position geometry");
    }
    const auto valid_indices = [tokens](const Tensor* tensor) {
        return tensor == nullptr ||
               (tensor->dtype == DType::I32 && tensor->is_contiguous() &&
                tensor->numel() == static_cast<std::int64_t>(kOutputWidth) * tokens);
    };
    if (!valid_indices(index_control.selected_indices) ||
        !valid_indices(index_control.reused_indices) ||
        (index_control.selected_indices != nullptr && index_control.reused_indices != nullptr)) {
        throw std::invalid_argument("flash_next_qsa: invalid index-sharing control");
    }
    const bool fp8_cache = cache.storage == KvCacheStorage::Fp8E4M3Row256;
    const bool bf16_cache = cache.storage == KvCacheStorage::BFloat16KeyValue;
    const bool valid_primary =
        (bf16_cache && cache.k_pages.dtype == DType::BF16 &&
         cache.v_pages.dtype == DType::BF16 && cache.k_scale_pages.data == nullptr &&
         cache.v_scale_pages.data == nullptr) ||
        (fp8_cache && cache.k_pages.dtype == DType::FP8_E4M3FN &&
         cache.v_pages.dtype == DType::FP8_E4M3FN &&
         cache.k_scale_pages.dtype == DType::FP16 && cache.v_scale_pages.dtype == DType::FP16 &&
         cache.k_scale_pages.ne[0] == 1 && cache.v_scale_pages.ne[0] == 1);
    if (!valid_primary || cache.k_pages.ne[0] != kHeadDim ||
        cache.v_pages.ne[0] != kHeadDim ||
        cache.k_pages.ne[2] != kKvHeads || cache.v_pages.ne[2] != kKvHeads) {
        std::ostringstream message;
        message << "flash_next_qsa: invalid primary cache geometry: k dtype="
                << static_cast<int>(cache.k_pages.dtype) << " shape=[" << cache.k_pages.ne[0]
                << ',' << cache.k_pages.ne[1] << ',' << cache.k_pages.ne[2] << ','
                << cache.k_pages.ne[3] << "] v dtype="
                << static_cast<int>(cache.v_pages.dtype) << " shape=[" << cache.v_pages.ne[0]
                << ',' << cache.v_pages.ne[1] << ',' << cache.v_pages.ne[2] << ','
                << cache.v_pages.ne[3] << ']';
        throw std::invalid_argument(message.str());
    }
    if (cache.auxiliary_pages.size() != 2 ||
        cache.auxiliary_pages[0].dtype != DType::BF16 ||
        cache.auxiliary_pages[0].ne[0] != kIndexDim ||
        cache.auxiliary_pages[1].dtype != DType::I32 ||
        cache.auxiliary_pages[1].ne[0] != 3) {
        throw std::invalid_argument("flash_next_qsa: invalid auxiliary cache geometry");
    }
    require_projection(weights.query_gate, 12288, 2560,
                       "flash_next_qsa: invalid packed query/gate weight");
    require_projection(weights.key, 512, 2560, "flash_next_qsa: invalid key weight");
    require_projection(weights.value, 512, 2560, "flash_next_qsa: invalid value weight");
    require_projection(weights.output, 2560, 6144, "flash_next_qsa: invalid output weight");
    require_weight(weights.index_query, 512, 2560, "flash_next_qsa: invalid index query weight");
    require_weight(weights.index_key, 128, 2560, "flash_next_qsa: invalid index key weight");
    auto scope = workspace.scope();
    Tensor query = workspace.alloc(DType::BF16, {6144, tokens});
    Tensor gate = workspace.alloc(DType::BF16, {6144, tokens});
    Tensor key = workspace.alloc(DType::BF16, {512, tokens});
    Tensor value = workspace.alloc(DType::BF16, {512, tokens});
    flash_next_project_query_gate(input, weights.query_gate, query, gate, workspace, stream,
                                  bf16_gemm);
    linear(input, weights.key, key, stream, bf16_gemm);
    linear(input, weights.value, value, stream, bf16_gemm);
    Tensor normalized_query = workspace.alloc(DType::BF16, {6144, tokens});
    Tensor normalized_key = workspace.alloc(DType::BF16, {512, tokens});
    Tensor query_heads = query.view({kHeadDim, kQueryHeads, tokens});
    Tensor key_heads = key.view({kHeadDim, kKvHeads, tokens});
    Tensor normalized_query_heads = normalized_query.view({kHeadDim, kQueryHeads, tokens});
    Tensor normalized_key_heads = normalized_key.view({kHeadDim, kKvHeads, tokens});
    Tensor rope_view = rope_positions.view({tokens, 3});
    rmsnorm(query_heads, weights.query_norm, 1.0e-6F, true, normalized_query_heads, stream);
    rmsnorm(key_heads, weights.key_norm, 1.0e-6F, true, normalized_key_heads, stream);
    rope(rope_view, 64, kTheta, normalized_query_heads, normalized_key_heads, stream);
    if (fp8_cache) {
        hadamard_rows_kernel<<<tokens * kQueryHeads, 32, 0, stream>>>(
            static_cast<__nv_bfloat16*>(normalized_query.data), tokens * kQueryHeads);
    }
    Tensor index_key = workspace.alloc(DType::BF16, {128, tokens});
    linear(input, weights.index_key, index_key, stream, bf16_gemm);
    Tensor index_query;
    if (index_control.reused_indices == nullptr) {
        Tensor index_query_projected = workspace.alloc(DType::BF16, {512, tokens});
        linear(input, weights.index_query, index_query_projected, stream, bf16_gemm);
        index_query = workspace.alloc(DType::BF16, {512, tokens});
        prepare_index_query_kernel<<<dim3(kIndexHeads, tokens), kIndexDim, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(index_query_projected.data),
            static_cast<const __nv_bfloat16*>(weights.index_query_norm.data),
            static_cast<const int*>(rope_view.data), tokens,
            static_cast<__nv_bfloat16*>(index_query.data));
    }
    const int logical_pages = cache.block_tables.ne[0];
    if (fp8_cache) {
        append_cache_fp8_kernel<<<tokens, 64, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(normalized_key.data),
            static_cast<const __nv_bfloat16*>(value.data),
            static_cast<const __nv_bfloat16*>(index_key.data),
            static_cast<const int*>(cache_positions.data), static_cast<const int*>(rope_view.data),
            static_cast<const int*>(valid_columns.data), static_cast<const int*>(table_rows.data),
            width, batch, logical_pages, static_cast<const int*>(cache.block_tables.data),
            static_cast<std::uint8_t*>(cache.k_pages.data),
            static_cast<std::uint8_t*>(cache.v_pages.data),
            static_cast<__half*>(cache.k_scale_pages.data),
            static_cast<__half*>(cache.v_scale_pages.data),
            static_cast<__nv_bfloat16*>(cache.auxiliary_pages[0].data),
            static_cast<int*>(cache.auxiliary_pages[1].data));
    } else {
        append_cache_kernel<<<tokens, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(normalized_key.data),
            static_cast<const __nv_bfloat16*>(value.data),
            static_cast<const __nv_bfloat16*>(index_key.data),
            static_cast<const int*>(cache_positions.data), static_cast<const int*>(rope_view.data),
            static_cast<const int*>(valid_columns.data), static_cast<const int*>(table_rows.data),
            width, batch, logical_pages, static_cast<const int*>(cache.block_tables.data),
            static_cast<__nv_bfloat16*>(cache.k_pages.data),
            static_cast<__nv_bfloat16*>(cache.v_pages.data),
            static_cast<__nv_bfloat16*>(cache.auxiliary_pages[0].data),
            static_cast<int*>(cache.auxiliary_pages[1].data));
    }
    compress_index_groups_kernel<<<tokens, kIndexDim, 0, stream>>>(
        static_cast<__nv_bfloat16*>(cache.auxiliary_pages[0].data),
        static_cast<const int*>(cache.auxiliary_pages[1].data),
        static_cast<const __nv_bfloat16*>(weights.index_key_norm.data),
        static_cast<const int*>(cache_positions.data),
        static_cast<const int*>(valid_columns.data), static_cast<const int*>(table_rows.data),
        width, batch, logical_pages, static_cast<const int*>(cache.block_tables.data));
    if (envelope.max_visible_keys == 0 ||
        envelope.max_visible_keys > static_cast<std::uint32_t>(logical_pages * kPagedKVPageSize)) {
        throw std::invalid_argument("flash_next_qsa: invalid visibility envelope");
    }
    Tensor attention = workspace.alloc(DType::BF16, {6144, tokens});
    const auto launch_batched = [&]<bool Fp8>(std::bool_constant<Fp8>, dim3 grid,
                                              const int* indices, int num_splits,
                                              float* partial_maximum,
                                              float* partial_denominator,
                                              float* partial_numerator,
                                              __nv_bfloat16* output) {
        if constexpr (Fp8) {
            static const cudaError_t configured = cudaFuncSetAttribute(
                selected_attention_batched_fp8_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, kPrefillSharedBytes);
            CUDA_CHECK(configured);
            selected_attention_batched_fp8_kernel<<<grid, 128, kPrefillSharedBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(normalized_query.data),
                static_cast<const std::uint8_t*>(cache.k_pages.data),
                static_cast<const std::uint8_t*>(cache.v_pages.data),
                static_cast<const __half*>(cache.k_scale_pages.data),
                static_cast<const __half*>(cache.v_scale_pages.data), indices,
                static_cast<const int*>(cache.block_tables.data), logical_pages,
                static_cast<const int*>(cache_positions.data),
                static_cast<const int*>(valid_columns.data),
                static_cast<const int*>(table_rows.data), width, tokens, num_splits,
                partial_maximum, partial_denominator, partial_numerator, output);
        } else {
            static const cudaError_t configured = cudaFuncSetAttribute(
                selected_attention_batched_bf16_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, kPrefillSharedBytes);
            CUDA_CHECK(configured);
            selected_attention_batched_bf16_kernel<<<grid, 128, kPrefillSharedBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(normalized_query.data),
                static_cast<const __nv_bfloat16*>(cache.k_pages.data),
                static_cast<const __nv_bfloat16*>(cache.v_pages.data), indices,
                static_cast<const int*>(cache.block_tables.data), logical_pages,
                static_cast<const int*>(cache_positions.data),
                static_cast<const int*>(valid_columns.data),
                static_cast<const int*>(table_rows.data), width, tokens, num_splits,
                partial_maximum, partial_denominator, partial_numerator, output);
        }
    };
    const auto dispatch_batched = [&](dim3 grid, const int* indices, int num_splits,
                                      float* partial_maximum, float* partial_denominator,
                                      float* partial_numerator, __nv_bfloat16* output) {
        if (fp8_cache) {
            launch_batched(std::true_type{}, grid, indices, num_splits, partial_maximum,
                           partial_denominator, partial_numerator, output);
        } else {
            launch_batched(std::false_type{}, grid, indices, num_splits, partial_maximum,
                           partial_denominator, partial_numerator, output);
        }
    };
    const auto launch_split = [&]<bool Fp8>(std::bool_constant<Fp8>, const int* indices,
                                            float* partial_maximum,
                                            float* partial_denominator,
                                            float* partial_numerator, int num_splits) {
        const dim3 grid(kQueryHeads / kSplitHeadsPerBlock, tokens, num_splits);
        if constexpr (Fp8) {
            selected_attention_split_fp8_kernel<<<grid, kSplitHeadsPerBlock * 32, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(normalized_query.data),
                static_cast<const std::uint8_t*>(cache.k_pages.data),
                static_cast<const std::uint8_t*>(cache.v_pages.data),
                static_cast<const __half*>(cache.k_scale_pages.data),
                static_cast<const __half*>(cache.v_scale_pages.data), indices,
                static_cast<const int*>(cache.block_tables.data), logical_pages,
                static_cast<const int*>(cache_positions.data),
                static_cast<const int*>(valid_columns.data),
                static_cast<const int*>(table_rows.data), width, tokens,
                partial_maximum, partial_denominator, partial_numerator, num_splits);
        } else {
            selected_attention_split_bf16_kernel<<<grid, kSplitHeadsPerBlock * 32, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(normalized_query.data),
                static_cast<const __nv_bfloat16*>(cache.k_pages.data),
                static_cast<const __nv_bfloat16*>(cache.v_pages.data), indices,
                static_cast<const int*>(cache.block_tables.data), logical_pages,
                static_cast<const int*>(cache_positions.data),
                static_cast<const int*>(valid_columns.data),
                static_cast<const int*>(table_rows.data), width, tokens,
                partial_maximum, partial_denominator, partial_numerator, num_splits);
        }
    };
    const auto dispatch_split = [&](const int* indices, float* partial_maximum,
                                    float* partial_denominator, float* partial_numerator,
                                    int num_splits) {
        if (fp8_cache) {
            launch_split(std::true_type{}, indices, partial_maximum, partial_denominator,
                         partial_numerator, num_splits);
        } else {
            launch_split(std::false_type{}, indices, partial_maximum, partial_denominator,
                         partial_numerator, num_splits);
        }
    };
    if (tokens > 16 && envelope.max_visible_keys <= kOutputWidth) {
        dispatch_batched(dim3(tokens, kKvHeads), nullptr, 1, nullptr, nullptr, nullptr,
                         static_cast<__nv_bfloat16*>(attention.data));
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Decode keeps a fixed launch extent across graph profiles.  Prompt
        // work uses only the causally reachable group frontier.
        Tensor indices;
        if (index_control.reused_indices != nullptr) {
            indices = index_control.reused_indices->view({kOutputWidth, tokens});
        } else if (index_control.selected_indices != nullptr) {
            indices = index_control.selected_indices->view({kOutputWidth, tokens});
        } else {
            indices = workspace.alloc(DType::I32, {kOutputWidth, tokens});
        }
        const int score_stride = tokens == 1
            ? std::max(1, logical_pages * kPagedKVPageSize / kRatio)
            : std::max(1, static_cast<int>(envelope.max_visible_keys / kRatio));
        const int group_extent = std::max(1, static_cast<int>(
            envelope.max_visible_keys / kRatio));
        bool hierarchical_selection = false;
        if (index_control.reused_indices == nullptr) {
            Tensor scores = workspace.alloc(DType::FP32, {score_stride, tokens});
            // The scalar warp scorer wins for short decode frontiers. Once the compressed frontier
            // is large, its FP32 dot products dominate token latency; the padded tensor-core scorer
            // amortizes its fixed staging cost and is substantially faster even for one query.
            const bool tensor_score = tokens > 16 || group_extent >= 8192;
            if (tensor_score) {
                static const cudaError_t score_configured = cudaFuncSetAttribute(
                    score_groups_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    kScoreSharedBytes);
                CUDA_CHECK(score_configured);
                const int score_blocks =
                    (group_extent + kScoreColumns * kScoreTilesPerBlock - 1) /
                    (kScoreColumns * kScoreTilesPerBlock);
                score_groups_mma_kernel<<<dim3(score_blocks, tokens), 64,
                                          kScoreSharedBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(index_query.data),
                    static_cast<const __nv_bfloat16*>(cache.auxiliary_pages[0].data),
                    static_cast<const int*>(cache.block_tables.data), logical_pages,
                    static_cast<const int*>(cache_positions.data),
                    static_cast<const int*>(valid_columns.data),
                    static_cast<const int*>(table_rows.data), width, group_extent,
                    score_stride, static_cast<float*>(scores.data));
            } else {
                const int score_blocks = std::min(510, (group_extent + 7) / 8);
                score_groups_batched_kernel<<<dim3(score_blocks, tokens), 256, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(index_query.data),
                    static_cast<const __nv_bfloat16*>(cache.auxiliary_pages[0].data),
                    static_cast<const int*>(cache.block_tables.data), logical_pages,
                    static_cast<const int*>(cache_positions.data),
                    static_cast<const int*>(valid_columns.data),
                    static_cast<const int*>(table_rows.data), width, tokens, group_extent,
                    score_stride, static_cast<float*>(scores.data));
            }
            CUDA_CHECK(cudaGetLastError());
            Tensor selected_groups = workspace.alloc(DType::I32, {kTopGroups, tokens});
            hierarchical_selection = tokens <= 16 && group_extent >= 8192;
            // Persistent-top-k collection order is part of the multi-token
            // verification reduction profile. Single-token decode has no MTP
            // acceptance dependency on it and emits final indices directly.
            const bool persistent_collection_order = tokens > 1 && tokens <= 16;
            if (hierarchical_selection) {
            const int candidate_capacity =
                ((group_extent + kTopkBlockItems - 1) / kTopkBlockItems) * kTopGroups;
            Tensor candidate_scores[2] = {
                workspace.alloc(DType::FP32, {candidate_capacity, tokens}),
                workspace.alloc(DType::FP32, {candidate_capacity, tokens}),
            };
            Tensor candidate_indices[2] = {
                workspace.alloc(DType::I32, {candidate_capacity, tokens}),
                workspace.alloc(DType::I32, {candidate_capacity, tokens}),
            };
            const float* current_scores = static_cast<const float*>(scores.data);
            const int* current_indices = nullptr;
            int current_count = group_extent;
            int current_stride = score_stride;
            int stage = 0;
            while (current_count > kTopGroups) {
                const int blocks = (current_count + kTopkBlockItems - 1) / kTopkBlockItems;
                const int output_count = blocks * kTopGroups;
                const bool final = blocks == 1;
                float* next_scores = final
                    ? static_cast<float*>(scores.data)
                    : static_cast<float*>(candidate_scores[stage & 1].data);
                int* next_indices = final
                    ? static_cast<int*>(persistent_collection_order
                                            ? selected_groups.data
                                            : indices.data)
                    : static_cast<int*>(candidate_indices[stage & 1].data);
                const int output_stride = output_count;
                hierarchical_top_groups_kernel<<<dim3(blocks, tokens), kTopkBlockThreads, 0,
                                                  stream>>>(
                    current_scores, current_indices, current_count, current_stride,
                    next_scores, next_indices, output_stride,
                    static_cast<const int*>(cache_positions.data),
                    static_cast<const int*>(valid_columns.data), width, stage,
                    final && !persistent_collection_order);
                if (final) { break; }
                current_scores = next_scores;
                current_indices = next_indices;
                current_count = output_count;
                current_stride = output_count;
                ++stage;
            }
            } else {
                select_top_groups_kernel<<<tokens, 256, 0, stream>>>(
                    static_cast<const float*>(scores.data),
                    static_cast<const int*>(cache_positions.data), score_stride, group_extent,
                    static_cast<int*>(selected_groups.data));
            }
            CUDA_CHECK(cudaGetLastError());
            if (!hierarchical_selection || persistent_collection_order) {
                if (persistent_collection_order) {
                    order_groups_like_persistent_topk_kernel<<<tokens, kTopkBlockThreads, 0,
                                                               stream>>>(
                        static_cast<int*>(selected_groups.data),
                        static_cast<const int*>(cache_positions.data), group_extent);
                }
                expand_indices_batched_kernel<<<dim3((kOutputWidth + 255) / 256, tokens), 256, 0,
                                                    stream>>>(
                    static_cast<const int*>(selected_groups.data),
                    static_cast<const int*>(cache_positions.data),
                    static_cast<const int*>(valid_columns.data), width,
                    static_cast<int*>(indices.data));
            }
        }
        if (tokens <= 16) {
            const bool dense_within_budget = envelope.max_visible_keys <= kOutputWidth;
            const int num_splits = dense_within_budget ? kMaxDecodeAttentionSplits
                                                       : decode_attention_splits(tokens);
            Tensor partial_maximum = workspace.alloc(
                DType::FP32, {kQueryHeads, tokens, num_splits});
            Tensor partial_denominator = workspace.alloc(
                DType::FP32, {kQueryHeads, tokens, num_splits});
            Tensor partial_numerator = workspace.alloc(
                DType::FP32, {kHeadDim, kQueryHeads, tokens, num_splits});
            if (dense_within_budget) {
                dispatch_split(static_cast<const int*>(indices.data),
                               static_cast<float*>(partial_maximum.data),
                               static_cast<float*>(partial_denominator.data),
                               static_cast<float*>(partial_numerator.data), num_splits);
            } else {
                dispatch_batched(dim3(tokens, kKvHeads, num_splits),
                                 static_cast<const int*>(indices.data), num_splits,
                                 static_cast<float*>(partial_maximum.data),
                                 static_cast<float*>(partial_denominator.data),
                                 static_cast<float*>(partial_numerator.data),
                                 static_cast<__nv_bfloat16*>(attention.data));
            }
            reduce_selected_attention_splits_kernel<<<dim3(kQueryHeads, tokens), kHeadDim, 0,
                                                      stream>>>(
                static_cast<const float*>(partial_maximum.data),
                static_cast<const float*>(partial_denominator.data),
                static_cast<const float*>(partial_numerator.data),
                static_cast<const int*>(valid_columns.data), width, tokens,
                static_cast<__nv_bfloat16*>(attention.data), num_splits);
        } else {
            dispatch_batched(dim3(tokens, kKvHeads), static_cast<const int*>(indices.data), 1,
                             nullptr, nullptr, nullptr,
                             static_cast<__nv_bfloat16*>(attention.data));
        }
    }
    Tensor attention_heads = attention.view({kHeadDim, kQueryHeads, tokens});
    Tensor gate_heads = gate.view({kHeadDim, kQueryHeads, tokens});
    sigmoid_mul(gate_heads, attention_heads, stream);
    linear(attention, weights.output, destination, stream, bf16_gemm);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_qsa_select_indices(const Tensor& indices, const Tensor& selectors,
                                   Tensor& destination, cudaStream_t stream) {
    const int width = indices.ne[1];
    const int batch = indices.ne[2];
    if (width <= 0 || batch <= 0 || indices.dtype != DType::I32 || !indices.is_contiguous() ||
        indices.ne[0] != kOutputWidth || selectors.dtype != DType::I32 ||
        !selectors.is_contiguous() || selectors.ne[0] != batch ||
        destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.ne[0] != kOutputWidth || destination.ne[1] != batch) {
        throw std::invalid_argument("flash_next_qsa_select_indices: invalid geometry");
    }
    select_shared_indices_kernel<<<dim3((kOutputWidth + 255) / 256, batch), 256, 0, stream>>>(
        static_cast<const int*>(indices.data), static_cast<const int*>(selectors.data),
        static_cast<int*>(destination.data), width, batch);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
