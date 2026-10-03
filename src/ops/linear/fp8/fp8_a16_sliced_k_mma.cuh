#pragma once

// Small-T row-scaled E4M3 weight x BF16 activation Tensor Core mainloop.
//
// A CTA owns RowTiles sixteen-row tiles and splits K across compile-time-selected warps; with
// PairSplit > 1 the K warps of one row tile span PairSplit CTAs (Fp8A16SlicedKMmaSchedule).
// Persistent E4M3 codes are widened exactly to BF16 MMA operands; the represented BF16 row multiplier is applied
// once to the complete FP32 dot product. The public activation is never quantized. K needs only be
// a multiple of one warp's 64-column tile: when it is not a whole number of K groups, the last
// group stages and multiplies only the tiles inside K, and the warps past it keep their sums.

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/fp8/fp8_a16_codec.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/fp8/fp8_operands.h"
#include "ops/linear/fp8/fp8_shared.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <class Schedule, class Output, class Epilogue, class RowPolicy>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_a16_sliced_k_mma_kernel(
    Fp8A16Operands operands, Output output, Epilogue epilogue, RowPolicy row_policy,
    int token_offset, Fp8SlicedSplitScratch scratch) {
    const auto* __restrict__ x            = operands.x;
    const auto* __restrict__ weight_codes = operands.codes;
    const auto* __restrict__ row_scales   = operands.scales;
    const int kHidden                     = Schedule::kStaticK ? Schedule::kStaticK : operands.k;
    constexpr int ActiveTokens =
        Schedule::kTokenCapacity ? Schedule::kTokenCapacity : Schedule::kBlockTokens;
    constexpr bool MaskedColumns = !Schedule::kExactTokens;
    constexpr int kTileK         = Schedule::kTileKPerWarp;
    constexpr int kWarps         = Schedule::kKWarps;
    constexpr int kRowTiles      = Schedule::kRowTiles;
    constexpr int kPairSplit     = Schedule::kPairSplit;
    constexpr int kLocalWarps    = Schedule::kLocalWarps;
    constexpr int kLocalBlockK   = Schedule::kLocalBlockK;
    constexpr int kLocalChunks   = kLocalBlockK / 16;
    constexpr int kBlockRows     = Schedule::kBlockRows;
    constexpr int kBlockK        = Schedule::kBlockK;
    const int kGroups            = (kHidden + kBlockK - 1) / kBlockK;
    constexpr int kBlockTokens   = Schedule::kBlockTokens;
    constexpr int kTokenMmas     = kBlockTokens / 8;
    constexpr int kTileRows      = kBlockRows / (RowPolicy::kPaired ? 2 : 1);
    static_assert(ActiveTokens >= 1 && ActiveTokens <= kBlockTokens);
    static_assert((kLocalWarps & 1) == 0);
    constexpr unsigned kMask = 0xffffffffU;

    union SharedStorage {
        struct {
            std::uint8_t codes[Schedule::kStages][kRowTiles * kBlockRows][kLocalBlockK];
            __nv_bfloat16 activations[Schedule::kStages][kLocalWarps][kBlockTokens * kTileK];
        } staging;

        float partial[kRowTiles * kLocalWarps * kTokenMmas * 32 * 4];
    };

    static_assert(sizeof(SharedStorage) == Schedule::kSharedBytes);
    auto& shared = *reinterpret_cast<SharedStorage*>(fp8_shared_storage<Schedule::kSharedBytes>());
    auto& code_shared = shared.staging.codes;
    auto& x_shared    = shared.staging.activations;

    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int gid        = lane >> 2;
    const int lid        = lane & 3;
    const int warp_tile  = warp / kLocalWarps;
    const int local_warp = warp - warp_tile * kLocalWarps;
    const int row_block  = static_cast<int>(blockIdx.x) / kPairSplit;
    const int split      = static_cast<int>(blockIdx.x) - row_block * kPairSplit;
    // Global K warp: the arithmetic of warp w of the single-CTA schedule.
    const int k_warp     = split * kLocalWarps + local_warp;
    const int split_k0   = split * kLocalBlockK;
    const int tile_index = row_block * kRowTiles + warp_tile;
    const int row0       = tile_index * kTileRows;
    const int token_begin = token_offset + static_cast<int>(blockIdx.y) * ActiveTokens;
    const int live_columns =
        MaskedColumns ? min(ActiveTokens, operands.tokens - token_begin) : ActiveTokens;

    const auto stage_activation = [&](int stage, int group_k0) {
        constexpr auto kActivationCache = Schedule::kActivationCache;
        constexpr bool kPadded     = Schedule::kActivationStage == Fp8ActivationStage::PaddedZero;
        constexpr int kStageTokens = kPadded ? kBlockTokens : ActiveTokens;
        constexpr int kWarpItems   = kStageTokens * (kTileK / 8);
        constexpr int kItems       = kLocalWarps * kWarpItems;
        for (int item = tid; item < kItems; item += Schedule::kThreads) {
            const int slot  = item / kWarpItems;
            const int rest  = item - slot * kWarpItems;
            const int token = rest / (kTileK / 8);
            const int k8    = rest - token * (kTileK / 8);
            const int k     = group_k0 + split_k0 + slot * kTileK + k8 * 8;
            if (k >= kHidden) continue;
            auto* destination =
                &x_shared[stage][slot][token * kTileK + fp8_a16_shared_col_64(token, k8 * 8)];
            if constexpr (!MaskedColumns && (!kPadded || ActiveTokens == kBlockTokens)) {
                cp_async<16, kActivationCache>(
                    destination, x + static_cast<std::int64_t>(token_begin + token) * kHidden + k);
            } else {
                const int source_token = token < live_columns ? token : 0;
                cp_async_zfill<16, kActivationCache>(
                    destination,
                    x + static_cast<std::int64_t>(token_begin + source_token) * kHidden + k,
                    token < live_columns ? 16 : 0);
            }
        }
    };

    const auto stage_codes = [&](int stage, int group_k0) {
        constexpr auto kWeightCache = Schedule::kWeightCache;
        const int k0                = group_k0 + split_k0;
        const int group_chunks      = max(0, min(kLocalBlockK, kHidden - k0)) / 16;
        for (int item = tid; item < kRowTiles * kBlockRows * kLocalChunks;
             item += Schedule::kThreads) {
            const int row   = item / kLocalChunks;
            const int chunk = item - row * kLocalChunks;
            if (chunk >= group_chunks) continue;
            const int tile           = row / kBlockRows;
            const int tile_row       = row - tile * kBlockRows;
            const int swizzled_chunk = chunk ^ (tile_row & 7);
            const int weight_row     = row_policy.weight_row(
                (row_block * kRowTiles + tile) * kTileRows, tile_row, operands.rows);
            cp_async<16, kWeightCache>(&code_shared[stage][row][swizzled_chunk * 16],
                                       weight_codes +
                                           static_cast<std::int64_t>(weight_row) * kHidden + k0 +
                                           chunk * 16);
        }
    };

    const int b_row                   = lane & 7;
    const int b_k_offset              = ((lane >> 3) & 1) << 3;
    const int warp_k0                 = local_warp * kTileK;
    const int code_row0               = warp_tile * kBlockRows;
    float accumulators[kTokenMmas][4] = {};

#pragma unroll
    for (int stage = 0; stage < Schedule::kStages; ++stage) {
        if (stage < kGroups) {
            stage_codes(stage, stage * kBlockK);
            stage_activation(stage, stage * kBlockK);
            cp_commit();
        }
    }
#pragma unroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int stage = group_index % Schedule::kStages;
        if (group_index + Schedule::kStages - 1 < kGroups)
            cp_wait<Schedule::kStages - 1>();
        else
            cp_wait<0>();
        __syncthreads();
        const bool warp_in_k = group_index * kBlockK + k_warp * kTileK < kHidden;
        if (warp_in_k) {
#pragma unroll
            for (int k_step = 0; k_step < kTileK / 16; ++k_step) {
                const int code_col        = k_step * 16 + lid * 2;
                const auto load_code_pair = [&](int row, int col) {
                    const int chunk  = (warp_k0 + col) >> 4;
                    const int offset = (chunk ^ (row & 7)) * 16 + (col & 15);
                    return static_cast<unsigned>(*reinterpret_cast<const std::uint16_t*>(
                        &code_shared[stage][code_row0 + row][offset]));
                };
                const unsigned a0 = fp8_e4m3x2_to_bf16x2_bits(load_code_pair(gid, code_col));
                const unsigned a1 = fp8_e4m3x2_to_bf16x2_bits(load_code_pair(gid + 8, code_col));
                const unsigned a2 = fp8_e4m3x2_to_bf16x2_bits(load_code_pair(gid, code_col + 8));
                const unsigned a3 =
                    fp8_e4m3x2_to_bf16x2_bits(load_code_pair(gid + 8, code_col + 8));
#pragma unroll
                for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                    unsigned b0;
                    unsigned b1;
                    const int row = token_mma * 8 + b_row;
                    ldmatrix_x2(
                        b0, b1,
                        smem_addr(&x_shared[stage][local_warp]
                                           [row * kTileK +
                                            fp8_a16_shared_col_64(row, k_step * 16 + b_k_offset)]));
                    mma_bf16(accumulators[token_mma][0], accumulators[token_mma][1],
                             accumulators[token_mma][2], accumulators[token_mma][3], a0, a1, a2, a3,
                             b0, b1);
                }
            }
        }

        __syncthreads();
        const int next = group_index + Schedule::kStages;
        if (next < kGroups) {
            stage_codes(stage, next * kBlockK);
            stage_activation(stage, next * kBlockK);
            cp_commit();
        }
    }

    // Odd K warps fold into their even partners; the pair sums then fold left to right.
    __syncthreads();
    auto* partial          = shared.partial;
    const auto partial_at = [&](int slot, int token_mma) {
        return partial +
               (((warp_tile * kLocalWarps + slot) * kTokenMmas + token_mma) * 32 + lane) * 4;
    };
    if ((local_warp & 1) != 0) {
#pragma unroll
        for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
            store_vec(partial_at(local_warp, token_mma),
                      make_float4(accumulators[token_mma][0], accumulators[token_mma][1],
                                  accumulators[token_mma][2], accumulators[token_mma][3]));
        }
    }
    __syncthreads();

    if ((local_warp & 1) == 0) {
#pragma unroll
        for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
            const float4 partner = load_vec<float4>(partial_at(local_warp + 1, token_mma));
            accumulators[token_mma][0] += partner.x;
            accumulators[token_mma][1] += partner.y;
            accumulators[token_mma][2] += partner.z;
            accumulators[token_mma][3] += partner.w;
            if constexpr (kPairSplit == 1) {
                if (local_warp != 0) {
                    store_vec(partial_at(local_warp, token_mma),
                              make_float4(accumulators[token_mma][0], accumulators[token_mma][1],
                                          accumulators[token_mma][2], accumulators[token_mma][3]));
                }
            }
        }
    }

    // With split pairs, each CTA publishes its pair sums; the row tile's last CTA folds them.
    [[maybe_unused]] float* split_pairs = nullptr;
    if constexpr (kPairSplit > 1) {
        constexpr int kPairs = kWarps / 2;
        split_pairs = scratch.pairs + static_cast<std::size_t>(
                                          row_block * gridDim.y + blockIdx.y) *
                                          kPairs * kTokenMmas * 32 * 4;
        if ((local_warp & 1) == 0) {
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                __stcg(reinterpret_cast<float4*>(split_pairs) +
                           ((k_warp / 2) * kTokenMmas + token_mma) * 32 + lane,
                       make_float4(accumulators[token_mma][0], accumulators[token_mma][1],
                                   accumulators[token_mma][2], accumulators[token_mma][3]));
            }
        }
        __threadfence();
        __syncthreads();
        __shared__ unsigned arrival;
        if (tid == 0) {
            unsigned* counter = scratch.counters + row_block * gridDim.y + blockIdx.y;
            arrival           = atomicAdd(counter, 1U);
            if (arrival == kPairSplit - 1) *counter = 0U;
        }
        __syncthreads();
        if (arrival != kPairSplit - 1) {
            if constexpr (requires(const Epilogue& e) { e.finish_cta(0, 0, 0); }) {
                if (blockIdx.x == 0) epilogue.finish_cta(token_begin, live_columns, tid);
            }
            return;
        }
        __threadfence();
    } else {
        __syncthreads();
    }

    if (local_warp == 0) {
        [[maybe_unused]] const auto destination = linear_output_tile<kTileRows>(output, row0);
        unsigned lane_scale = 0;
        if (lid < 2) {
            lane_scale = static_cast<unsigned>(reinterpret_cast<const std::uint16_t*>(
                row_scales)[row_policy.weight_row(row0, gid + lid * 8, operands.rows)]);
        }
        const unsigned top_scale_bits    = __shfl_sync(kMask, lane_scale, lane & ~3);
        const unsigned bottom_scale_bits = __shfl_sync(kMask, lane_scale, (lane & ~3) + 1);
        const float top_scale =
            __bfloat162float(__ushort_as_bfloat16(static_cast<std::uint16_t>(top_scale_bits)));
        const float bottom_scale =
            __bfloat162float(__ushort_as_bfloat16(static_cast<std::uint16_t>(bottom_scale_bits)));

#pragma unroll
        for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
            float4 sum;
            if constexpr (kPairSplit == 1) {
                sum = make_float4(accumulators[token_mma][0], accumulators[token_mma][1],
                                  accumulators[token_mma][2], accumulators[token_mma][3]);
#pragma unroll
                for (int slot = 2; slot < kWarps; slot += 2) {
                    const float4 value = load_vec<float4>(partial_at(slot, token_mma));
                    sum.x += value.x;
                    sum.y += value.y;
                    sum.z += value.z;
                    sum.w += value.w;
                }
            } else {
                const auto* pairs = reinterpret_cast<const float4*>(split_pairs);
                sum               = __ldcg(pairs + token_mma * 32 + lane);
#pragma unroll
                for (int pair = 1; pair < kWarps / 2; ++pair) {
                    const float4 value = __ldcg(pairs + (pair * kTokenMmas + token_mma) * 32 + lane);
                    sum.x += value.x;
                    sum.y += value.y;
                    sum.z += value.z;
                    sum.w += value.w;
                }
            }
            const int local_token = token_mma * 8 + 2 * lid;
            const int token0      = token_begin + local_token;
            if constexpr (fp8_stream_quad_rows<RowPolicy>) {
                // Lane gid holds streams gid/4 and gid/4 + 2 of position gid % 4; lane ^ 16
                // (gid ^ 4) holds the other two. Both exchange, then the low half finishes
                // token0 and the high half token0 + 1, each with all four streams in order.
                const float own[4]{sum.x * top_scale, sum.y * top_scale, sum.z * bottom_scale,
                                   sum.w * bottom_scale};
                float other[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) other[i] = __shfl_xor_sync(kMask, own[i], 16);
                // Low lanes hold streams 0 and 2, high lanes streams 1 and 3.
                const bool low   = gid < 4;
                const int column = low ? 0 : 1;
                const float streams[4]{low ? own[0] : other[1], low ? other[0] : own[1],
                                       low ? own[2] : other[3], low ? other[2] : own[3]};
                if (local_token + column < live_columns)
                    epilogue.mix_streams((row0 >> 2) + (gid & 3), token0 + column, streams);
            } else {
                const int row_a = row_policy.weight_row(row0, gid, operands.rows);
                const int row_b = row_policy.weight_row(row0, gid + 8, operands.rows);
                if constexpr (RowPolicy::kPaired) {
                    if (local_token < live_columns)
                        epilogue.apply_pair(destination, row_a, token0, sum.x * top_scale,
                                            sum.z * bottom_scale);
                    if (local_token + 1 < live_columns)
                        epilogue.apply_pair(destination, row_a, token0 + 1, sum.y * top_scale,
                                            sum.w * bottom_scale);
                } else {
                    if (local_token < live_columns) {
                        destination.store(row_a, token0,
                                          epilogue.apply(row_a, token0, sum.x * top_scale));
                        destination.store(row_b, token0,
                                          epilogue.apply(row_b, token0, sum.z * bottom_scale));
                    }
                    if (local_token + 1 < live_columns) {
                        destination.store(row_a, token0 + 1,
                                          epilogue.apply(row_a, token0 + 1, sum.y * top_scale));
                        destination.store(row_b, token0 + 1,
                                          epilogue.apply(row_b, token0 + 1, sum.w * bottom_scale));
                    }
                }
            }
        }
    }
    // An epilogue may finish per-token work once per token tile, in the tile's first row CTA.
    if constexpr (requires(const Epilogue& e) { e.finish_cta(0, 0, 0); }) {
        if (blockIdx.x == 0) epilogue.finish_cta(token_begin, live_columns, tid);
    }
}

} // namespace ninfer::ops::detail
