#pragma once

// Dense MXFP8 x BF16-activation linear kernels (the bounded A16 fallback).
//
//   out[T,N] = x[T,K] * (E4M3(codes) * 2^(ue8m0 - 127))[N,K]^T
//
// The persistent words are decoded in the kernel; every product is computed
// against the exact E4M3-to-BF16 widening with the per-32-group UE8M0 exponent
// folded in as an FP32 power of two. Accumulation is FP32 and the output is
// BF16. Group scales are read straight from global (two bytes per weight row
// and K tile); only codes and activations are staged through shared memory.
// Partial row extents are supported: loads clamp to the last valid row and the
// epilogue stores guarded rows only.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/fp8/fp8_a16_codec.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/mxfp8/mxfp8_a16_codec.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// ---------------------------------------------------------------------------
// T=1 GEMV. Each warp owns contiguous output rows and reads naturally aligned
// E4M3 code packs; the owning group scale is applied before the FP32 FMA.
// ---------------------------------------------------------------------------

template <int WarpsPerCta, int RowsPerWarp, int ValuesPerLane, int AccumulatorChains,
          int PhaseUnroll, int MinBlocksPerSm>
struct Mxfp8A16GemvSchedule {
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 8 || ValuesPerLane == 16);
    static_assert(AccumulatorChains > 0 && (AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(AccumulatorChains <= ValuesPerLane);
    static_assert(PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4);
    static constexpr int kWarpsPerCta    = WarpsPerCta;
    static constexpr int kRowsPerWarp    = RowsPerWarp;
    static constexpr int kValuesPerLane  = ValuesPerLane;
    static constexpr int kChains         = AccumulatorChains;
    static constexpr int kPhaseUnroll    = PhaseUnroll;
    static constexpr int kMinBlocksPerSm = MinBlocksPerSm;
    static constexpr int kThreads        = WarpsPerCta * 32;
    static constexpr int kBlockRows      = WarpsPerCta * RowsPerWarp;
};

template <class Schedule>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void mxfp8_a16_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ weight_codes,
    const std::uint8_t* __restrict__ group_scales, __nv_bfloat16* __restrict__ out, int rows,
    int K) {
    constexpr int kValuesPerPhase = kWarpSize * Schedule::kValuesPerLane;
    static_assert((kValuesPerPhase % 32) == 0,
                  "one lane pack must stay inside one MXFP8 scale group");
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int row_begin =
        static_cast<int>(blockIdx.x) * Schedule::kBlockRows + warp * Schedule::kRowsPerWarp;
    const int scale_row_words = K / 32;
    const auto* activation_pairs = reinterpret_cast<const std::uint32_t*>(x);
    float accumulators[Schedule::kRowsPerWarp][Schedule::kChains] = {};

    const auto accumulate_phase = [&](int value_begin) {
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            const int weight_row = row_begin + local_row;
            if (weight_row >= rows) { continue; }
            const auto* source =
                weight_codes + static_cast<std::int64_t>(weight_row) * K + value_begin;
            const float group_scale = mxfp8_group_scale(
                __ldg(group_scales + static_cast<std::int64_t>(weight_row) * scale_row_words +
                      value_begin / 32));
            uint4 packed_words;
            if constexpr (Schedule::kValuesPerLane == 16) {
                packed_words = load_ldg<uint4>(source);
            } else {
                const uint2 v = load_ldg<uint2>(source);
                packed_words = make_uint4(v.x, v.y, 0, 0);
            }
#pragma unroll
            for (int pair = 0; pair < Schedule::kValuesPerLane / 2; ++pair) {
                const float2 activation =
                    bf16x2_bits_to_float2(activation_pairs[value_begin / 2 + pair]);
                const std::uint16_t packed = static_cast<std::uint16_t>(
                    (&packed_words.x)[pair >> 1] >> ((pair & 1) * 16));
                const float2 weight = decode_mxfp8_e4m3x2(packed);
                const int chain_a   = (2 * pair) & (Schedule::kChains - 1);
                const int chain_b   = (2 * pair + 1) & (Schedule::kChains - 1);
                accumulators[local_row][chain_a] =
                    fmaf(weight.x * group_scale, activation.x, accumulators[local_row][chain_a]);
                accumulators[local_row][chain_b] =
                    fmaf(weight.y * group_scale, activation.y, accumulators[local_row][chain_b]);
            }
        }
    };

    const int kPhases = K / kValuesPerPhase;
#pragma unroll Schedule::kPhaseUnroll
    for (int phase = 0; phase < kPhases; ++phase) {
        accumulate_phase(phase * kValuesPerPhase + lane * Schedule::kValuesPerLane);
    }

#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
        const int weight_row = row_begin + local_row;
        if (weight_row >= rows) { continue; }
        float total = 0.0F;
#pragma unroll
        for (int chain = 0; chain < Schedule::kChains; ++chain) {
            total += accumulators[local_row][chain];
        }
        total = warp_reduce_sum(total);
        if (lane == 0) {
            out[weight_row] = __float2bfloat16_rn(total);
        }
    }
}

// ---------------------------------------------------------------------------
// Large-T tensor-core GEMM. Codes are staged, widened exactly to BF16 with the
// group scale folded in, and contracted with FP32-accumulating BF16 MMAs.
// ---------------------------------------------------------------------------

template <int BlockRows, int BlockTokens, int BlockK, int WarpRows, int WarpTokens,
          int ActivationStages, int MinBlocksPerSm, Cache WeightCache, Cache ActivationCache>
struct Mxfp8A16MmaSchedule {
    static_assert(BlockK == 64 || BlockK == 128);
    static_assert(BlockK % 32 == 0, "MXFP8 groups are 32 wide");
    static_assert(BlockRows % WarpRows == 0 && BlockTokens % WarpTokens == 0);
    static_assert(WarpRows % 16 == 0 && WarpTokens % 8 == 0);
    static_assert(ActivationStages == 1 || ActivationStages == 2);
    static constexpr int kBlockRows         = BlockRows;
    static constexpr int kBlockTokens       = BlockTokens;
    static constexpr int kBlockK            = BlockK;
    static constexpr int kWarpRows          = WarpRows;
    static constexpr int kWarpTokens        = WarpTokens;
    static constexpr int kActivationStages  = ActivationStages;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr Cache kWeightCache     = WeightCache;
    static constexpr Cache kActivationCache = ActivationCache;
    static constexpr int kWarpsRows         = kBlockRows / kWarpRows;
    static constexpr int kWarpsTokens       = kBlockTokens / kWarpTokens;
    static constexpr int kWarps             = kWarpsRows * kWarpsTokens;
    static constexpr int kThreads           = kWarps * 32;
    static constexpr int kMmaRows           = kWarpRows / 16;
    static constexpr int kMmaTokens         = kWarpTokens / 8;
    static constexpr int kMmaK              = kBlockK / 16;
    static constexpr int kSharedBytes =
        kBlockRows * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kActivationStages * kBlockTokens * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kBlockRows * kBlockK;
    static_assert(kSharedBytes <= 99 * 1024);
};

template <class Schedule, bool FullTokens>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void mxfp8_a16_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ weight_codes,
    const std::uint8_t* __restrict__ group_scales, __nv_bfloat16* __restrict__ out, int rows,
    int input_rows, int token_offset, int count) {
    const int K = input_rows;
    constexpr int BM   = Schedule::kBlockRows;
    constexpr int BN   = Schedule::kBlockTokens;
    constexpr int BK   = Schedule::kBlockK;
    constexpr int WM   = Schedule::kWarpRows;
    constexpr int WN   = Schedule::kWarpTokens;
    constexpr int MT   = Schedule::kMmaRows;
    constexpr int NT   = Schedule::kMmaTokens;
    constexpr int KSUB = Schedule::kMmaK;
    constexpr int THREADS = Schedule::kThreads;
    constexpr int kGroupsPerRowTile = BK / 32;
    const int scale_row_words = K / 32;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* weight_shared = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    auto* activation_shared = weight_shared + BM * BK;
    auto* code_shared = reinterpret_cast<std::uint8_t*>(activation_shared +
                                                        Schedule::kActivationStages * BN * BK);

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / Schedule::kWarpsTokens;
    const int wn   = warp - wm * Schedule::kWarpsTokens;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    const int row_begin   = static_cast<int>(blockIdx.x) * BM;
    const int token_begin = token_offset + static_cast<int>(blockIdx.y) * BN;

    float accumulators[MT][NT][4] = {};

    const int a_matrix     = lane >> 3;
    const int a_inner_row  = lane & 7;
    const int a_row_offset = a_inner_row + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row  = lane & 7;
    const int b_col_offset = ((lane >> 3) & 1) << 3;

    const auto stage_activation = [&](int stage, int k_tile) {
        const int k_begin = k_tile * BK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += THREADS) {
            const int local_token = item / (BK / 8);
            const int k8          = item - local_token * (BK / 8);
            const int token       = token_begin + local_token;
            const int kk          = k8 * 8;
            auto* destination = &activation_shared[stage * BN * BK + local_token * BK +
                                                   mxfp8_shared_col_64(local_token, kk)];
            if constexpr (FullTokens) {
                cp_async<16, Schedule::kActivationCache>(
                    destination, x + static_cast<std::int64_t>(token) * K + k_begin + kk);
            } else {
                const bool valid = token < token_offset + count;
                cp_async_zfill<16, Schedule::kActivationCache>(
                    destination,
                    x + static_cast<std::int64_t>(valid ? token : 0) * K + k_begin + kk,
                    valid ? 16 : 0);
            }
        }
    };

    const auto stage_codes = [&](int k_tile) {
        const int k_begin = k_tile * BK;
#pragma unroll 1
        for (int item = tid; item < BM * (BK / 16); item += THREADS) {
            const int local_row = item / (BK / 16);
            const int chunk     = item - local_row * (BK / 16);
            const int weight_row =
                row_begin + local_row < rows ? row_begin + local_row : rows - 1;
            cp_async<16, Schedule::kWeightCache>(
                &code_shared[local_row * BK + chunk * 16],
                weight_codes + static_cast<std::int64_t>(weight_row) * K + k_begin + chunk * 16);
        }
    };

    const auto widen_codes = [&](int k_tile) {
        const int half      = lane >> 4;
        const int half_lane = lane & 15;
        const int k_begin   = k_tile * BK;
#pragma unroll 1
        for (int row_pair = warp * 2; row_pair < BM; row_pair += Schedule::kWarps * 2) {
            const int row        = row_pair + half;
            const int weight_row = row_begin + row < rows ? row_begin + row : rows - 1;
            const auto* row_scales =
                group_scales + static_cast<std::int64_t>(weight_row) * scale_row_words;
#pragma unroll
            for (int col32 = 0; col32 < BK; col32 += 32) {
                const float group_scale =
                    mxfp8_group_scale(__ldg(row_scales + k_begin / 32 + col32 / 32));
                const int col           = col32 + half_lane * 2;
                const unsigned packed   = static_cast<unsigned>(
                    *reinterpret_cast<const std::uint16_t*>(&code_shared[row * BK + col]));
                const __nv_bfloat162 widened = mxfp8_widen_pair_scaled(packed, group_scale);
                store_vec(&weight_shared[row * BK + mxfp8_shared_col_64(row, col)], widened);
            }
        }
    };

    stage_activation(0, 0);
    stage_codes(0);
    cp_commit();

    const int kTiles = K / BK;
#pragma unroll 1
    for (int k_tile = 0; k_tile < kTiles; ++k_tile) {
        const int stage = k_tile % Schedule::kActivationStages;
        cp_wait<0>();
        __syncthreads();

        widen_codes(k_tile);
        __syncthreads();

        const int next = k_tile + 1;
        if (next < kTiles) {
            if constexpr (Schedule::kActivationStages == 2) { stage_activation(next & 1, next); }
            stage_codes(next);
            cp_commit();
        }

        unsigned a_fragments[MT][4];
        unsigned b_fragments[NT][2];
#pragma unroll
        for (int k_step = 0; k_step < KSUB; ++k_step) {
            const int ks = k_step * 16;
#pragma unroll
            for (int mma_row = 0; mma_row < MT; ++mma_row) {
                const int row = wm * WM + mma_row * 16 + a_row_offset;
                ldmatrix_x4(a_fragments[mma_row][0], a_fragments[mma_row][1],
                            a_fragments[mma_row][2], a_fragments[mma_row][3],
                            smem_addr(&weight_shared[row * BK +
                                                     mxfp8_shared_col_64(row, ks + a_col_offset)]));
            }
#pragma unroll
            for (int mma_token = 0; mma_token < NT; ++mma_token) {
                const int row = wn * WN + mma_token * 8 + b_inner_row;
                ldmatrix_x2(b_fragments[mma_token][0], b_fragments[mma_token][1],
                            smem_addr(&activation_shared[stage * BN * BK + row * BK +
                                                         mxfp8_shared_col_64(row, ks + b_col_offset)]));
            }
#pragma unroll
            for (int mma_row = 0; mma_row < MT; ++mma_row) {
#pragma unroll
                for (int mma_token = 0; mma_token < NT; ++mma_token) {
                    mma_bf16(accumulators[mma_row][mma_token][0],
                             accumulators[mma_row][mma_token][1],
                             accumulators[mma_row][mma_token][2],
                             accumulators[mma_row][mma_token][3], a_fragments[mma_row][0],
                             a_fragments[mma_row][1], a_fragments[mma_row][2],
                             a_fragments[mma_row][3], b_fragments[mma_token][0],
                             b_fragments[mma_token][1]);
                }
            }
        }

        if constexpr (Schedule::kActivationStages == 1) {
            if (next < kTiles) {
                __syncthreads();
                stage_activation(0, next);
                cp_commit();
            }
        }
    }

    const LinearBf16Output destination{out, rows};
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int row0 = row_begin + wm * WM + mi * 16 + gid;
        const int row1 = row0 + 8;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int token0 = token_begin + wn * WN + ni * 8 + 2 * lid;
            const auto& v    = accumulators[mi][ni];
            if (row0 < rows) {
                if (FullTokens || token0 < token_offset + count) {
                    destination.store(row0, token0, v[0]);
                }
                if (FullTokens || token0 + 1 < token_offset + count) {
                    destination.store(row0, token0 + 1, v[1]);
                }
            }
            if (row1 < rows) {
                if (FullTokens || token0 < token_offset + count) {
                    destination.store(row1, token0, v[2]);
                }
                if (FullTokens || token0 + 1 < token_offset + count) {
                    destination.store(row1, token0 + 1, v[3]);
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
