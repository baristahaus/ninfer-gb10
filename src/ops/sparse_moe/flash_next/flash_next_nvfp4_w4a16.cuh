#pragma once

// NVFP4 W4A16 grouped kernel for the Flash-Next 512-way MoE routed experts. The bank is the
// expert_block_scale_k16_m128x4_v1 layout (E2M1 code plane, m128x4-swizzled E4M3 G16 scale
// plane, per-expert FP32 weight divisors) without activation divisors: activations stay the
// represented public BF16 inputs. Codes times their raw E4M3 scales are exactly representable
// in BF16; the represented weight is that product divided by the expert's weight divisor, and
// the divisor scales the complete FP32 reduction - exactly like the shared NVFP4 A16 linear
// route. The kernel keeps the W4A4 kernel's persistent expert-job policy: one job covers one
// expert's weight rows for one 32-token tile of the packed assignments, with the activation
// operand as the MMA M dimension so accumulator column pairs are consecutive weight rows.

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::flash_next {

template <int BlockM, int BlockN, int BlockK, int WarpsM, int WarpsN, int MinBlocksPerSm>
struct Nvfp4W4a16GroupedSchedule {
    static_assert(BlockM > 0 && (BlockM % 128) == 0);
    static_assert(BlockN > 0 && (BlockN % 16) == 0);
    static_assert(BlockK == 64);
    static_assert((BlockM / WarpsN) % 8 == 0);
    static_assert((BlockN / WarpsM) % 16 == 0);

    static constexpr int kBlockM         = BlockM;  // weight rows per block (gate+up pairs for gate)
    static constexpr int kBlockN         = BlockN;  // packed tokens per block
    static constexpr int kBlockK         = BlockK;
    static constexpr int kWarpsM         = WarpsM;  // token warps
    static constexpr int kWarpsN         = WarpsN;  // weight-row warps
    static constexpr int kMinBlocksPerSm = MinBlocksPerSm;
    static constexpr int kWarps          = WarpsM * WarpsN;
    static constexpr int kThreads        = kWarps * 32;
    static constexpr int kWarpM          = BlockN / WarpsM;  // tokens per warp
    static constexpr int kWarpN          = BlockM / WarpsN;  // weight rows per warp
    static constexpr int kMmaM           = kWarpM / 16;
    static constexpr int kMmaN           = kWarpN / 8;
    static constexpr int kCodeRowBytes   = BlockK / 2;
    static constexpr int kSharedBytes =
        BlockM * BlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        BlockN * BlockK * static_cast<int>(sizeof(__nv_bfloat16)) + BlockM * (BlockK / 2);
};

// Persistent over (job, row block) work items. `configure` publishes the job's packed-token
// window [token_begin, token_end), the row-block base, the expert's divisor alpha, and points
// the row policy at the expert.
template <class RowPolicy>
struct Nvfp4W4a16GroupedWork {
    static constexpr bool kPersistent = true;
    const int* job_count;
    const int* job_experts;
    const int* job_columns;
    const int* offsets;
    const float* weight_divisors;
    int token_tile;
    int output_rows;

    __device__ __forceinline__ int work_count(int rows_per_block) const {
        return *job_count * (output_rows / rows_per_block);
    }

    __device__ __forceinline__ void configure(int work, int rows_per_block, int& token_begin,
                                              int& token_end, int& row_begin, float& alpha,
                                              RowPolicy& rows) const {
        const int row_blocks = output_rows / rows_per_block;
        const int job        = work / row_blocks;
        const int row_block  = work - job * row_blocks;
        const int expert     = job_experts[job];
        token_begin          = offsets[expert] + job_columns[job];
        token_end            = min(token_begin + token_tile, offsets[expert + 1]);
        row_begin            = row_block * rows_per_block;
        alpha                = 1.0F / weight_divisors[expert];
        rows.expert          = expert;
    }
};

template <class Schedule, class Output, class RowPolicy, class WorkPolicy, bool PairRows>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void
nvfp4_w4a16_grouped_mma_kernel(const __nv_bfloat16* __restrict__ x,
                               const std::uint8_t* __restrict__ weight_codes,
                               const std::uint8_t* __restrict__ weight_scales, int input_rows,
                               Output output, RowPolicy row_policy, WorkPolicy work_policy) {
    constexpr int BM      = Schedule::kBlockM;
    constexpr int BN      = Schedule::kBlockN;
    constexpr int BK      = Schedule::kBlockK;
    constexpr int MT      = Schedule::kMmaM;
    constexpr int NT      = Schedule::kMmaN;
    constexpr int THREADS = Schedule::kThreads;
    constexpr int kRowsPerBlock = PairRows ? BM / 2 : BM;
    constexpr int kOutputStride = BM + 8;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* weight_shared     = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    auto* activation_shared = weight_shared + BM * BK;
    auto* code_shared       = reinterpret_cast<std::uint8_t*>(activation_shared + BN * BK);

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / Schedule::kWarpsN;
    const int wn   = warp - wm * Schedule::kWarpsN;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    const int a_matrix     = lane >> 3;
    const int a_inner_row  = lane & 7;                  // token row inside the 16-token A tile
    const int a_row_offset = a_inner_row + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_row_offset = lane & 7;                  // weight row inside the 8-row B tile
    const int b_col_offset = ((lane >> 3) & 1) * 8;

    int work       = WorkPolicy::kPersistent ? static_cast<int>(blockIdx.x) : 0;
    const int work_limit =
        WorkPolicy::kPersistent ? work_policy.work_count(kRowsPerBlock) : 1;

    while (work < work_limit) {
        int token_begin;
        int token_end;
        int row_begin;
        float work_alpha = 1.0F;
        work_policy.configure(work, kRowsPerBlock, token_begin, token_end, row_begin, work_alpha,
                              row_policy);

        const int k_tiles = input_rows / BK;
        float accumulators[MT][NT][4] = {};

        for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
            const int k_begin = k_tile * BK;
            // Stage the packed BF16 activation tile [BN, BK]; rows past the job's window zero-fill.
#pragma unroll 1
            for (int item = tid; item < BN * (BK / 8); item += THREADS) {
                const int local_token = item / (BK / 8);
                const int k8          = item - local_token * (BK / 8);
                const int token       = token_begin + local_token;
                const int kk          = k8 * 8;
                auto* destination =
                    &activation_shared[local_token * BK + nvfp4_a16_shared_col_64(local_token, kk)];
                const bool valid = token < token_end;
                cp_async_zfill<16, Cache::ca>(
                    destination,
                    x + static_cast<std::int64_t>(valid ? token : 0) * input_rows + k_begin + kk,
                    valid ? 16 : 0);
            }
            // Stage the code tile [BM, BK/2].
#pragma unroll 1
            for (int item = tid; item < BM * (BK / 32); item += THREADS) {
                const int local_row  = item / (BK / 32);
                const int chunk      = item - local_row * (BK / 32);
                const int weight_row = row_policy.weight_row(row_begin, local_row);
                cp_async<16, Cache::cg>(
                    &code_shared[local_row * Schedule::kCodeRowBytes + chunk * 16],
                    weight_codes + static_cast<std::int64_t>(weight_row) * (input_rows / 2) +
                        k_begin / 2 + chunk * 16);
            }
            cp_commit();
            cp_wait<0>();
            __syncthreads();

            // Widen the codes to BF16 with their G16 scales, read straight from the swizzled
            // plane; each 16-value group shares one scale byte.
#pragma unroll 1
            for (int item = tid; item < BM * (BK / 16); item += THREADS) {
                const int local_row  = item / (BK / 16);
                const int group      = item - local_row * (BK / 16);
                const int col        = group * 16;
                const int weight_row = row_policy.weight_row(row_begin, local_row);
                const std::uint8_t scale_byte = __ldg(
                    weight_scales +
                    nvfp4_scale_byte_offset(weight_row, k_begin / 16 + group, input_rows));
                const auto* codes = &code_shared[local_row * Schedule::kCodeRowBytes + group * 8];
#pragma unroll
                for (int pair = 0; pair < 8; ++pair) {
                    const unsigned widened = nvfp4_scaled_pair_bf16(codes[pair], scale_byte);
                    store_vec(&weight_shared[local_row * BK +
                                             nvfp4_a16_shared_col_64(local_row, col + pair * 2)],
                              widened);
                }
            }
            __syncthreads();

#pragma unroll
            for (int k_step = 0; k_step < BK / 16; ++k_step) {
                const int ks = k_step * 16;
                unsigned a_fragments[MT][4];
                unsigned b_fragments[NT][2];
#pragma unroll
                for (int mma_m = 0; mma_m < MT; ++mma_m) {
                    const int token = wm * Schedule::kWarpM + mma_m * 16 + a_row_offset;
                    ldmatrix_x4(a_fragments[mma_m][0], a_fragments[mma_m][1],
                                a_fragments[mma_m][2], a_fragments[mma_m][3],
                                smem_addr(&activation_shared[token * BK +
                                                             nvfp4_a16_shared_col_64(token, ks + a_col_offset)]));
                }
#pragma unroll
                for (int mma_n = 0; mma_n < NT; ++mma_n) {
                    const int row = wn * Schedule::kWarpN + mma_n * 8 + b_row_offset;
                    ldmatrix_x2(b_fragments[mma_n][0], b_fragments[mma_n][1],
                                smem_addr(&weight_shared[row * BK +
                                                         nvfp4_a16_shared_col_64(row, ks + b_col_offset)]));
                }
#pragma unroll
                for (int mma_m = 0; mma_m < MT; ++mma_m) {
#pragma unroll
                    for (int mma_n = 0; mma_n < NT; ++mma_n) {
                        mma_bf16(accumulators[mma_m][mma_n][0],
                                 accumulators[mma_m][mma_n][1],
                                 accumulators[mma_m][mma_n][2],
                                 accumulators[mma_m][mma_n][3], a_fragments[mma_m][0],
                                 a_fragments[mma_m][1], a_fragments[mma_m][2],
                                 a_fragments[mma_m][3], b_fragments[mma_n][0],
                                 b_fragments[mma_n][1]);
                    }
                }
            }
            __syncthreads();
        }

        // Epilogue: scale by the expert's weight divisor, stage [token][row] through shared
        // memory (reusing the weight plane: 32 tokens x (BM + 8) words <= BM x BK words), and
        // store through the output policy.
        auto* shared_output = weight_shared;
        static_assert(BN * (BM + 8) <= BM * BK);
        const int accumulator_row = lane >> 2;
        const int accumulator_col = 2 * (lane & 3);
#pragma unroll
        for (int mma_m = 0; mma_m < MT; ++mma_m) {
            const int token0 = token_begin + wm * Schedule::kWarpM + mma_m * 16 + accumulator_row;
            const int token1 = token0 + 8;
#pragma unroll
            for (int mma_n = 0; mma_n < NT; ++mma_n) {
                const int local_row0 = wn * Schedule::kWarpN + mma_n * 8 + accumulator_col;
                const auto& v        = accumulators[mma_m][mma_n];
                auto* destination0 = reinterpret_cast<__nv_bfloat162*>(
                    shared_output + (token0 - token_begin) * kOutputStride + local_row0);
                auto* destination1 = reinterpret_cast<__nv_bfloat162*>(
                    shared_output + (token1 - token_begin) * kOutputStride + local_row0);
                *destination0 = __floats2bfloat162_rn(v[0] * work_alpha, v[1] * work_alpha);
                *destination1 = __floats2bfloat162_rn(v[2] * work_alpha, v[3] * work_alpha);
            }
        }
        __syncthreads();
        constexpr int kStoredRows    = PairRows ? BM / 2 : BM;
        constexpr int kVectorsPerRow = kStoredRows / 8;
        constexpr int kOutputVectors = BN * kVectorsPerRow;
        for (int task = tid; task < kOutputVectors; task += THREADS) {
            const int token_local = task / kVectorsPerRow;
            const int row_vector  = task - token_local * kVectorsPerRow;
            const int token       = token_begin + token_local;
            if (token < token_end) {
                const uint4 values =
                    load_vec<uint4>(shared_output + token_local * kOutputStride + row_vector * 8);
                if constexpr (PairRows) {
                    const uint4 paired = load_vec<uint4>(shared_output + token_local * kOutputStride +
                                                         kStoredRows + row_vector * 8);
                    output.store_pair_vector(row_begin + row_vector * 8, token, values, paired);
                } else {
                    output.store_vector(row_begin + row_vector * 8, token, values);
                }
            }
        }
        __syncthreads();
        work += static_cast<int>(gridDim.x);
    }
}

} // namespace ninfer::ops::detail::flash_next
