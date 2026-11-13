#pragma once

#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/kernels/q5/q5_instances.cuh"
#include "ops/linear/kernels/q5/q5_mma_launch.cuh"
#include "ops/linear/kernels/q5/q5_sliced_k_launch.cuh"
#include "ops/linear/kernels/q5/q5_simt_launch.cuh"

namespace ninfer::ops::detail {
struct Q5LinearAddTileEpilogue {
    template <class Schedule, bool Full>
    __device__ __forceinline__ void
    finish_tile(LinearBf16StridedOutput output, __nv_bfloat16* scratch,
                const float (&acc)[Schedule::kMmaRows][Schedule::kMmaTokens][4], int row_begin,
                int token_begin, int rows, int token_end) const {
        constexpr int R = Schedule::kBlockRows;
        constexpr int T = Schedule::kBlockTokens;
        static_assert(R <= Schedule::kBlockK && R % 8 == 0);
        const int tid        = int(threadIdx.x);
        const int warp       = tid >> 5;
        const int lane       = tid & 31;
        const int warp_row   = warp / Schedule::kWarpGridTokens;
        const int warp_token = warp % Schedule::kWarpGridTokens;
#pragma unroll
        for (int mi = 0; mi < Schedule::kMmaRows; ++mi) {
            const int row = warp_row * Schedule::kWarpRows + mi * 16 + (lane >> 2);
#pragma unroll
            for (int ni = 0; ni < Schedule::kMmaTokens; ++ni) {
                const int token = warp_token * Schedule::kWarpTokens + ni * 8 + 2 * (lane & 3);
                scratch[token * R + row]           = __float2bfloat16_rn(acc[mi][ni][0]);
                scratch[(token + 1) * R + row]     = __float2bfloat16_rn(acc[mi][ni][1]);
                scratch[token * R + row + 8]       = __float2bfloat16_rn(acc[mi][ni][2]);
                scratch[(token + 1) * R + row + 8] = __float2bfloat16_rn(acc[mi][ni][3]);
            }
        }
        __syncthreads();

        union alignas(16) Pack {
            int4 raw;
            common::Bf16x8Pack values;
        };

        for (int item = tid; item < T * (R / 8); item += Schedule::kThreads) {
            const int local_token = item / (R / 8);
            const int local_row   = (item % (R / 8)) * 8;
            const int row         = row_begin + local_row;
            const int token       = token_begin + local_token;
            if (Full || (row < rows && token < token_end)) {
                auto* dst =
                    output.data + std::int64_t(token) * output.leading_dim + output.row_begin + row;
                const auto* projected = scratch + local_token * R + local_row;
                if (Full || row + 8 <= rows) {
                    Pack value, residual;
                    value.raw    = common::load_vec<int4>(projected);
                    residual.raw = common::load_vec<int4>(dst);
#pragma unroll
                    for (int pair = 0; pair < 4; ++pair)
                        residual.values.pair[pair] =
                            __floats2bfloat162_rn(__low2float(residual.values.pair[pair]) +
                                                      __low2float(value.values.pair[pair]),
                                                  __high2float(residual.values.pair[pair]) +
                                                      __high2float(value.values.pair[pair]));
                    common::store_vec(dst, residual.raw);
                } else {
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        if (row + i < rows)
                            dst[i] = __float2bfloat16_rn(__bfloat162float(dst[i]) +
                                                         __bfloat162float(projected[i]));
                }
            }
        }
    }
};

template <class Schedule>
void launch_q5_add_sliced(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream) {
    auto* data        = static_cast<__nv_bfloat16*>(residual.data);
    const auto stride = std::int64_t(residual.nb[1] / sizeof(__nv_bfloat16));
    launch_q5_a16_sliced_k_mma<Schedule>(q5_linear_operands(x, w),
                                         LinearBf16StridedOutput{data, stride, 0},
                                         LinearResidualAddEpilogue{{data, stride, 0}}, stream);
}

template <class Schedule>
void launch_q5_add_pointwise(const Tensor& x, const Weight& w, Tensor& residual,
                             cudaStream_t stream) {
    auto* data        = static_cast<__nv_bfloat16*>(residual.data);
    const auto stride = std::int64_t(residual.nb[1] / sizeof(__nv_bfloat16));
    launch_q5_a16_mma<Schedule>(q5_linear_operands(x, w), LinearBf16StridedOutput{data, stride, 0},
                                LinearResidualAddEpilogue{{data, stride, 0}}, stream);
}

template <class Schedule>
void launch_q5_add_collective(const Tensor& x, const Weight& w, Tensor& residual,
                              cudaStream_t stream) {
    auto* data        = static_cast<__nv_bfloat16*>(residual.data);
    const auto stride = std::int64_t(residual.nb[1] / sizeof(__nv_bfloat16));
    launch_q5_a16_mma<Schedule>(q5_linear_operands(x, w), LinearBf16StridedOutput{data, stride, 0},
                                Q5LinearAddTileEpilogue{}, stream);
}

template <class Schedule>
void launch_q5_add_simt(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream) {
    auto* data        = static_cast<__nv_bfloat16*>(residual.data);
    const auto stride = std::int64_t(residual.nb[1] / sizeof(__nv_bfloat16));
    launch_q5_a16_direct_simt<Schedule>(q5_linear_operands(x, w),
                                        LinearBf16StridedOutput{data, stride, 0},
                                        LinearResidualAddEpilogue{{data, stride, 0}}, stream);
}


// One instantiation point per schedule. Both add/q5 shape files select the same sliced-K, collective
// and pointwise schedules, so each template instantiation used to be emitted in both translation
// units; with separable device compilation the module then registered the same __global__ kernel
// twice, which is what made compute-sanitizer and cuda-gdb refuse the binary. The definitions live in
// instances.cu and these declarations keep every other translation unit from emitting them again.
extern template void launch_q5_add_sliced<q5_instances::SlicedR16T8W4S2>(
    const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream);
extern template void launch_q5_add_sliced<q5_instances::SlicedR16T16W4S2>(
    const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream);
extern template void launch_q5_add_sliced<q5_instances::SlicedR16T24W4S2>(
    const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream);
extern template void launch_q5_add_sliced<q5_instances::SlicedR32T24W4S2Pairwise>(
    const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream);
extern template void launch_q5_add_sliced<q5_instances::SlicedR32T32W2S2>(
    const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream);
extern template void launch_q5_add_sliced<q5_instances::SlicedR32T32W4S2>(
    const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream);
extern template void launch_q5_add_sliced<q5_instances::SlicedR32T64W2S1>(
    const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream);
extern template void launch_q5_add_collective<q5_instances::MmaR64T128>(
    const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream);
extern template void launch_q5_add_pointwise<q5_instances::MmaR32T128>(
    const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream);

} // namespace ninfer::ops::detail
