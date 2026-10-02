#include "ops/linear/bf16/flash_next/bf16_launch.h"

#include "core/device.h"
#include "ops/linear/bf16/flash_next/bf16_gemv.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail::flash_next {
namespace {

struct Bf16QueryGateOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* gate;

    __device__ __forceinline__ void store(std::int32_t parent_row,
                                          __nv_bfloat16 value) const {
        constexpr int kHeadWidth = 256;
        constexpr int kPackedHeadWidth = 2 * kHeadWidth;
        const int head = parent_row / kPackedHeadWidth;
        const int within_head = parent_row - head * kPackedHeadWidth;
        if (within_head < kHeadWidth) {
            query[head * kHeadWidth + within_head] = value;
        } else {
            gate[head * kHeadWidth + within_head - kHeadWidth] = value;
        }
    }
};

template <class Geometry>
void launch_geometry(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Bf16LinearDecodeSchedule<Geometry>;

    const Bf16ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data)};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    bf16_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.qdata),
        output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_bf16_query_gate_decode(const Tensor& x, const Weight& weight, Tensor& query,
                                   Tensor& gate, cudaStream_t stream) {
    using Geometry = Bf16GemvGeometry<12288, 2560>;
    using Schedule = Bf16LinearDecodeSchedule<Geometry>;
    if (x.ne[1] != 1 || weight.n != Geometry::kOutputRows ||
        weight.k != Geometry::kInputRows || query.numel() != 6144 || gate.numel() != 6144) {
        throw std::invalid_argument("BF16 query/gate decode: invalid exact problem");
    }
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const Bf16QueryGateOutput output{static_cast<__nv_bfloat16*>(query.data),
                                     static_cast<__nv_bfloat16*>(gate.data)};
    bf16_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weight.qdata), output);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void bf16_shared_swiglu_decode_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* gate_weight,
    const __nv_bfloat16* up_weight, __nv_bfloat16* out) {
    using Geometry = Bf16GemvGeometry<640, 2560>;
    using Schedule = Bf16LinearDecodeSchedule<Geometry>;
    static_assert(Schedule::kRowsPerCta == 1);
    __shared__ float gate_partials[Schedule::kWarpsPerRow];
    __shared__ float up_partials[Schedule::kWarpsPerRow];
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    float gate_accumulator[1][Schedule::kAccumulatorChains] = {};
    float up_accumulator[1][Schedule::kAccumulatorChains] = {};
    compute_bf16_gemv_rows<Geometry, Schedule>(x, gate_weight, row, warp, lane,
                                               gate_accumulator);
    compute_bf16_gemv_rows<Geometry, Schedule>(x, up_weight, row, warp, lane,
                                               up_accumulator);
    float gate_total = 0.0F;
    float up_total = 0.0F;
#pragma unroll
    for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
        gate_total += gate_accumulator[0][chain];
        up_total += up_accumulator[0][chain];
    }
    gate_total = warp_reduce_sum(gate_total);
    up_total = warp_reduce_sum(up_total);
    if (lane == 0) {
        gate_partials[warp] = gate_total;
        up_partials[warp] = up_total;
    }
    __syncthreads();
    if (warp == 0) {
        gate_total = lane < Schedule::kWarpsPerRow ? gate_partials[lane] : 0.0F;
        up_total = lane < Schedule::kWarpsPerRow ? up_partials[lane] : 0.0F;
        gate_total = warp_reduce_sum(gate_total);
        up_total = warp_reduce_sum(up_total);
        if (lane == 0) {
            const float gate = __bfloat162float(__float2bfloat16_rn(gate_total));
            const float up = __bfloat162float(__float2bfloat16_rn(up_total));
            out[row] = __float2bfloat16_rn((gate / (1.0F + expf(-gate))) * up);
        }
    }
}

void launch_bf16_shared_swiglu_decode(const Tensor& x, const Weight& gate_weight,
                                      const Weight& up_weight, Tensor& out,
                                      cudaStream_t stream) {
    using Geometry = Bf16GemvGeometry<640, 2560>;
    using Schedule = Bf16LinearDecodeSchedule<Geometry>;
    if (x.ne[1] != 1 || gate_weight.n != Geometry::kOutputRows ||
        gate_weight.k != Geometry::kInputRows || up_weight.n != Geometry::kOutputRows ||
        up_weight.k != Geometry::kInputRows || out.numel() != Geometry::kOutputRows) {
        throw std::invalid_argument("BF16 shared SwiGLU decode: invalid exact problem");
    }
    bf16_shared_swiglu_decode_kernel<<<Geometry::kOutputRows, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(gate_weight.qdata),
        static_cast<const __nv_bfloat16*>(up_weight.qdata),
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void launch_bf16_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
#define NINFER_BF16_DECODE(N, K)                                                               \
    if (weight.n == (N) && weight.k == (K)) {                                                  \
        launch_geometry<Bf16GemvGeometry<(N), (K)>>(x, weight, out, stream);                   \
        return;                                                                                 \
    }


    NINFER_BF16_DECODE(320, 10240)
    NINFER_BF16_DECODE(10240, 320)
    NINFER_BF16_DECODE(12288, 2560)
    NINFER_BF16_DECODE(512, 2560)
    NINFER_BF16_DECODE(128, 2560)
    NINFER_BF16_DECODE(2560, 6144)
    NINFER_BF16_DECODE(640, 2560)
    NINFER_BF16_DECODE(10240, 2560)
    NINFER_BF16_DECODE(6144, 2560)
    NINFER_BF16_DECODE(2560, 640)
    NINFER_BF16_DECODE(2560, 2560)
    NINFER_BF16_DECODE(248320, 2560)
    NINFER_BF16_DECODE(117248, 2560)
#undef NINFER_BF16_DECODE
    throw std::invalid_argument("bf16 linear decode: unsupported exact problem");
}

} // namespace ninfer::ops::detail::flash_next
