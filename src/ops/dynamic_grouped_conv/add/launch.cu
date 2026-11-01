#include "ops/dynamic_grouped_conv/add/launch.h"
#include "core/device.h"
#include "ops/linear/kernels/q8/q8_geometry.h"
#include "ops/linear/kernels/q8/q8_schedule.cuh"
#include "ops/linear/kernels/q8/q8_launch.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"
#include <cuda_bf16.h>
#include <array>
#include <utility>

namespace ninfer::ops::detail::dynamic_grouped_conv::add {
namespace {

template <int H>
__device__ __forceinline__ void finish_value(int row, int col, int width, float current,
                                             float previous, const __nv_bfloat16* base,
                                             const __nv_bfloat16* delta, __nv_bfloat16* residual) {
    constexpr int kGroupsLocal = H / 16;
    const int index = col * H + row, di = col * 2 * kGroupsLocal + row / 16;
    float value = fmaf(__bfloat162float(base[2 * H + row]) + __bfloat162float(delta[di]),
                       current, __bfloat162float(residual[index]));
    if (col % width != 0)
        value = fmaf(
            __bfloat162float(base[3 * H + row]) + __bfloat162float(delta[di + kGroupsLocal]),
            previous, value);
    residual[index] = __float2bfloat16_rn(value);
}

template <int InputRows, int TileColumns>
constexpr auto select_tiled_schedule() {
    constexpr int Warps =
        InputRows == 4096 ? (TileColumns <= 40 ? 8 : 4) : (TileColumns <= 32 ? 8 : 4);
    constexpr Cache Activation =
        InputRows == 4096 && ((TileColumns > 24 && TileColumns <= 40) || TileColumns > 48)
            ? Cache::cg
            : Cache::ca;
    using Schedule = Q8A16SlicedKMmaSchedule<TileColumns, Warps, 1, Warps == 8 ? 2 : 3,
                                             Q8ScaleAccess::Shared, Activation>;
    return Schedule{};
}

template <int H, int InputRows, int TileColumns>
void tiled_projection(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Geometry = Q8LinearGeometry<H, InputRows>;
    using Schedule = decltype(select_tiled_schedule<InputRows, TileColumns>());
    LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), H};
    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, TileColumns, false>,
        Q8SlicedKIdentityRows>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{},
                               stream);
    CUDA_CHECK(cudaGetLastError());
}

// Live columns stay dynamic; only the eight-column MMA accumulator layout is specialized.
template <int H, int C, std::size_t... I>
constexpr auto make_launchers(std::index_sequence<I...>) {
    return std::array<Q8Launch, sizeof...(I)>{
        &tiled_projection<H, C, 8 * (1 + static_cast<int>(I))>...};
}

constexpr auto k5120_4096  = make_launchers<5120, 4096>(std::make_index_sequence<11>{});
constexpr auto k5120_17408 = make_launchers<5120, 17408>(std::make_index_sequence<11>{});
constexpr auto k2560_4096  = make_launchers<2560, 4096>(std::make_index_sequence<11>{});
constexpr auto k2560_7680  = make_launchers<2560, 7680>(std::make_index_sequence<11>{});

template <int H>
__global__ void finish_kernel(const __nv_bfloat16* projected, const __nv_bfloat16* base,
                              const __nv_bfloat16* delta, __nv_bfloat16* residual, int width) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y;
    if (row >= H) return;
    const int index = col * H + row;
    finish_value<H>(row, col, width, __bfloat162float(projected[index]),
                    col % width ? __bfloat162float(projected[index - H]) : 0.0f, base, delta,
                    residual);
}

template <int H>
void launch_shape(Plan plan, const Tensor& x, const Weight& weight, const Tensor& base,
                  const Tensor& delta, Tensor& residual, Tensor& projected,
                  cudaStream_t stream) {
    const int tokens  = x.ne[1] * x.ne[2];
    const Tensor flat = x.view({x.ne[0], tokens});
    Tensor result     = projected.view({H, tokens});
    switch (plan.projection) {
    case Projection::TiledMma: {
        if constexpr (H == 5120) {
            const auto& launchers = x.ne[0] == 4096 ? k5120_4096 : k5120_17408;
            launchers[plan.tile_index](flat, weight, result, stream);
        } else {
            const auto& launchers = x.ne[0] == 4096 ? k2560_4096 : k2560_7680;
            launchers[plan.tile_index](flat, weight, result, stream);
        }
        break;
    }
    case Projection::MmaK128:
        launch_q8_a16_mma_r64x32_t64_k128_a1(flat, weight, result, stream);
        break;
    }
    const dim3 grid((H + 255) / 256, tokens);
    finish_kernel<H><<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data),
        static_cast<const __nv_bfloat16*>(base.data),
        static_cast<const __nv_bfloat16*>(delta.data),
        static_cast<__nv_bfloat16*>(residual.data), x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void launch(Plan plan, const Tensor& x, const Weight& weight, const Tensor& base,
            const Tensor& delta, Tensor& residual, Tensor& projected, cudaStream_t stream) {
    if (residual.ne[0] == 5120)
        return launch_shape<5120>(plan, x, weight, base, delta, residual, projected, stream);
    if (residual.ne[0] == 2560)
        return launch_shape<2560>(plan, x, weight, base, delta, residual, projected, stream);
    throw std::logic_error("linear dynamic grouped conv add: unsupported hidden width");
}
} // namespace ninfer::ops::detail::dynamic_grouped_conv::add
