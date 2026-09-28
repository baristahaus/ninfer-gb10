#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_output.cuh"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/row_major_mma_epilogue.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

// Flash-Next shared expert: one row-scaled FP8 parent [1280,2560] whose gate rows [0,640)
// precede their up rows [640,1280). It runs the Flash-Next dense FP8 route with paired rows: a
// GEMV at T=1, sliced-K Tensor Core tiles through 64 tokens (T=2..4 on Tensor Cores, the same
// per-element profile as the wider MTP verify batches) and tiled Tensor Core GEMMs beyond. The
// schedules follow the dense route's instances; they are starting points, not measured winners.

namespace ninfer::ops::detail {
namespace {

constexpr int kGateUpRows   = 1280;
constexpr int kIntermediate = kGateUpRows / 2;
constexpr int kInputRows    = 2560;
using Gemv                  = Fp8A16GemvSchedule<4, 2, 16, 4, Fp8CodeCache::Streaming, 1, 4>;

} // namespace

void fp8_linear_swiglu_flash_next_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                         cudaStream_t stream) {
    if (x.ne[0] != kInputRows || out.ne[0] != kIntermediate || out.ne[1] != x.ne[1] ||
        weight.n != kGateUpRows || weight.k != kInputRows) {
        throw std::invalid_argument("fp8 linear_swiglu Flash-Next: invalid exact problem");
    }
    const auto p      = fp8_a16_operands(x, weight);
    const auto output = LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const int tokens  = x.ne[1];
    if (tokens == 1) {
        static_assert(Gemv::kRowsPerWarp == 2);
        return launch_fp8_a16_gemv<Fp8ScheduleInstance<Gemv, kInputRows>>(
            p, output, Fp8SwiGluEpilogue{}, stream, Fp8SwiGluRows<1, kIntermediate>{});
    }
    const auto sliced = [&]<int T, int W, int Stages>() {
        using S = Fp8ScheduleInstance<Fp8SlicedInstance<T, W, Stages>, kInputRows>;
        launch_fp8_a16_sliced_k_mma<S>(p, output, Fp8SwiGluEpilogue{}, stream,
                                       SwiGluRowMajorMmaRows<S>{});
    };
    const auto mma = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, kInputRows>;
        launch_fp8_a16_mma<S>(p, output, SwiGluRowMajorMmaEpilogue{}, stream,
                              SwiGluRowMajorMmaRows<S>{});
    };
    if (tokens <= 8) return sliced.template operator()<8, 8, 2>();
    if (tokens <= 16) return sliced.template operator()<16, 8, 2>();
    if (tokens <= 32) return sliced.template operator()<16, 4, 2>();
    if (tokens <= 64) return sliced.template operator()<32, 4, 1>();
    if (tokens <= 128)
        return mma.template operator()<Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>();
    mma.template operator()<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>();
}

} // namespace ninfer::ops::detail
