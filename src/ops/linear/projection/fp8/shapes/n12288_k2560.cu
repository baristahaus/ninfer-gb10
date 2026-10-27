#include "ops/linear/kernels/fp8/fp8_geometry.h"
#include "ops/linear/kernels/fp8/fp8_schedule.cuh"
#include "ops/linear/projection/fp8/fp8_dense_route.cuh"
#include "ops/linear/projection/fp8/fp8_shapes.h"

// Flash-Next projection [12288, 2560] on the Flash-Next dense FP8 route.

namespace ninfer::ops::detail {
namespace {
using Geometry = Fp8Geometry<12288, 2560>;
using Gemv     = Fp8A16GemvSchedule<8, 2, 16, 4, Fp8CodeCache::Default, 1, 2>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    fp8_dense::launch_dense_a16<Geometry, Gemv>(x, weight, out, stream);
}

bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8ProjN12288K2560{12288, 2560, launch_a16, nullptr, uses_a8};
} // namespace ninfer::ops::detail
