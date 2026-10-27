#include "ops/linear/kernels/fp8/fp8_geometry.h"
#include "ops/linear/kernels/fp8/fp8_schedule.cuh"
#include "ops/linear/projection/fp8/fp8_dense_route.cuh"
#include "ops/linear/projection/fp8/fp8_shapes.h"

// Flash-Next projection [10240, 320] on the Flash-Next dense FP8 route.

namespace ninfer::ops::detail {
namespace {
using Geometry = Fp8Geometry<10240, 320>;
using Gemv     = Fp8A16GemvSchedule<4, 4, 16, 4, Fp8CodeCache::Streaming, 1, 4>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    fp8_dense::launch_dense_a16<Geometry, Gemv>(x, weight, out, stream);
}

bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8ProjN10240K320{10240, 320, launch_a16, nullptr, uses_a8};
} // namespace ninfer::ops::detail
