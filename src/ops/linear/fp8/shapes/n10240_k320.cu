#include "ops/linear/fp8/fp8_flash_next_route.cuh"
#include "ops/linear/fp8/fp8_shapes.h"

// Flash-Next HyperConnection up projection. At T=1 each row is one predicated 16-byte phase
// (20 of 32 lanes); the schedule is a starting point, not a measured winner.

namespace ninfer::ops::detail {
namespace {
using Geometry = Fp8Geometry<10240, 320>;
using Gemv     = Fp8A16GemvSchedule<4, 4, 16, 4, Fp8CodeCache::Streaming, 1, 4>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    flash_next::launch_fp8_dense_linear<Geometry, Gemv>(x, weight, out, stream);
}

bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8N10240K320{10240, 320, launch_a16, nullptr, uses_a8};
} // namespace ninfer::ops::detail
