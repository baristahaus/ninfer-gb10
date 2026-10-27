#include "ops/linear/fp8/fp8_flash_next_route.cuh"
#include "ops/linear/fp8/fp8_shapes.h"

// Flash-Next shared-expert down projection. At T=1 each row is one full 512-value phase and a
// predicated 128-value tail; the schedule is a starting point, not a measured winner.

namespace ninfer::ops::detail {
namespace {
using Geometry = Fp8Geometry<2560, 640>;
using Gemv     = Fp8A16GemvSchedule<4, 2, 16, 4, Fp8CodeCache::Streaming, 1, 4>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    flash_next::launch_fp8_dense_linear<Geometry, Gemv>(x, weight, out, stream);
}

bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8N2560K640{2560, 640, launch_a16, nullptr, uses_a8};
} // namespace ninfer::ops::detail
