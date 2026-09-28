#include "ops/linear/fp8/fp8_flash_next_route.cuh"
#include "ops/linear/fp8/fp8_shapes.h"

// Flash-Next QSA packed query/gate projection.

namespace ninfer::ops::detail {
namespace {
using Geometry = Fp8Geometry<12288, 2560>;
using Gemv     = Fp8A16GemvSchedule<8, 2, 16, 4, Fp8CodeCache::Default, 1, 2>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    flash_next::launch_fp8_dense_linear<Geometry, Gemv>(x, weight, out, stream);
}

bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8N12288K2560{12288, 2560, launch_a16, nullptr, uses_a8};
} // namespace ninfer::ops::detail
