#include "ops/linear/fp8/fp8_a16_route.cuh"
#include "ops/linear/fp8/fp8_shapes.h"

// Flash-Next QSA packed query/gate projection.

namespace ninfer::ops::detail {
namespace {
using Route = Fp8A16Route<Fp8Geometry<12288, 2560>,
                          Fp8GemvSchedule<8, 2, 16, 4, Fp8CodeCache::Default, 1, 2>>;

bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8N12288K2560{12288, 2560, Route::launch, nullptr, uses_a8};
} // namespace ninfer::ops::detail
