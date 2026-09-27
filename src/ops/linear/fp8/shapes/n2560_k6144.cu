#include "ops/linear/fp8/fp8_a16_route.cuh"
#include "ops/linear/fp8/fp8_shapes.h"

// Flash-Next GDN and QSA output projections.

namespace ninfer::ops::detail {
namespace {
using Route =
    Fp8A16Route<Fp8Geometry<2560, 6144>, Fp8GemvSchedule<8, 2, 16, 4, Fp8CodeCache::Default, 2, 2>>;

bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8N2560K6144{2560, 6144, Route::launch, nullptr, uses_a8};
} // namespace ninfer::ops::detail
