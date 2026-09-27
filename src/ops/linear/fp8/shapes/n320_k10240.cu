#include "ops/linear/fp8/fp8_a16_route.cuh"
#include "ops/linear/fp8/fp8_shapes.h"

// Flash-Next HyperConnection low-rank down projection. Four rows per CTA keep 80 CTAs
// streaming the 10240-wide rows; the 64-row tail tile divides the 320 output rows.

namespace ninfer::ops::detail {
namespace {
using Route = Fp8A16Route<Fp8Geometry<320, 10240>,
                          Fp8GemvSchedule<4, 1, 16, 4, Fp8CodeCache::Streaming, 4, 4>,
                          Fp8A16GemmSchedule<64, 64, 64, 64, 16, 2, 2>>;

bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8N320K10240{320, 10240, Route::launch, nullptr, uses_a8};
} // namespace ninfer::ops::detail
