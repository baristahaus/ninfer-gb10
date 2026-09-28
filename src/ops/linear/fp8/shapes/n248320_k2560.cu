#include "ops/linear/fp8/fp8_flash_next_route.cuh"
#include "ops/linear/fp8/fp8_shapes.h"

// Flash-Next output head.

namespace ninfer::ops::detail {
namespace {
using Route = flash_next::Fp8FlashNextVocabularyRoute<
    2560, Fp8A16GemvSchedule<8, 2, 16, 4, Fp8CodeCache::Streaming, 1, 2>>;

bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8N248320K2560{248320, 2560, Route::launch, nullptr, uses_a8};
} // namespace ninfer::ops::detail
