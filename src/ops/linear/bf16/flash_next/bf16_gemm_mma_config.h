#pragma once
#include "core/weight.h"

#include "ops/linear/bf16/flash_next/bf16_gemm_mma.cuh"

namespace ninfer::ops::detail::flash_next {

// Measured large-T production schedule for the BF16 computation core. Geometry remains a template
// argument so an exact problem can replace any tile, pipeline, cache, raster, or fragment choice
// without changing either Linear or semantic-Op dispatch.
template <class Geometry>
using Bf16MmaProductionSchedule =
    Bf16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                    Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;

template <class Geometry>
struct Bf16MmaScheduleSelector {
    using Type = Bf16MmaProductionSchedule<Geometry>;
};

template <>
struct Bf16MmaScheduleSelector<Bf16GemvGeometry<10240, 320>> {
    using Type = Bf16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                                 Bf16MmaFragmentPipeline::PingPong,
                                 Bf16MmaRaster::TokenFast>;
};

template <>
struct Bf16MmaScheduleSelector<Bf16GemvGeometry<48, 2560>> {
    using Type = Bf16MmaSchedule<16, 128, 64, 16, 32, 2, 2, Cache::cg, Cache::cg,
                                 Bf16MmaFragmentPipeline::PingPong,
                                 Bf16MmaRaster::TokenFast>;
};

template <>
struct Bf16MmaScheduleSelector<Bf16GemvGeometry<320, 10240>> {
    using Type = Bf16MmaSchedule<32, 64, 64, 16, 32, 4, 2, Cache::cg, Cache::cg,
                                 Bf16MmaFragmentPipeline::PingPong,
                                 Bf16MmaRaster::TokenFast>;
};

template <>
struct Bf16MmaScheduleSelector<Bf16GemvGeometry<2560, 6144>> {
    using Type = Bf16MmaSchedule<128, 128, 64, 32, 32, 2, 1, Cache::cg, Cache::cg,
                                 Bf16MmaFragmentPipeline::PingPong,
                                 Bf16MmaRaster::TokenFast>;
};

template <class Geometry>
using SelectedBf16MmaSchedule = typename Bf16MmaScheduleSelector<Geometry>::Type;

} // namespace ninfer::ops::detail::flash_next
