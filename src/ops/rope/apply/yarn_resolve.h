#pragma once

// Host-side resolution of the rope Op's opt-in YaRN request into the per-launch device inputs.
//
// Everything that can be decided on the host is decided here so the kernel only selects and
// applies: the reference's ramp bounds cost four double logarithms and the per-pair frequency one
// double pow, none of which belongs in a per-token CTA. Kept header-only and free of core device
// types so the host gate (tests/ops/rope/test_yarn.cpp) can exercise the exact code the launcher
// runs.

#include "ops/kernel/yarn.h"
#include "ops/rope/kernels/yarn_device.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// A disabled value is returned for `factor <= 1`, which is the state the kernel treats as plain
// RoPE: it reads the compiled-in table and applies no scale, bit for bit as before YaRN existed.
//
// Vision 2-D has its own effective dimension (36, not 72) and its own table, and the DFlash
// D128R128 drafter is a separate component, so neither is scaled. Vision with an enabled factor is
// rejected rather than silently ignored, because scaling it would be a caller error that no
// result would reveal.
inline RopeYarnDevice resolve_yarn(float factor, std::int32_t original_max_position,
                                   float beta_fast, float beta_slow, float theta, int axes,
                                   int rotary_dim) {
    RopeYarnDevice device;
    if (!(factor > 1.0F)) { return device; }
    if (!std::isfinite(factor) || !(beta_fast > 0.0F) || !(beta_slow > 0.0F) ||
        original_max_position <= 0 || !(theta > 0.0F) || !std::isfinite(theta)) {
        throw std::invalid_argument("rope: invalid YaRN parameters");
    }
    if (axes == 2) { throw std::invalid_argument("rope: YaRN does not apply to Vision 2-D RoPE"); }
    // The NTK-by-parts table is the 32 pairs of the D256R64 text geometry; another rotary width
    // would need its own table and its own beta bounds.
    if (rotary_dim != 64) { throw std::invalid_argument("rope: YaRN requires rotary_dim=64"); }

    YarnParams params;
    params.factor                = factor;
    params.original_max_position = original_max_position;
    params.beta_fast             = beta_fast;
    params.beta_slow             = beta_slow;
    params.theta                 = theta;
    params.rotary_dim            = rotary_dim;
    const YarnRamp ramp = yarn_ramp(params);
    for (int pair = 0; pair < 32; ++pair) {
        device.inv_frequency[pair] = yarn_inv_frequency(params, ramp, pair);
    }
    device.scale      = yarn_attention_scale(params);
    device.pair_count = 32;
    device.enabled    = true;
    return device;
}

} // namespace ninfer::ops::detail
