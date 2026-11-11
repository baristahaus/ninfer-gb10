// SPDX-FileCopyrightText: ValerioDolci/capy-flash
// SPDX-License-Identifier: Apache-2.0
#pragma once

// YaRN (Yet another RoPE extensioN) NTK-by-parts frequency scaling, device side.
//
// Adapted from capy-flash src/ops/kernel/yarn.h @ ccaf64cd (2026-11-07, "feat(rope): opt-in
// YaRN context extension (--rope-yarn-factor)"), which follows the reference HF transformers
// `_compute_yarn_parameters`. capy keeps that code on the host because its text RoPE reads a
// compiled-in __constant__ table, so enabling YaRN there is a cudaMemcpyToSymbol of 32 floats.
// Our live generic rope kernel already derives its frequency on the device
// (`powf(theta, -2*pair/rotary_dim)`, src/ops/rope/kernels/rotate.cuh rope_generic_kernel), so
// the scaled frequency belongs in that same expression: one definition shared by the kernel and
// by the host-side test, with no constant table to install or keep in sync.
//
// SCOPE: text only. The rotary pairs this scales are the D256R64 / MRoPE-D256R64 ones
// (rotary_dim 64, theta 1e7), whose 32 frequencies are the ones YaRN's beta bounds are defined
// over. The axial Vision geometry (D72R72) uses its own effective dimension and must keep its
// compiled-in table; a call site gates on `factor > 1` and never reaches this for Vision.
//
// THE ALGORITHM. Plain RoPE extrapolates every frequency, which breaks past the trained window;
// pure position interpolation (divide all frequencies by the factor) preserves long range but
// crushes local detail. YaRN splits the rotary pairs by wavelength:
//   - high-frequency pairs (wavelength << trained window) are EXTRAPOLATED unchanged; they carry
//     local ordering, and interpolating them is what destroys short-context quality;
//   - low-frequency pairs (wavelength >> trained window) are INTERPOLATED by 1/factor, because
//     the model never saw those phases and they must be compressed into the range it did see;
//   - a linear ramp between the two boundaries, so nothing changes discontinuously.
// The boundaries are `beta_fast`/`beta_slow` expressed as "number of full rotations across the
// original window"; 32 and 1 are the paper's values and Qwen's defaults.
//
// Both halves matter. YaRN also scales attention by `mscale`, and omitting it looks exactly like
// the frequency scaling having failed (diffuse attention, poor deep retrieval). See
// yarn_attention_scale below for where the reference applies it.
//
// Disabled path. `factor <= 1` returns plain RoPE and the same float expression the generic
// kernel uses today, so a caller that has not opted in cannot change a single bit.

#include <cmath>

#if defined(__CUDACC__)
#    define NINFER_YARN_HD __host__ __device__
#else
#    define NINFER_YARN_HD
#endif

namespace ninfer::ops {

// One model's YaRN configuration. Defaults are Qwen3.8-27B's: 262,144 native positions,
// rope_theta 1e7, partial_rotary_factor 0.25 of head_dim 256 -> rotary_dim 64 (32 pairs, which
// is exactly the width of kTextRopeInvFrequency). A call site must supply these from
// ModelConfig rather than rely on the defaults.
struct YarnParams {
    float factor              = 1.0F;    // 1.0 = disabled
    int original_max_position = 262144;  // the model's TRUE native window, never scaled
    float beta_fast           = 32.0F;
    float beta_slow           = 1.0F;
    float theta               = 1.0e7F;
    int rotary_dim            = 64;      // full rotary width; the table holds rotary_dim/2 pairs
};

// The ramp boundaries as PAIR indices: pairs below `low` extrapolate, pairs above `high`
// interpolate, pairs between ramp linearly. Kept in double because the reference floors one
// boundary and ceils the other, and a float error at an integer boundary would silently move the
// whole extrapolate/interpolate split.
struct YarnRamp {
    double low  = 0.0;
    double high = 1.0;
};

// The pair index whose wavelength completes `rotations` full turns across the original window.
// `dim` is the FULL rotary width; the result is already in pair space because the 2*log(base)
// denominator cancels the factor of two in the pair->dimension mapping.
NINFER_YARN_HD inline double yarn_correction_dim(double rotations, int dim, double base,
                                                 int original_max_position) {
    constexpr double kTwoPi = 6.283185307179586;
    return (static_cast<double>(dim) *
            ::log(static_cast<double>(original_max_position) / (rotations * kTwoPi))) /
           (2.0 * ::log(base));
}

// The reference's `_yarn_find_correction_range`: floor the fast bound, ceil the slow one, clamp
// to the pair range, and widen a degenerate ramp instead of dividing by zero.
NINFER_YARN_HD inline YarnRamp yarn_ramp(const YarnParams& p) {
    const double fast = yarn_correction_dim(static_cast<double>(p.beta_fast), p.rotary_dim,
                                            static_cast<double>(p.theta), p.original_max_position);
    const double slow = yarn_correction_dim(static_cast<double>(p.beta_slow), p.rotary_dim,
                                            static_cast<double>(p.theta), p.original_max_position);
    YarnRamp ramp;
    ramp.low  = ::floor(fast) < 0.0 ? 0.0 : ::floor(fast);
    const double limit = static_cast<double>(p.rotary_dim) - 1.0;
    ramp.high = ::ceil(slow) > limit ? limit : ::ceil(slow);
    if (ramp.high - ramp.low < 1.0e-3) { ramp.high = ramp.low + 1.0e-3; }
    return ramp;
}

// Inverse frequency of pair `pair` with a precomputed ramp. Use this form from a kernel that
// emits every pair in one CTA: the ramp costs four double logarithms.
NINFER_YARN_HD inline float yarn_inv_frequency(const YarnParams& p, YarnRamp ramp, int pair) {
    const double base     = static_cast<double>(p.theta);
    const double exponent = 2.0 * static_cast<double>(pair) / static_cast<double>(p.rotary_dim);
    // Disabled: keep the live generic text kernel's own float expression
    // (-2.0F * pair / rotary_dim, src/ops/rope/kernels/rotate.cuh generic_axis_frequency) so the
    // default path is bit-identical to the engine before YaRN existed. 2*pair/rotary_dim is exact
    // for every power-of-two rotary_dim, so the double intermediate rounds to the same float.
    if (!(p.factor > 1.0F)) { return ::powf(p.theta, -static_cast<float>(exponent)); }
    const double extrapolation = 1.0 / ::pow(base, exponent);
    double ramp_weight         = (static_cast<double>(pair) - ramp.low) / (ramp.high - ramp.low);
    if (ramp_weight < 0.0) { ramp_weight = 0.0; }
    if (ramp_weight > 1.0) { ramp_weight = 1.0; }
    const double interpolation = extrapolation / static_cast<double>(p.factor);
    return static_cast<float>(interpolation * ramp_weight + extrapolation * (1.0 - ramp_weight));
}

// Convenience form: one pair, ramp computed here.
NINFER_YARN_HD inline float yarn_inv_frequency(const YarnParams& p, int pair) {
    return yarn_inv_frequency(p, yarn_ramp(p), pair);
}

// YaRN's second half: attention temperature. The reference's `get_mscale` is
// `0.1 * mscale * ln(scale) + 1` for scale > 1, with mscale 1, and HF multiplies it into cos and
// sin (`cos = emb.cos() * attention_scaling`), NOT into the softmax scale. Because RoPE is
// linear, scaling both cos and sin by m scales the rotated q and k each by m, so the effective
// logit multiplier is m^2. That is the reference behaviour and the values the checkpoints were
// tuned against; a single factor of m here would be a silent quality regression, so do not
// "simplify" it to the softmax scale.
NINFER_YARN_HD inline float yarn_attention_scale(const YarnParams& p) {
    if (!(p.factor > 1.0F)) { return 1.0F; }
    return static_cast<float>(0.1 * ::log(static_cast<double>(p.factor)) + 1.0);
}

}  // namespace ninfer::ops

#undef NINFER_YARN_HD
