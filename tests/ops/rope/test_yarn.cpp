// test_yarn.cpp - host-only gate on the YaRN frequency math (src/ops/kernel/yarn.h).
//
// WHY THIS NEEDS AN ORACLE AND NOT A PARITY CHECK: a wrong YaRN frequency table does not crash.
// It silently stops finding things at depth, which is indistinguishable from "long context is
// hard". So every claim below is measured against an independent double-precision implementation
// of the reference formula (HF transformers `_compute_yarn_parameters`), written from the
// definition in this file and sharing no code with the header under test. The disabled path is
// additionally checked against the engine's own shipped kTextRopeInvFrequency table, because that
// table is what the fixed rope instances already rotate with - "the default path is untouched"
// means the generic path this function feeds still agrees with them bit-for-bit in effect.
//
// Host-only: yarn.h compiles its functions for the host when __CUDACC__ is undefined, so the same
// definition the kernel will call is the one measured here.

#include "ops/kernel/yarn.h"
#include "ops/rope/apply/yarn_resolve.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

using ninfer::ops::YarnParams;
using ninfer::ops::YarnRamp;
using ninfer::ops::yarn_attention_scale;
using ninfer::ops::yarn_inv_frequency;
using ninfer::ops::yarn_ramp;

// The engine's compiled-in text RoPE table, transcribed from the LIVE kernel
// (kRopeR64Theta1e7Frequencies in src/ops/rope/kernels/rotate.cuh; the master-generation
// duplicate kTextRopeInvFrequency in src/ops/kernel/rope.cuh is not built and holds the same 32
// values). It is a secondary oracle only: check_shipped_table_matches_math below proves it equals
// 1e7^(-2i/64) to 1e-9, so a transcription slip cannot pass silently.
constexpr double kShipped[32] = {
    1.000000000e+00, 6.042963902e-01, 3.651741273e-01, 2.206734069e-01, 1.333521432e-01,
    8.058421878e-02, 4.869675252e-02, 2.942727176e-02, 1.778279410e-02, 1.074607828e-02,
    6.493816316e-03, 3.924189758e-03, 2.371373706e-03, 1.433012570e-03, 8.659643234e-04,
    5.232991147e-04, 3.162277660e-04, 1.910952975e-04, 1.154781985e-04, 6.978305849e-05,
    4.216965034e-05, 2.548296748e-05, 1.539926526e-05, 9.305720409e-06, 5.623413252e-06,
    3.398208329e-06, 2.053525026e-06, 1.240937761e-06, 7.498942093e-07, 4.531583638e-07,
    2.738419634e-07, 1.654817100e-07,
};

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) { ++failures; }
}

double relative_error(double got, double want) {
    if (want == 0.0) { return std::fabs(got); }
    return std::fabs(got - want) / std::fabs(want);
}

// ------------------------------------------------------------------ independent oracle
// Written from HF transformers `_compute_yarn_parameters` + `_yarn_find_correction_dim` +
// `_yarn_find_correction_range` + `_yarn_linear_ramp_mask`, in double, with no reference to
// yarn.h. `dim` is the full rotary width; the returned table holds dim/2 pair-space entries.
struct Oracle {
    std::vector<double> inv_freq;
    double low  = 0.0;
    double high = 0.0;
    double mscale = 1.0;
};

double oracle_correction_dim(double rotations, int dim, double base, double original_max_position) {
    return (static_cast<double>(dim) *
            std::log(original_max_position / (rotations * 2.0 * 3.14159265358979323846))) /
           (2.0 * std::log(base));
}

Oracle oracle_yarn(double factor, int dim, double base, double original_max_position,
                   double beta_fast, double beta_slow) {
    Oracle out;
    const int half = dim / 2;
    out.low  = std::max(0.0, std::floor(
                                 oracle_correction_dim(beta_fast, dim, base, original_max_position)));
    out.high = std::min(static_cast<double>(dim - 1),
                        std::ceil(oracle_correction_dim(beta_slow, dim, base,
                                                        original_max_position)));
    if (out.high - out.low < 1e-3) { out.high = out.low + 1e-3; }
    out.mscale = factor > 1.0 ? 0.1 * std::log(factor) + 1.0 : 1.0;
    out.inv_freq.resize(static_cast<std::size_t>(half));
    for (int i = 0; i < half; ++i) {
        const double pos_freq      = std::pow(base, 2.0 * i / static_cast<double>(dim));
        const double extrapolation = 1.0 / pos_freq;
        const double interpolation = 1.0 / (factor * pos_freq);
        // `_yarn_linear_ramp_mask` is the interpolation weight: 0 at `low`, 1 at `high`.
        double ramp = (static_cast<double>(i) - out.low) / (out.high - out.low);
        ramp        = std::min(1.0, std::max(0.0, ramp));
        out.inv_freq[static_cast<std::size_t>(i)] = interpolation * ramp +
                                                    extrapolation * (1.0 - ramp);
    }
    return out;
}

YarnParams params_for(double factor) {
    YarnParams p;
    p.factor = static_cast<float>(factor);
    return p;
}

// The transcribed shipped table must itself equal the plain-RoPE formula; otherwise it is not an
// oracle, it is a copy of whatever was typed here.
void check_shipped_table_matches_math() {
    double worst = 0.0;
    for (int i = 0; i < 32; ++i) {
        const double want = std::pow(1.0e7, -2.0 * i / 64.0);
        worst = std::max(worst, relative_error(kShipped[i], want));
    }
    std::printf("  worst relative error of the transcribed shipped table vs 1e7^(-2i/64): %.3e\n",
                worst);
    check(worst < 1.0e-9, "transcribed kTextRopeInvFrequency equals the plain-RoPE formula");
}

// ------------------------------------------------------------------ the disabled path
void check_disabled_is_untouched() {
    for (double factor : {1.0, 0.5}) {
        const YarnParams p = params_for(factor);
        double worst_table = 0.0;
        double worst_math  = 0.0;
        for (int i = 0; i < 32; ++i) {
            const double got = yarn_inv_frequency(p, i);
            worst_table      = std::max(worst_table, relative_error(got, kShipped[i]));
            worst_math       = std::max(worst_math,
                                        relative_error(got, std::pow(1.0e7, -2.0 * i / 64.0)));
        }
        std::printf("  factor %.1f: worst rel. error vs shipped table %.3e, vs double %.3e\n",
                    factor, worst_table, worst_math);
        check(worst_table < 1.0e-6,
              factor == 1.0 ? "factor 1.0 reproduces the engine's shipped table"
                            : "factor 0.5 is treated as disabled and reproduces the table");
        check(worst_math < 1.0e-6, "disabled path matches the plain-RoPE formula");
        check(yarn_attention_scale(p) == 1.0F, "disabled path leaves the attention scale at 1.0");
    }
}

// ------------------------------------------------------------------ the enabled path
void check_enabled_matches_oracle() {
    for (double factor : {2.0, 4.0}) {
        const YarnParams p    = params_for(factor);
        const Oracle oracle   = oracle_yarn(factor, 64, 1.0e7, 262144.0, 32.0, 1.0);
        const YarnRamp ramp   = yarn_ramp(p);
        check(ramp.low == oracle.low && ramp.high == oracle.high,
              "ramp boundaries equal the reference in PAIR space");
        std::printf("  factor %.1f: ramp [%.0f, %.0f]\n", factor, ramp.low, ramp.high);

        double worst = 0.0;
        for (int i = 0; i < 32; ++i) {
            worst = std::max(worst, relative_error(yarn_inv_frequency(p, ramp, i),
                                                   oracle.inv_freq[static_cast<std::size_t>(i)]));
        }
        std::printf("  factor %.1f: worst rel. error vs double oracle over all 32 pairs: %.3e\n",
                    factor, worst);
        check(worst < 1.0e-6, "every pair matches the independent double-precision oracle");

        // Pairs at or below `low` must be extrapolated unchanged; interpolating them is what
        // wrecks short-context quality, and it is the one failure mode a ramp bug produces.
        bool high_kept = true;
        for (int i = 0; i <= static_cast<int>(ramp.low); ++i) {
            if (relative_error(yarn_inv_frequency(p, i), kShipped[i]) > 1.0e-6) {
                high_kept = false;
            }
        }
        check(high_kept, "pairs 0..low are extrapolated unchanged");

        // The lowest pair is fully interpolated, i.e. divided by exactly the factor.
        const double ratio = kShipped[31] / yarn_inv_frequency(p, 31);
        std::printf("  factor %.1f: lowest pair divided by %.6f (want %.1f)\n", factor, ratio,
                    factor);
        check(std::fabs(ratio - factor) < 1.0e-5, "lowest pair is interpolated by 1/factor");

        // Monotone decreasing: the ramp must not reorder the pairs.
        bool monotone = true;
        for (int i = 1; i < 32; ++i) {
            if (!(yarn_inv_frequency(p, i) < yarn_inv_frequency(p, i - 1))) { monotone = false; }
        }
        check(monotone, "frequencies stay strictly decreasing across the ramp");

        const double mscale = yarn_attention_scale(p);
        std::printf("  factor %.1f: attention scale %.9f (oracle %.9f)\n", factor, mscale,
                    oracle.mscale);
        check(relative_error(mscale, oracle.mscale) < 1.0e-7,
              "attention scale is 0.1*ln(factor)+1");
    }
}

// ------------------------------------------------------------------ position-level claim
// The task's numerical claim: for a known factor and position, the YaRN-adjusted angle must match
// an independent double-precision computation. This is where a pair-space/dim-space indexing bug
// becomes visible as a wrong phase rather than a plausible-looking table.
void check_known_factor_and_position() {
    const double factor = 4.0;
    const YarnParams p  = params_for(factor);
    const Oracle oracle = oracle_yarn(factor, 64, 1.0e7, 262144.0, 32.0, 1.0);
    // 262,144 is the native window; 524,288 and 1,048,576 are the extended range the option
    // is meant to reach (native x 2 and native x 4).
    for (long long position : {1LL, 1024LL, 262144LL, 524288LL, 1048576LL}) {
        double worst_angle = 0.0;
        double worst_cos   = 0.0;
        for (int i = 0; i < 32; ++i) {
            const double want_freq  = oracle.inv_freq[static_cast<std::size_t>(i)];
            const double want_angle = static_cast<double>(position) * want_freq;
            const double got_angle  = static_cast<double>(position) * yarn_inv_frequency(p, i);
            // Relative to the angle magnitude, not to the frequency: a phase error is what
            // actually degrades retrieval, and it grows with position.
            worst_angle = std::max(worst_angle, std::fabs(got_angle - want_angle) /
                                                    std::max(1.0, std::fabs(want_angle)));
            worst_cos   = std::max(worst_cos, std::fabs(std::cos(got_angle) - std::cos(want_angle)));
        }
        std::printf("  position %7lld: worst angle error %.3e rad, worst |dcos| %.3e\n", position,
                    worst_angle, worst_cos);
        check(worst_angle < 1.0e-6, "angle at a known position matches the double oracle");
    }
}

// ------------------------------------------------------------------ degenerate inputs
void check_degenerate_ramp() {
    YarnParams p;
    p.factor     = 2.0F;
    p.beta_fast  = 1.0F;  // both bounds land on the same pair: high - low collapses
    p.beta_slow  = 1.0F;
    const YarnRamp ramp = yarn_ramp(p);
    bool finite         = std::isfinite(ramp.low) && std::isfinite(ramp.high) &&
                  ramp.high > ramp.low;
    for (int i = 0; i < 32; ++i) {
        if (!std::isfinite(yarn_inv_frequency(p, ramp, i))) { finite = false; }
    }
    check(finite, "a collapsed ramp stays finite and keeps high > low");
}


// ------------------------------------------------------------------ launch resolution
// The launcher resolves the Op's request into the per-launch device struct
// (src/ops/rope/apply/yarn_resolve.h). That is the code that decides whether the kernel sees a
// YaRN table at all, so it carries the "default path is untouched" claim: at factor 0 or 1 the
// struct must be the default-constructed one, which is what makes the kernel take the compiled-in
// table and apply no scale.
void check_resolution_disabled_is_default() {
    for (float factor : {0.0F, 1.0F, 0.5F}) {
        const auto device = ninfer::ops::detail::resolve_yarn(factor, 262144, 32.0F, 1.0F, 1.0e7F,
                                                              /*axes=*/1, /*rotary_dim=*/64);
        const ninfer::ops::detail::RopeYarnDevice pristine{};
        bool identical = !device.enabled && device.pair_count == 0 && device.scale == 1.0F;
        for (int pair = 0; pair < 32; ++pair) {
            identical = identical && device.inv_frequency[pair] == pristine.inv_frequency[pair];
        }
        std::printf("  factor %.1f: enabled=%d pairs=%d scale=%.1f\n", factor,
                    device.enabled ? 1 : 0, device.pair_count, static_cast<double>(device.scale));
        check(identical, "a disabled factor resolves to the default device struct");
    }
}

void check_resolution_enabled_matches_oracle() {
    for (double factor : {2.0, 4.0}) {
        const auto device = ninfer::ops::detail::resolve_yarn(
            static_cast<float>(factor), 262144, 32.0F, 1.0F, 1.0e7F, /*axes=*/1, /*rotary_dim=*/64);
        check(device.enabled && device.pair_count == 32, "an enabled factor carries 32 pairs");
        const Oracle oracle = oracle_yarn(factor, 64, 1.0e7, 262144.0, 32.0, 1.0);
        double worst        = 0.0;
        for (int pair = 0; pair < 32; ++pair) {
            worst = std::max(worst, relative_error(device.inv_frequency[pair],
                                                   oracle.inv_freq[static_cast<std::size_t>(pair)]));
        }
        std::printf("  factor %.1f: resolved table worst rel. error %.3e, scale %.9f\n", factor,
                    worst, static_cast<double>(device.scale));
        check(worst < 1.0e-6, "the resolved launch table matches the double oracle");
        check(relative_error(device.scale, oracle.mscale) < 1.0e-7,
              "the resolved attention scale matches the oracle");
    }
}

void check_resolution_rejects_unsupported() {
    using ninfer::ops::detail::resolve_yarn;
    const auto throws = [](auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    check(throws([&] { resolve_yarn(2.0F, 262144, 32.0F, 1.0F, 1.0e7F, 1, 128); }),
          "a rotary width other than 64 is rejected, not silently left unscaled");
    check(throws([&] { resolve_yarn(2.0F, 262144, 32.0F, 1.0F, 1.0e7F, 2, 72); }),
          "Vision 2-D with an enabled factor is rejected");
    check(throws([&] { resolve_yarn(2.0F, 0, 32.0F, 1.0F, 1.0e7F, 1, 64); }),
          "a zero native window is rejected");
    check(throws([&] { resolve_yarn(2.0F, 262144, 0.0F, 1.0F, 1.0e7F, 1, 64); }),
          "a non-positive beta is rejected");
    check(!throws([&] { resolve_yarn(2.0F, 262144, 32.0F, 1.0F, 1.0e7F, 3, 64); }),
          "Text MRoPE is accepted");
}

}  // namespace

int main() {
    std::printf("YaRN frequency math (host, double-precision oracle)\n");
    check_shipped_table_matches_math();
    check_disabled_is_untouched();
    check_enabled_matches_oracle();
    check_known_factor_and_position();
    check_degenerate_ramp();
    check_resolution_disabled_is_default();
    check_resolution_enabled_matches_oracle();
    check_resolution_rejects_unsupported();
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures,
                failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
