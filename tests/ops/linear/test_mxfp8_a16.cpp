#include "core/weight.h"
#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

// The MXFP8 A16 fallback route: GEMV at T=1, the tiled dequant MMA beyond.
// Shapes are the QAD recipe's MXFP8 projection sites, including the partial
// 48-row GDN control projections and the K=320 post-mixer gate/up leaves
// (K % 256 != 0, so T=1 must route through the MMA path). All K values are
// whole MMA K tiles (K % 64 == 0), as the layout guarantees for these sites.
std::vector<Invocation> a16_calls() {
    std::vector<Invocation> calls;
    for (int t = 1; t <= 48; ++t) calls.push_back({t});
    for (int t : {63, 64, 65, 95, 96, 97, 127, 128, 129, 300, 1024}) calls.push_back({t});
    for (int t : {3, 8, 24, 41, 97})
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    calls.push_back({1, CallForm::A16Convenience});
    for (int t : {64, 97, 129})
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    return calls;
}

int run_mxfp8_a16() {
    int failures = 0;
    failures += run_shape("MXFP8_A16", ActivationCompute::A16, make_mxfp8_weight,
                          {48, 2560, 831U, Comparison::Sampled, true, a16_calls()});
    failures += run_shape("MXFP8_A16", ActivationCompute::A16, make_mxfp8_weight,
                          {512, 2560, 833U, Comparison::Sampled, true, a16_calls()});
    failures += run_shape("MXFP8_A16", ActivationCompute::A16, make_mxfp8_weight,
                          {640, 2560, 837U, Comparison::Sampled, true, a16_calls()});
    failures += run_shape("MXFP8_A16", ActivationCompute::A16, make_mxfp8_weight,
                          {2560, 6144, 839U, Comparison::Sampled, true, a16_calls()});
    failures += run_shape("MXFP8_A16", ActivationCompute::A16, make_mxfp8_weight,
                          {6144, 2560, 841U, Comparison::Sampled, true, a16_calls()});
    failures += run_shape("MXFP8_A16", ActivationCompute::A16, make_mxfp8_weight,
                          {10240, 2560, 843U, Comparison::Sampled, true, a16_calls()});
    failures += run_shape("MXFP8_A16", ActivationCompute::A16, make_mxfp8_weight,
                          {12288, 2560, 847U, Comparison::Sampled, true, a16_calls()});
    // The post-mixer gate/up leaves are K=320: not a whole gemv warp phase, so
    // T=1 must route through the tiled MMA path and stay correct there.
    failures += run_shape("MXFP8_A16", ActivationCompute::A16, make_mxfp8_weight,
                          {10240, 320, 853U, Comparison::Sampled, true, a16_calls()});
    failures += run_shape("MXFP8_A16", ActivationCompute::A16, make_mxfp8_weight,
                          {320, 10240, 859U, Comparison::Sampled, true, a16_calls()});
    // The shared-expert down projection is K=640: not a whole gemv warp phase
    // either, so it also exercises the T=1 MMA route.
    failures += run_shape("MXFP8_A16", ActivationCompute::A16, make_mxfp8_weight,
                          {2560, 640, 863U, Comparison::Sampled, true, a16_calls()});
    failures += run_shape("MXFP8_A16", ActivationCompute::A16, make_mxfp8_weight,
                          {640, 2560, 877U, Comparison::Sampled, true, a16_calls()});

    constexpr std::array policies{ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8,
                                  ops::LinearPolicy::AllowA4};
    for (const ops::LinearPolicy policy : policies) {
        try {
            const std::size_t capacity = ops::linear_workspace_capacity_bytes(
                QType::MXFP8, 2560, 2560, policy, 1, 2048);
            if (capacity != 0) {
                std::cerr << "MXFP8 A16 route reported nonzero workspace\n";
                ++failures;
            }
        } catch (const std::exception& error) {
            std::cerr << "MXFP8 policy was rejected: " << error.what() << '\n';
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (!cuda_available()) { return 77; }
    const int failures = run_mxfp8_a16();
    if (failures != 0) { std::cerr << "MXFP8 A16 failures: " << failures << '\n'; }
    return failures == 0 ? 0 : 1;
}
