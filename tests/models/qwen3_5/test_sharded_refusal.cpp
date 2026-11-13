// SPDX-FileCopyrightText: ValerioDolci/capy-flash
// SPDX-License-Identifier: Apache-2.0
//
// THE SHARDED-LANE REFUSAL. internal.h documents the peer mirrors' presence as the flag the round
// entry refuses on, because the sharded forward is not wired. That flag was documented and never
// checked until 55dafc84, and the resulting divergence presented as a mailbox give-up one exchange
// behind rather than as an unwired lane.
//
// WHAT THIS COVERS: the predicate and the message -- has_peer_mirrors=true yields the documented
// refusal, false yields none -- which is exactly what RankProgramImpl::progress() reads.
// WHAT IT DOES NOT COVER: the attach path that makes the mirrors present (attach_rank_mirrors needs
// a real sharded program with a mailbox and two devices). Constructing that here would be a fixture
// far larger than the behaviour under test, and the end-to-end verification on the real lane is
// recorded in 55dafc84's commit message instead.

#include "models/qwen3_5/program/internal.h"

#include <iostream>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

} // namespace

int main() {
    using ninfer::models::qwen3_5::detail::kShardedForwardWired;
    using ninfer::models::qwen3_5::detail::sharded_forward_refusal;

    const std::string refused = sharded_forward_refusal(true, false);
    expect(!refused.empty(), "a rank with the peer mirrors must be refused");
    expect(refused.find("sharded forward is not wired") != std::string::npos,
           "the refusal names the reason: the sharded forward is not wired");
    expect(refused.find("Refusing the round") != std::string::npos,
           "the refusal says the round is refused");

    // A rank without the mirrors is the single-device lane: never refused, wired or not.
    expect(sharded_forward_refusal(false, false).empty(),
           "a rank without the peer mirrors must not be refused");
    expect(sharded_forward_refusal(false, true).empty(),
           "a single-device lane with a wired forward must not be refused");

    // A sharded lane whose forward IS wired must run: the refusal must not outlive its reason.
    expect(sharded_forward_refusal(true, true).empty(),
           "a sharded lane with a wired forward must not be refused");

    // THE CASE dee06f5d GOT WRONG. Mirrors present AND the group bundle present (so `sharded_` is
    // non-null, which is what dee06f5d passed as the 'wired' input) but the forward is NOT
    // implemented: dee06f5d's caller passed `sharded_ != nullptr` == true here and the refusal
    // returned empty, so the round would have RUN the unwired forward. The flag says what it means,
    // so this case refuses.
    expect(!sharded_forward_refusal(true, kShardedForwardWired).empty(),
           "mirrors + group bundle + forward NOT wired must refuse (the case dee06f5d ran)");
    expect(sharded_forward_refusal(true, kShardedForwardWired).find(
               "sharded forward is not wired") != std::string::npos,
           "that refusal names the reason");

    if (failures != 0) {
        std::cerr << failures << " assertion(s) failed\n";
        return 1;
    }
    std::cout << "sharded refusal: ok\n";
    return 0;
}
