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
    using ninfer::models::qwen3_5::detail::sharded_forward_refusal;

    const std::string refused = sharded_forward_refusal(true);
    expect(!refused.empty(), "a rank with the peer mirrors must be refused");
    expect(refused.find("sharded forward is not wired") != std::string::npos,
           "the refusal names the reason: the sharded forward is not wired");
    expect(refused.find("Refusing the round") != std::string::npos,
           "the refusal says the round is refused");

    const std::string allowed = sharded_forward_refusal(false);
    expect(allowed.empty(), "a rank without the peer mirrors must not be refused");

    if (failures != 0) {
        std::cerr << failures << " assertion(s) failed\n";
        return 1;
    }
    std::cout << "sharded refusal: ok\n";
    return 0;
}
