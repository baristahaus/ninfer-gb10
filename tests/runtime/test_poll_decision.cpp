// The two-rank round's release decision. Host-only: no Device, no Protocol, no engine.
//
// Case 3 is the regression test for the discarded-peer-failure defect: before 629a154d, a peer's
// ExecutionFailed event was polled and thrown away, and the ticket was released on rank 0's event,
// so a rank-1 failure never reached the engine. On the old code this case released the round.
#include "runtime/executor/poll_decision.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

using ninfer::runtime::decide_poll;
using ninfer::runtime::PollDecision;
using ninfer::runtime::PolledRank;
using ninfer::runtime::PollOutcome;

int failures = 0;

void expect(bool condition, const char* label) {
    if (condition) { return; }
    std::cerr << label << '\n';
    ++failures;
}

void expect_outcome(const PollDecision& decision, PollOutcome outcome, const char* label) {
    expect(decision.outcome == outcome, label);
}

void expect_fail_rank(const PollDecision& decision, std::size_t rank, const char* label) {
    expect(decision.outcome == PollOutcome::Fail, label);
    expect(decision.failing_rank == rank, label);
}

PolledRank succeeded(std::uint64_t ticket) { return PolledRank{true, false, ticket}; }
PolledRank failed(std::uint64_t ticket) { return PolledRank{true, true, ticket}; }
PolledRank absent() { return PolledRank{}; }

void case_1_all_succeed() {
    const std::array<PolledRank, 2> ranks{succeeded(7), succeeded(7)};
    const PollDecision decision = decide_poll(ranks);
    expect_outcome(decision, PollOutcome::Release, "case 1: both ranks completed -> Release");
    expect(!decision.ticket_mismatch, "case 1: no mismatch");
}

void case_2_peer_missing() {
    const std::array<PolledRank, 2> ranks{succeeded(7), absent()};
    const PollDecision decision = decide_poll(ranks);
    expect_outcome(decision, PollOutcome::Wait,
                   "case 2: a peer with no event must hold the round -> Wait");
}

void case_3_peer_failed() {
    // THE REGRESSION. Rank 0 completed; the peer failed. The old code released the round on rank
    // 0's event and the peer's failure vanished. The ticket must fail on the peer.
    const std::array<PolledRank, 2> ranks{succeeded(7), failed(7)};
    const PollDecision decision = decide_poll(ranks);
    expect_fail_rank(decision, 1, "case 3: a peer's failure is the ticket's failure -> Fail(1)");
}

void case_4_rank0_failed() {
    const std::array<PolledRank, 2> ranks{failed(7), succeeded(7)};
    const PollDecision decision = decide_poll(ranks);
    expect_fail_rank(decision, 0, "case 4: rank 0 failed -> Fail(0)");
}

void case_5_ticket_mismatch() {
    const std::array<PolledRank, 2> ranks{succeeded(7), succeeded(8)};
    const PollDecision decision = decide_poll(ranks);
    expect_outcome(decision, PollOutcome::Wait,
                   "case 5: a peer on a different ticket must hold the round -> Wait");
    expect(decision.ticket_mismatch, "case 5: the mismatch is reported");
    expect(decision.mismatch_rank == 1, "case 5: the mismatching rank is reported");
}

void case_6_three_ranks_middle_fails() {
    // Three ranks so the failing index is not hard-coded to 1.
    const std::array<PolledRank, 3> ranks{succeeded(7), failed(7), succeeded(7)};
    const PollDecision decision = decide_poll(ranks);
    expect_fail_rank(decision, 1, "case 6: the middle rank's failure is reported on rank 1");
    const std::array<PolledRank, 3> last_fails{succeeded(7), succeeded(7), failed(7)};
    expect_fail_rank(decide_poll(last_fails), 2,
                     "case 6: the last rank's failure is reported on rank 2");
    const std::array<PolledRank, 3> all_succeed{succeeded(7), succeeded(7), succeeded(7)};
    expect_outcome(decide_poll(all_succeed), PollOutcome::Release,
                   "case 6: three ranks all complete -> Release");
}

void case_empty() {
    const std::vector<PolledRank> ranks;
    expect_outcome(decide_poll(ranks), PollOutcome::Wait, "no ranks -> Wait");
    const std::array<PolledRank, 1> one{succeeded(7)};
    expect_outcome(decide_poll(one), PollOutcome::Release, "one rank completes -> Release");
}

} // namespace

int main() {
    case_1_all_succeed();
    case_2_peer_missing();
    case_3_peer_failed();
    case_4_rank0_failed();
    case_5_ticket_mismatch();
    case_6_three_ranks_middle_fails();
    case_empty();
    if (failures != 0) {
        std::cerr << "poll decision test failed with " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
