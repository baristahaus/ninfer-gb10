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

// WHICH PEERS A MISSING RANK-0 EVENT MUST ADVANCE. This is the regression test for the width-2
// deadlock: poll() used to return on a missing rank-0 event without touching any peer, and one
// Program::progress() call can carry rank 0 into a cross-rank collective, so the peer rank 0 was
// waiting on was never advanced. On the old rule every case below that expects a peer to be advanced
// expected nothing.
void case_peers_behind() {
    using ninfer::runtime::peers_behind;
    const auto expect_mask = [](std::uint64_t actual, std::uint64_t wanted, const char* label) {
        expect(actual == wanted, label);
    };

    // Level: nobody is behind, so nobody is advanced. This is the healthy round, and it is what keeps
    // the original 'never run a peer ahead of rank 0' guarantee.
    const std::array<std::size_t, 2> level{129, 129};
    expect_mask(peers_behind(level), 0, "level counts advance no peer");

    // Rank 1 one collective behind rank 0: it has not issued the collective rank 0 waits for.
    const std::array<std::size_t, 2> behind{129, 128};
    expect_mask(peers_behind(behind), std::uint64_t{1} << 1, "a peer one behind is advanced");

    // Rank 1 ahead: it is not waiting on rank 0's collective, and advancing it further would be the
    // 'run a peer ahead' failure the original rule was written to prevent.
    const std::array<std::size_t, 2> ahead{128, 129};
    expect_mask(peers_behind(ahead), 0, "a peer ahead is not advanced");

    // Three ranks, the middle one behind: only that one is advanced, and the index is not hard-coded.
    const std::array<std::size_t, 3> middle{5, 4, 5};
    expect_mask(peers_behind(middle), std::uint64_t{1} << 1, "only the behind rank is advanced");
    const std::array<std::size_t, 3> last_behind{5, 5, 4};
    expect_mask(peers_behind(last_behind), std::uint64_t{1} << 2, "the last behind rank is advanced");

    // A rank with no collective at all (the single-device lane, and a family that issues none): a
    // leader count of zero puts nobody behind it, so a width-1 round behaves exactly as before.
    const std::array<std::size_t, 2> none{0, 0};
    expect_mask(peers_behind(none), 0, "no collectives -> no peer advanced");
    const std::vector<std::size_t> empty;
    expect_mask(peers_behind(empty), 0, "no ranks -> no peer advanced");
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
    case_peers_behind();
    if (failures != 0) {
        std::cerr << "poll decision test failed with " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
