#pragma once
// The two-rank round's release decision, as a pure function.
//
// A round is released only when EVERY rank has completed the same ticket. The decision is over the
// ranks' polled events, so it is a pure function of those events and belongs here rather than inside
// LocalExecutor, which needs real Devices and cannot be unit-tested without stubbing a core
// contract.
//
// The invariant this encodes is the lockstep criterion: rank 0's event alone is not enough. A peer
// can retire LATER than rank 0 -- a frame whose commit did device work parks in
// ComputeStage::Retiring until an event on its own stream completes, and one rank's stream can
// drain before the other's -- so releasing on rank 0's event alone reaches the next start_frame
// while a peer's compute slot is still parked. Equally, a peer's FAILURE is the ticket's failure:
// discarding it makes a peer failure invisible to the engine.
//
// Two of the outcomes are not the same kind of "wait": `Wait` means no progress yet and the caller
// should retry, while `ticket_mismatch` means a peer reported a DIFFERENT ticket, which is a symptom
// of a sequencing bug and is named rather than silently dropped.
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::runtime {

// One rank's polled result, reduced to the three facts the decision needs. Kept free of the Event
// variant so this can be tested without a Device or a Protocol.
struct PolledRank {
    bool has_event    = false;
    bool failed       = false;
    std::uint64_t ticket_id = 0;
};

enum class PollOutcome {
    Wait,   // no progress this poll; the caller retries
    Release,// every rank completed the same ticket; release rank 0's event
    Fail,   // a rank failed; the ticket fails on that rank
};

struct PollDecision {
    PollOutcome outcome      = PollOutcome::Wait;
    std::size_t failing_rank = 0;     // valid when outcome == Fail
    bool ticket_mismatch     = false; // a peer reported a different ticket; outcome is Wait
    std::size_t mismatch_rank = 0;    // valid when ticket_mismatch
};

// The six rules, in order:
//   1. rank 0 has no event              -> Wait
//   2. rank 0 failed                    -> Fail(0)
//   3. any peer has no event            -> Wait  (back-pressure: do not release the round)
//   4. any peer failed                  -> Fail(that peer)
//   5. any peer's ticket differs        -> Wait, ticket_mismatch
//   6. otherwise                        -> Release
[[nodiscard]] inline PollDecision decide_poll(std::span<const PolledRank> ranks) {
    PollDecision decision;
    if (ranks.empty() || !ranks[0].has_event) { return decision; } // 1
    if (ranks[0].failed) {                                          // 2
        decision.outcome      = PollOutcome::Fail;
        decision.failing_rank = 0;
        return decision;
    }
    for (std::size_t rank = 1; rank < ranks.size(); ++rank) {
        if (!ranks[rank].has_event) { return decision; }            // 3
        if (ranks[rank].failed) {                                   // 4
            decision.outcome      = PollOutcome::Fail;
            decision.failing_rank = rank;
            return decision;
        }
        if (ranks[rank].ticket_id != ranks[0].ticket_id) {          // 5
            decision.ticket_mismatch = true;
            decision.mismatch_rank   = rank;
            return decision;
        }
    }
    decision.outcome = PollOutcome::Release;                        // 6
    return decision;
}

} // namespace ninfer::runtime
