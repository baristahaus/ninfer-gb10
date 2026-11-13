#pragma once
// The collective-participation audit.
//
// A collective -- a mailbox exchange or the gather built on it -- must be reached by every rank the
// group has, the same number of times, in the same order, in the same frame. A collective reachable
// only through a per-rank condition violates that, and the two ways it shows are a deadlock (the
// rank that skipped it never publishes, the rank that ran it spins until the mailbox gives up) or,
// in another arrangement, a rank computing on a gather its peer never performed -- a wrong answer
// with no hang, which is worse because it is silent.
//
// This audit counts collective calls per frame per rank, keyed by the call site's label, and
// compares the two maps at the end of the frame's enqueue phase -- before the device work spins, so
// a mismatch is reported in milliseconds naming the site, instead of after a 65-second give-up
// naming a warp.
//
// DEBUG BUILDS ONLY, and off unless NINFER_COLLECTIVE_AUDIT is set. It is an assertion about a
// contract, not a production check: it must cost nothing in a release build and change no behaviour
// when unset.
#include <cstdlib>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ninfer::core::audit {

#ifndef NDEBUG

// READ THIS BEFORE CITING THE AUDIT'S SILENCE AS A RESULT. Everything in this namespace is
// compiled out under NDEBUG, so a release build contains no audit at all. An audit that is
// COMPILED OUT and an audit that is SILENT are indistinguishable from the outside: both print
// nothing, and `NINFER_COLLECTIVE_AUDIT=1` has no effect on a release binary because the env gate
// itself is inside the #ifndef. A run that "did not fire" in a release build is therefore not a
// negative result -- it is no measurement. Two audit runs in this workstream were read that way
// before the guard was noticed (the per-round audit and the per-frame audit before it), and they
// should not be cited as having excluded anything. To make the audit a measurement, build with
// NDEBUG undefined (a Debug or RelWithDebInfo-with-asserts build) and confirm the run is a debug
// binary first. See docs/operations.md, `NINFER_COLLECTIVE_AUDIT`.

struct CollectiveCounts {
    std::map<std::string, std::size_t> by_site;
};

// One entry per rank. The process owns both ranks' Programs, so one map per rank is enough.
inline CollectiveCounts& counts(int rank) {
    static CollectiveCounts per_rank[2];
    return per_rank[(rank == 1) ? 1 : 0];
}

// The ROUND scope. A per-frame check cannot see a frame LAG: if rank 0 is one frame ahead, each
// frame's maps are equal and the mismatch only exists across the round. That is the failure this
// scope was added for -- the two ranks can be in different frames, which every earlier instrument
// assumed they were not.
inline CollectiveCounts& round_counts(int rank) {
    static CollectiveCounts per_rank[2];
    return per_rank[(rank == 1) ? 1 : 0];
}

inline bool enabled() {
    static const bool value = std::getenv("NINFER_COLLECTIVE_AUDIT") != nullptr;
    return value;
}

inline void reset_frame() {
    counts(0).by_site.clear();
    counts(1).by_site.clear();
}

inline void reset_round() {
    round_counts(0).by_site.clear();
    round_counts(1).by_site.clear();
}

inline void record(int rank, const char* site) {
    if (!enabled()) { return; }
    const char* key = site != nullptr ? site : "?";
    ++counts(rank).by_site[key];
    ++round_counts(rank).by_site[key];
}

[[nodiscard]] inline std::string format(const CollectiveCounts& value) {
    std::string text = "{";
    bool first       = true;
    for (const auto& [site, count] : value.by_site) {
        if (!first) { text += ", "; }
        first = false;
        text += site;
        text += ":";
        text += std::to_string(count);
    }
    text += "}";
    return text;
}

// Called at the end of the frame's enqueue phase. Returns true when the ranks agree; otherwise
// throws with BOTH maps, because a reader needs the counts, not just the fact that they differ.
[[nodiscard]] inline bool verify(const char* where) {
    if (!enabled()) { return true; }
    if (counts(0).by_site == counts(1).by_site) { return true; }
    std::ostringstream message;
    message << "collective audit: the two ranks did not issue the same collectives in this frame ("
            << where << "); rank0 " << format(counts(0)) << " rank1 " << format(counts(1))
            << ". A collective must be unconditional within the frame, or guarded by a quantity "
               "both ranks agree on: a rank that skips one never publishes the flag its peer waits "
               "for, and a rank that runs one its peer skipped computes on an exchange that never "
               "happened.";
    throw std::logic_error(message.str());
}

// The round-scope check: cumulative from the round's start, run when the round releases. This is
// the scope that catches a frame lag, because a lag makes the per-frame maps equal.
[[nodiscard]] inline bool verify_round(const char* where) {
    if (!enabled()) { return true; }
    if (round_counts(0).by_site == round_counts(1).by_site) { return true; }
    std::ostringstream message;
    message << "collective audit (round): the two ranks did not issue the same collectives in this "
               "round ("
            << where << "); rank0 " << format(round_counts(0)) << " rank1 "
            << format(round_counts(1))
            << ". The two ranks can be in DIFFERENT FRAMES: a rank one frame ahead has issued a "
               "collective its peer has not reached, and the per-frame maps look equal because each "
               "frame matched. The lockstep criterion -- a round is released only when every rank "
               "has completed it -- is the invariant; this count is how it is enforced.";
    throw std::logic_error(message.str());
}

#else

inline bool enabled() { return false; }
inline void reset_frame() {}
inline void reset_round() {}
inline void record(int, const char*) {}
[[nodiscard]] inline bool verify(const char*) { return true; }
[[nodiscard]] inline bool verify_round(const char*) { return true; }

#endif

} // namespace ninfer::core::audit
