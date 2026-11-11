#pragma once

// DIAGNOSTIC (NINFER_ROUND_TRACE=1): the round-dispatch trace.
//
// The two-GPU lane hands one round to two Programs from one coordinator thread. This trace makes
// that handoff visible: every executor submit (ticket, frame kind, and the rank each `start` is
// actually called on), every per-rank poll result, every per-rank commit, and one-time gate lines
// that name where a single rank's return or result releases the next round.
//
// UNSET, EVERY CALL BELOW IS INERT. `round_trace_enabled()` is one cached getenv; callers guard
// their formatting with it, and no CUDA or Program state is touched. SET, the lines go to stderr as
// `[round-trace] ...`.
//
// POLL THROTTLE. A poll is high frequency (the engine worker loop yields and polls while a round is
// outstanding), so poll lines are throttled by LocalExecutor::trace_poll_result: the first 8 polls
// per rank print in full, then at most one line per rank per 500 ms; a poll that carries an event
// or a recorded failure always prints. Every throttled line states the throttle, because a
// truncated log must never be read as a system fact.
//
// RANK. Where both are available, a line prints `rank` (the Program's rank in the group) and `dev`
// (that rank's CUDA device index). Where only a device is available, the device index IS the rank
// on this lane: LocalExecutor drives its workers in rank order and the two-GPU lane's device list
// is {0, 1}.

#include "runtime/executor/contract.h"

#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ninfer::runtime {

inline bool round_trace_enabled() noexcept {
    static const bool enabled = std::getenv("NINFER_ROUND_TRACE") != nullptr;
    return enabled;
}

inline std::uint64_t round_trace_now_ns() noexcept {
    using Clock = std::chrono::steady_clock;
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count());
}

inline const char* round_trace_frame_kind_name(FrameKind kind) noexcept {
    return kind == FrameKind::Compute ? "compute" : "context";
}

// One line, one write: concurrent writers cannot interleave halves of two lines. Callers that
// format arguments should check round_trace_enabled() first; this function checks again so a
// missed guard cannot print while the variable is unset. A line that does not fit the buffer ends
// with `[truncated]` rather than looking complete.
inline void round_trace(const char* format, ...) {
    if (!round_trace_enabled()) { return; }
    char line[1024];
    const int prefix = std::snprintf(line, sizeof(line), "[round-trace] ");
    if (prefix < 0) { return; }
    std::size_t used = static_cast<std::size_t>(prefix);
    if (used >= sizeof(line) - 1) { return; }
    va_list args;
    va_start(args, format);
    const int body = std::vsnprintf(line + used, sizeof(line) - used, format, args);
    va_end(args);
    if (body < 0) { return; }
    const std::size_t body_size = static_cast<std::size_t>(body);
    const bool truncated       = body_size >= sizeof(line) - used - 1;
    if (truncated) {
        static constexpr char marker[] = " [truncated]";
        const std::size_t marker_size  = sizeof(marker) - 1;
        std::size_t at                 = sizeof(line) - 1 - marker_size;
        if (at < used) { at = used; }
        std::memcpy(line + at, marker, marker_size);
        used = at + marker_size;
    } else {
        used += body_size;
    }
    line[used++] = '\n';
    (void)std::fwrite(line, 1, used, stderr);
}

} // namespace ninfer::runtime
