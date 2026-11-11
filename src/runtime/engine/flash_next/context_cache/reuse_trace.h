#pragma once

// Maintainer-only admission trace for the Flash-Next prefix cache. It answers one question with
// quantities: at the moment a request is admitted, which checkpoints does the catalog offer, what
// identity/length does each have, and which exact predicate rejects it? The trace reads host
// catalog metadata only -- never device memory -- so no CUDA stream ordering applies to it.
//
// Inert unless NINFER_FLASH_NEXT_REUSE_TRACE is set. The variable is read once at first use, so set
// it before the first Engine call. The line count is capped per rank, and every line states the
// cap: a truncated log must never be read as "no further events happened". One THROTTLED line is
// printed when the cap is reached, and nothing after it.
//
// The Flash-Next route runs one Program per rank, so the callers pass rank 0. The counters are a
// per-rank static array, not a process-wide int: one rank must not consume another's serial.

#include <array>
#include <atomic>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace ninfer::flashnext_runtime {

inline constexpr unsigned kReuseTraceLinesPerRank = 256U;

[[nodiscard]] inline bool reuse_trace_enabled() noexcept {
    static const char* enabled = std::getenv("NINFER_FLASH_NEXT_REUSE_TRACE");
    return enabled != nullptr && *enabled != '\0';
}

inline void reuse_trace_line(int rank, const char* format, ...) {
    if (!reuse_trace_enabled()) { return; }
    if (rank < 0 || rank >= 2) { rank = 0; }
    static std::array<std::atomic<unsigned>, 2> emitted{};
    const unsigned index =
        emitted[static_cast<std::size_t>(rank)].fetch_add(1U, std::memory_order_relaxed) + 1U;
    if (index > kReuseTraceLinesPerRank) {
        if (index == kReuseTraceLinesPerRank + 1U) {
            std::fprintf(stderr,
                         "prefix reuse trace: rank %d THROTTLED at %u lines (cap %u per rank); "
                         "further lines suppressed\n",
                         rank, kReuseTraceLinesPerRank, kReuseTraceLinesPerRank);
        }
        return;
    }
    std::fprintf(stderr,
                 "prefix reuse trace: rank %d line %u/%u (cap %u per rank; later lines "
                 "suppressed): ",
                 rank, index, kReuseTraceLinesPerRank, kReuseTraceLinesPerRank);
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fputc('\n', stderr);
}

} // namespace ninfer::flashnext_runtime
