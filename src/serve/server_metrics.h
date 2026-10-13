#pragma once

#include "serve/generation_service.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <string_view>

namespace ninfer::serve {

// Serving-layer telemetry counters rendered as Prometheus text on GET /metrics. Monotonic
// counters are delta-sampled by the consumer (EngineMetrics, scrape agents); gauges are read
// from Engine snapshots at render time. The serving layer owns the request-observed counters
// (tokens, latency, speculative acceptance, cache hits); the Engine owns execution gauges
// (batch occupancy, pool occupancy, catalogued prompts, state-slot occupancy).
class ServerMetrics {
public:
    // Completed generation request: the outcome carries prompt/completion/reasoning tokens,
    // prefix-cache hits, speculative drafted/accepted/rounds, and per-request timing.
    void record_request_done(const GenerationOutcome& outcome);

    // Requests rejected before generation (validation, admission) and requests that failed
    // during generation or response streaming.
    void record_request_rejected();
    void record_request_failure();

    // Prometheus text (text/plain; version=0.0.4) with ninfer:-prefixed counters and gauges.
    [[nodiscard]] std::string render_prometheus(const ninfer::RuntimeStats& stats,
                                                const ninfer::MemorySummary& memory,
                                                std::string_view model_id) const;

private:
    std::atomic<std::uint64_t> requests_total_{0};
    std::atomic<std::uint64_t> requests_rejected_total_{0};
    std::atomic<std::uint64_t> requests_failed_total_{0};
    std::atomic<std::uint64_t> prompt_tokens_total_{0};
    std::atomic<std::uint64_t> completion_tokens_total_{0};
    std::atomic<std::uint64_t> reasoning_tokens_total_{0};
    std::atomic<std::uint64_t> cached_tokens_total_{0};
    std::atomic<std::uint64_t> mtp_rounds_total_{0};
    std::atomic<std::uint64_t> mtp_drafted_total_{0};
    std::atomic<std::uint64_t> mtp_accepted_total_{0};
    std::atomic<std::uint64_t> latency_requests_{0};
    std::atomic<std::uint64_t> request_latency_seconds_millis_{0};
    std::atomic<std::uint64_t> ttft_seconds_millis_{0};
    std::atomic<std::uint64_t> prefill_seconds_millis_{0};
    std::atomic<std::uint64_t> decode_seconds_millis_{0};
};

} // namespace ninfer::serve
