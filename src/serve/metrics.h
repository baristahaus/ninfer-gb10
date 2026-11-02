#pragma once

#include "serve/generation_service.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace ninfer::serve {

// Fixed-size request aggregates. Runtime counters remain owned and published by Engine.
class Metrics {
public:
    void configure(std::string model, const EngineOptions& options,
                   const EngineDescription& description, const StartupReport& startup,
                   EngineSnapshot baseline);
    void first_token(const GenerationFirstTokenObservation& observation);
    void done(const GenerationOutcome& outcome);
    void rejected();
    void failed(bool cancelled);
    void response_failed();
    // One HTTP request observed at the pre-routing hook, before protocol mapping.
    void http_request();
    [[nodiscard]] std::string render(const EngineSnapshot& snapshot, bool ready) const;

private:
    struct Histogram {
        // Disjoint bins while recording; cumulative buckets are produced only when scraped.
        std::array<std::uint64_t, 23> bins{};
        double sum = 0.0;
        void observe(double seconds);
    };

    struct Requests {
        std::uint64_t completed         = 0;
        std::uint64_t cancelled         = 0;
        std::uint64_t failed            = 0;
        std::uint64_t rejected          = 0;
        std::uint64_t response_failures = 0;
        std::array<std::uint64_t, 3> constraint_outcomes{};
        std::array<std::uint64_t, 3> constraint_cache{};
        double constraint_prepare_seconds     = 0;
        double constraint_mask_seconds        = 0;
        double constraint_matcher_seconds     = 0;
        std::uint64_t constraint_positions    = 0;
        std::uint64_t constraint_upload_bytes = 0;
        // Semantic finish reasons of settled generation requests.
        std::uint64_t finish_stop_token         = 0;
        std::uint64_t finish_stop_string        = 0;
        std::uint64_t finish_output_limit       = 0;
        std::uint64_t finish_context_capacity   = 0;
        Histogram ttft;
        Histogram duration;
        Histogram queue;
        Histogram prefill;
        Histogram decode;
        // Per-request mean inter-token latency: the generation wall span holds N-1 token
        // intervals, observed once per request with at least two generated tokens.
        Histogram itl;
    };

    std::string model_;
    EngineOptions options_;
    EngineDescription description_;
    StartupReport startup_;
    EngineSnapshot baseline_;
    double start_seconds_ = 0.0;
    mutable std::mutex mutex_;
    Requests requests_;
    // Per-request decode step participation indexed by batch size - 1. Every step of size B
    // is observed once by each of its B requests, so the cohort sum divided by B is the exact
    // number of completed steps at that size.
    std::vector<std::uint64_t> batch_participation_{};
    // Speculative positions (0-based, sized by the startup draft window) aggregated over
    // settled requests.
    std::vector<std::uint64_t> spec_drafted_per_pos_{};
    std::vector<std::uint64_t> spec_accepted_per_pos_{};
    // Transport-layer counter; the pre-routing hook runs outside the request lock.
    std::atomic<std::uint64_t> http_requests_{0};
};

} // namespace ninfer::serve
