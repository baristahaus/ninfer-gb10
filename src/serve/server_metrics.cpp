#include "serve/server_metrics.h"

#include "core/paged_kv_cache.h"

#include <cstdio>

namespace ninfer::serve {

namespace {

// Sub-second timing sums are accumulated in integral milliseconds and rendered as seconds.
[[nodiscard]] constexpr std::uint64_t seconds_to_millis(double seconds) noexcept {
    return seconds < 0.0 ? 0U : static_cast<std::uint64_t>(seconds * 1000.0);
}

void append_metric(std::string& out, std::string_view type, std::string_view name,
                   std::string_view help, std::uint64_t value) {
    out += "# HELP ";
    out.append(name);
    out += ' ';
    out.append(help);
    out += '\n';
    out += "# TYPE ";
    out.append(name);
    out += ' ';
    out.append(type);
    out += '\n';
    out.append(name);
    out += ' ';
    out += std::to_string(value);
    out += '\n';
}

} // namespace

void ServerMetrics::record_request_done(const GenerationOutcome& outcome) {
    requests_total_.fetch_add(1, std::memory_order_relaxed);
    prompt_tokens_total_.fetch_add(static_cast<std::uint64_t>(outcome.prompt_tokens),
                                   std::memory_order_relaxed);
    completion_tokens_total_.fetch_add(static_cast<std::uint64_t>(outcome.completion_tokens),
                                       std::memory_order_relaxed);
    reasoning_tokens_total_.fetch_add(static_cast<std::uint64_t>(outcome.reasoning_tokens),
                                       std::memory_order_relaxed);
    cached_tokens_total_.fetch_add(outcome.metrics.prefix_cache_hit_tokens,
                                   std::memory_order_relaxed);
    mtp_rounds_total_.fetch_add(outcome.metrics.speculative_rounds, std::memory_order_relaxed);
    mtp_drafted_total_.fetch_add(outcome.metrics.speculative_draft_tokens,
                                 std::memory_order_relaxed);
    mtp_accepted_total_.fetch_add(outcome.metrics.speculative_accepted_tokens,
                                  std::memory_order_relaxed);
    latency_requests_.fetch_add(1, std::memory_order_relaxed);
    request_latency_seconds_millis_.fetch_add(seconds_to_millis(outcome.metrics.total_seconds),
                                              std::memory_order_relaxed);
    ttft_seconds_millis_.fetch_add(seconds_to_millis(outcome.metrics.ttft_seconds),
                                   std::memory_order_relaxed);
    prefill_seconds_millis_.fetch_add(seconds_to_millis(outcome.metrics.prefill_seconds),
                                      std::memory_order_relaxed);
    decode_seconds_millis_.fetch_add(seconds_to_millis(outcome.metrics.decode_seconds),
                                     std::memory_order_relaxed);
}

void ServerMetrics::record_request_rejected() {
    requests_rejected_total_.fetch_add(1, std::memory_order_relaxed);
}

void ServerMetrics::record_request_failure() {
    requests_failed_total_.fetch_add(1, std::memory_order_relaxed);
}

std::string ServerMetrics::render_prometheus(const ninfer::RuntimeStats& stats,
                                             const ninfer::MemorySummary& memory,
                                             std::string_view model_id) const {
    const std::uint64_t latency_count = latency_requests_.load(std::memory_order_relaxed);
    const std::uint64_t latency_millis =
        request_latency_seconds_millis_.load(std::memory_order_relaxed);
    const std::uint64_t ttft_millis = ttft_seconds_millis_.load(std::memory_order_relaxed);
    const std::uint64_t prefill_millis = prefill_seconds_millis_.load(std::memory_order_relaxed);
    const std::uint64_t decode_millis = decode_seconds_millis_.load(std::memory_order_relaxed);

    const std::uint64_t occupied_main_kv_tokens =
        static_cast<std::uint64_t>(stats.device_main_kv_occupied_pages) *
        static_cast<std::uint64_t>(ninfer::kPagedKVPageSize);
    const std::uint64_t pool_tokens      = memory.kv_capacity;
    const std::uint64_t pool_free_tokens = occupied_main_kv_tokens < pool_tokens
                                               ? pool_tokens - occupied_main_kv_tokens
                                               : 0;

    std::string out;
    out.reserve(3072);

    // The model identity is a labeled gauge so metric scrapers can attribute samples without a
    // side-channel; EngineMetrics reads the model_name label from it.
    out += "# HELP ninfer:model_info Serving model identity.\n";
    out += "# TYPE ninfer:model_info gauge\n";
    out += "ninfer:model_info{model_name=\"";
    out.append(model_id);
    out += "\"} 1\n";

    append_metric(out, "counter", "ninfer:requests_total", "Completed generation requests.",
                   requests_total_.load(std::memory_order_relaxed));
    append_metric(out, "counter", "ninfer:requests_rejected_total",
                   "Requests rejected before generation.",
                   requests_rejected_total_.load(std::memory_order_relaxed));
    append_metric(out, "counter", "ninfer:requests_failed_total",
                   "Requests that failed during generation or response streaming.",
                   requests_failed_total_.load(std::memory_order_relaxed));
    append_metric(out, "counter", "ninfer:prompt_tokens_total",
                   "Prompt tokens accepted at the API boundary.",
                   prompt_tokens_total_.load(std::memory_order_relaxed));
    append_metric(out, "counter", "ninfer:computed_prefill_tokens_total",
                   "Prompt tokens the Engine actually prefilled; reused checkpoint-prefix "
                   "tokens are excluded.",
                   stats.computed_prefill_tokens);
    append_metric(out, "counter", "ninfer:generation_tokens_total",
                   "Completion tokens delivered at the API boundary.",
                   completion_tokens_total_.load(std::memory_order_relaxed));
    append_metric(out, "counter", "ninfer:reasoning_tokens_total",
                   "Reasoning (thinking) tokens delivered at the API boundary.",
                   reasoning_tokens_total_.load(std::memory_order_relaxed));
    append_metric(out, "counter", "ninfer:cached_tokens_total",
                   "Prompt tokens served from the prefix cache.",
                   cached_tokens_total_.load(std::memory_order_relaxed));
    append_metric(out, "counter", "ninfer:mtp_rounds_total", "Speculative decode rounds.",
                   mtp_rounds_total_.load(std::memory_order_relaxed));
    append_metric(out, "counter", "ninfer:mtp_drafted_total", "Tokens drafted by the speculative head.",
                   mtp_drafted_total_.load(std::memory_order_relaxed));
    append_metric(out, "counter", "ninfer:mtp_accepted_total",
                   "Drafted tokens accepted by the target model.",
                   mtp_accepted_total_.load(std::memory_order_relaxed));
    append_metric(out, "counter", "ninfer:request_latency_seconds_count",
                   "Requests contributing to request_latency_seconds_sum.", latency_count);
    {
        char line[128];
        std::snprintf(line, sizeof(line), "ninfer:request_latency_seconds_sum %.3f\n",
                      static_cast<double>(latency_millis) / 1000.0);
        out += "# HELP ninfer:request_latency_seconds_sum Total request wall time in seconds.\n";
        out += "# TYPE ninfer:request_latency_seconds_sum counter\n";
        out += line;
        std::snprintf(line, sizeof(line), "ninfer:time_to_first_token_seconds_sum %.3f\n",
                      static_cast<double>(ttft_millis) / 1000.0);
        out += "# HELP ninfer:time_to_first_token_seconds_sum Total time to first token in "
               "seconds.\n";
        out += "# TYPE ninfer:time_to_first_token_seconds_sum counter\n";
        out += line;
        std::snprintf(line, sizeof(line), "ninfer:prefill_seconds_sum %.3f\n",
                      static_cast<double>(prefill_millis) / 1000.0);
        out += "# HELP ninfer:prefill_seconds_sum Total prefill time in seconds.\n";
        out += "# TYPE ninfer:prefill_seconds_sum counter\n";
        out += line;
        std::snprintf(line, sizeof(line), "ninfer:decode_seconds_sum %.3f\n",
                      static_cast<double>(decode_millis) / 1000.0);
        out += "# HELP ninfer:decode_seconds_sum Total decode time in seconds.\n";
        out += "# TYPE ninfer:decode_seconds_sum counter\n";
        out += line;
    }

    append_metric(out, "gauge", "ninfer:running_requests", "Requests currently decoding.",
                 stats.running_requests);
    append_metric(out, "gauge", "ninfer:prefilling_requests", "Requests currently prefilling.",
                 stats.prefilling_requests);
    append_metric(out, "gauge", "ninfer:waiting_requests", "Requests waiting for a lane.",
                 stats.waiting_requests);
    append_metric(out, "gauge", "ninfer:pool_tokens", "Main KV pool capacity in tokens.", pool_tokens);
    append_metric(out, "gauge", "ninfer:pool_free_tokens", "Free Main KV pool tokens.",
                 pool_free_tokens);
    append_metric(out, "gauge", "ninfer:kept_prompts",
                 "Catalogued private continuations plus shared-prefix owners.",
                 stats.kept_prompts);
    append_metric(out, "gauge", "ninfer:context_length", "Configured maximum context length.",
                 memory.max_context);
    append_metric(out, "gauge", "ninfer:device_state_occupied_slots",
                 "Occupied Device StateImage slots.", stats.device_state_occupied_slots);
    append_metric(out, "gauge", "ninfer:host_state_occupied_slots", "Occupied Host StateImage slots.",
                 stats.host_state_occupied_slots);
    append_metric(out, "gauge", "ninfer:host_kv_occupied_bytes", "Occupied Host KV spill bytes.",
                 stats.host_kv_occupied_bytes);
    append_metric(out, "gauge", "ninfer:host_kv_capacity_bytes", "Host KV spill capacity in bytes.",
                 memory.host_kv_capacity_bytes);
    return out;
}

} // namespace ninfer::serve
