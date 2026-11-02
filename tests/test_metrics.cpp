#include "serve/metrics.h"

#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::serve;

int main() {
    int failures     = 0;
    const auto check = [&](bool pass, const char* message) {
        if (!pass) {
            std::cerr << message << '\n';
            ++failures;
        }
    };
    EngineOptions options;
    options.max_concurrency                 = 2;
    options.context_cache.extra_state_slots = 1;
    options.speculative.draft_tokens        = 2;
    EngineDescription description;
    description.architecture                = "arch";
    description.ranks                       = {{0, 0, 0, "cuda", false}};
    description.device_state_capacity       = 3;
    description.host_context_capacity_bytes = 1048576;
    StartupReport startup;
    EngineSnapshot baseline;
    baseline.runtime.generated_tokens         = 4;
    baseline.runtime.speculative_draft_tokens = 3;
    baseline.ranks.resize(1);
    baseline.ranks[0].gpu_intervals[0] = {1, 1000, 1000};
    baseline.executions[1].units       = 3;
    baseline.executions[1].rows        = 5;
    Metrics metrics;
    metrics.configure("custom\"model\\name\nline", options, description, startup, baseline);
    EngineSnapshot running = baseline;
    running.executions[1].units += 2;
    running.executions[1].rows += 4;
    running.ranks[0].gpu_intervals[0] = {4, 9000, 3000};
    running.runtime.generated_tokens += 5;
    running.runtime.speculative_draft_tokens += 6;
    running.runtime.speculative_accepted_tokens = 3;
    running.runtime.running_requests            = 1;
    running.runtime.host_context_occupied_bytes = 4096;
    running.runtime.host_context_reserved_bytes = 1024;
    running.runtime.kv_pools = {{0, "text", 2, 8, 64}, {1, "draft", 1, 4, 64}, {2, "index", 3, 6, 64}};
    running.runtime.context_transfers.kv.resize(3);
    running.runtime.context_transfers.kv[2][1].bytes = 8192;
    metrics.first_token({.prepare_seconds = 0.05, .elapsed_since_submit_seconds = 0.2});
    const auto live = metrics.render(running, true);
    check(live.find("ninfer_execution_units_total{kind=\"decode\"} 2\n") != std::string::npos &&
              live.find("ninfer_execution_rows_total{kind=\"decode\"} 4\n") != std::string::npos &&
              live.find("ninfer_rank_gpu_interval_samples_total{rank=\"0\",node=\"0\",interval="
                        "\"result_stream\"} 3\n") != std::string::npos &&
              live.find("ninfer_rank_gpu_interval_samples_total{rank=\"0\",node=\"0\",interval="
                        "\"commit_stream\"} 0\n") != std::string::npos,
          "metrics must distinguish batch/row deltas and measured/unmeasured GPU intervals");
    check(live.find("ninfer_generation_tokens_total 5\n") != std::string::npos,
          "live generated count must exclude startup warmup");
    check(live.find("ninfer_spec_decode_draft_tokens_total 6\n") != std::string::npos,
          "speculative work must be observable before request completion");
    check(live.find("ninfer_time_to_first_token_seconds_count 1\n") != std::string::npos &&
              live.find("ninfer_requests_total{outcome=\"completed\"} 0\n") != std::string::npos,
          "TTFT must be visible before completion");
    check(live.find("_bucket{le=\"0.25\"} 1\n") != std::string::npos &&
              live.find("ninfer_time_to_first_token_seconds_sum 0.25\n") != std::string::npos,
          "TTFT must include preparation and use inclusive histogram buckets");
    check(live.find("model_name=\"custom\\\"model\\\\name\\nline\"") != std::string::npos,
          "model label must escape quotes, backslashes and newlines");
    // Process memory and CPU seconds are sampled at scrape time and legitimately move between
    // renders; the idempotency invariant covers every other line.
    const auto stable = [](std::string text) {
        std::string out;
        std::size_t start = 0;
        while (start < text.size()) {
            const auto end = text.find('\n', start);
            const auto line = text.substr(start, end == std::string::npos ? end : end - start);
            if (line.rfind("ninfer_process_", 0) != 0) { out += line; out += '\n'; }
            start = end == std::string::npos ? text.size() : end + 1;
        }
        return out;
    };
    check(stable(metrics.render(running, true)) == stable(live),
          "scrapes must not consume or reset counters");
    check(live.find("ninfer_host_context_used_bytes 4096\n") != std::string::npos,
          "reserved bytes must not be added to occupancy twice");

    check(live.find("ninfer_device_kv_used_page_groups{pool_id=\"2\"} 3\n") != std::string::npos &&
              live.find("ninfer_context_transfer_bytes_total{resource=\"kv\",pool_id=\"2\","
                        "direction=\"h2d\"} 8192\n") != std::string::npos,
          "pool metrics dropped the third pool or used a model role as its identity");

    GenerationOutcome outcome;
    outcome.finish_reason = FinishReason::OutputLimit;
    outcome.stats.timings.total_seconds = 2.0;
    outcome.stats.timings.prefill_seconds = 0.0625;
    outcome.stats.timings.decode_seconds = 0.5;
    outcome.stats.timings.generation_wall_seconds = 1.0;
    outcome.stats.engine_timing.queue_wait_seconds = 0.1;
    outcome.generated_token_ids.resize(3);
    outcome.stats.decode_by_batch.resize(2);
    outcome.stats.decode_by_batch[1].total.count = 2;
    outcome.stats.speculative.enabled = true;
    outcome.stats.speculative.draft_window = 2;
    outcome.stats.speculative.drafted_per_position = {5, 3};
    outcome.stats.speculative.accepted_per_position = {4, 1};
    for (int probe = 0; probe < 3; ++probe) { metrics.http_request(); }
    std::vector<std::thread> writers;
    for (int i = 0; i < 4; ++i) {
        writers.emplace_back([&] {
            for (int n = 0; n < 100; ++n) { metrics.done(outcome); }
        });
    }
    for (auto& writer : writers) { writer.join(); }
    outcome.finish_reason = FinishReason::Cancelled;
    metrics.done(outcome);
    metrics.failed(true);
    metrics.failed(false);
    metrics.rejected();
    const auto final = metrics.render(running, false);
    check(final.find("ninfer_engine_ready 0\n") != std::string::npos,
          "unavailable engine must remain observable");
    check(final.find("ninfer_requests_total{outcome=\"completed\"} 400\n") != std::string::npos &&
              final.find("ninfer_requests_total{outcome=\"cancelled\"} 2\n") != std::string::npos &&
              final.find("ninfer_requests_total{outcome=\"failed\"} 1\n") != std::string::npos &&
              final.find("ninfer_requests_total{outcome=\"rejected\"} 1\n") != std::string::npos,
          "concurrent settlements and terminal classifications must be preserved");
    check(final.find("ninfer_request_duration_seconds_bucket{le=\"+Inf\"} 401\n") !=
                  std::string::npos &&
              final.find("ninfer_request_duration_seconds_count 401\n") != std::string::npos &&
              final.find("ninfer_request_duration_seconds_sum 802\n") != std::string::npos,
          "histogram counts and sums must include cancelled outcomes without fabricating failure "
          "durations");
    check(final.find("ninfer_time_to_first_token_seconds_count 1\n") != std::string::npos,
          "request settlement must not count the first token again");
    // Every one of the 401 settled outcomes (400 completed plus the one cancelled) carried the
    // enriched fields.
    check(final.find("ninfer_request_prefill_time_seconds_count 401\n") != std::string::npos &&
              final.find("ninfer_request_prefill_time_seconds_sum 25.0625\n") != std::string::npos &&
              final.find("ninfer_request_prefill_time_seconds_bucket{le=\"0.25\"} 401\n") !=
                  std::string::npos,
          "prefill execution exposure must render as its own histogram");
    check(final.find("ninfer_request_decode_time_seconds_count 401\n") != std::string::npos &&
              final.find("ninfer_request_decode_time_seconds_sum 200.5\n") != std::string::npos,
          "decode execution exposure must render as its own histogram");
    check(final.find("ninfer_inter_token_latency_seconds_count 401\n") != std::string::npos &&
              final.find("ninfer_inter_token_latency_seconds_sum 200.5\n") != std::string::npos &&
              final.find("ninfer_inter_token_latency_seconds_bucket{le=\"0.5\"} 401\n") !=
                  std::string::npos,
          "mean inter-token latency must be one observation per request with at least two tokens");
    check(final.find("ninfer_decode_batch_tokens_bucket{le=\"1\"} 0\n") != std::string::npos &&
              final.find("ninfer_decode_batch_tokens_bucket{le=\"2\"} 401\n") != std::string::npos &&
              final.find("ninfer_decode_batch_tokens_sum 802\n") != std::string::npos &&
              final.find("ninfer_decode_batch_tokens_count 401\n") != std::string::npos,
          "decode steps must be attributed to their batch size by participation divided by size");
    check(final.find("ninfer_request_finish_reasons_total{reason=\"output_limit\"} 400\n") !=
                  std::string::npos &&
              final.find("ninfer_request_finish_reasons_total{reason=\"cancelled\"} 2\n") !=
                  std::string::npos,
          "semantic finish reasons must split cancelled from the terminal completions");
    check(final.find("ninfer_spec_decode_draft_tokens_per_pos_total{position=\"1\"} 2005\n") !=
                  std::string::npos &&
              final.find("ninfer_spec_decode_draft_tokens_per_pos_total{position=\"2\"} 1203\n") !=
                  std::string::npos &&
              final.find("ninfer_spec_decode_accepted_tokens_per_pos_total{position=\"1\"} 1604\n") !=
                  std::string::npos &&
              final.find("ninfer_spec_decode_accepted_tokens_per_pos_total{position=\"2\"} 401\n") !=
                  std::string::npos,
          "draft and acceptance totals must aggregate per draft position");
    check(final.find("ninfer_kv_cache_usage_ratio{pool=\"text\"} 0.25\n") != std::string::npos &&
              final.find("ninfer_kv_pool_occupied_tokens{pool=\"text\"} 128\n") != std::string::npos &&
              final.find("ninfer_kv_pool_capacity_tokens{pool=\"text\"} 512\n") != std::string::npos,
          "KV pools must expose usage ratio and page-group-scaled token gauges");
    check(final.find("ninfer_http_requests_total 3\n") != std::string::npos,
          "the pre-routing hook counter must count observed HTTP requests");
    check(final.find("ninfer_engine_info{architecture=\"arch\",kv_cache=\"bf16\","
                      "kv_capacity_tokens=\"0\",max_context=\"0\",max_concurrency=\"0\","
                      "prefill_chunk=\"0\",ranks=\"1\",hardware=\"cuda\"} 1\n") != std::string::npos,
          "engine_info must carry the full model card with quoted label values");
    check(final.find("ninfer_process_resident_memory_bytes ") != std::string::npos &&
              final.find("ninfer_process_cpu_seconds_total ") != std::string::npos,
          "process memory and CPU counters must render at scrape time");
    check(final.find("ninfer_admission_blocked_total{reason=\"context_transaction\"} 0\n") !=
                  std::string::npos &&
              final.find("ninfer_state_transfer_pages_total{direction=\"d2h\"} 0\n") !=
                  std::string::npos &&
              final.find("ninfer_pressure_spill_pages_total 0\n") != std::string::npos,
          "Flash-Next admission, tier traffic and pressure surfaces must render zeros");
    outcome.stats.constraint = ConstraintObservation{.complete       = true,
                                                     .terminated     = false,
                                                     .cache          = ConstraintCacheAccess::Built,
                                                     .mask_positions = 5,
                                                     .mask_upload_bytes = 128};
    metrics.done(outcome);
    const auto constrained = metrics.render(running, false);
    check(constrained.find(
              "ninfer_constraint_requests_total{outcome=\"complete_interrupted\"} 1\n") !=
                  std::string::npos &&
              constrained.find("ninfer_constraint_mask_positions_total 5\n") != std::string::npos &&
              constrained.find("ninfer_constraint_mask_upload_bytes_total 128\n") !=
                  std::string::npos,
          "interrupted complete constraint lost its state or actual work");
    return failures == 0 ? 0 : 1;
}
