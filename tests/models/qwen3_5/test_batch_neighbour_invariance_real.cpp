// Batch-neighbour invariance: a request's greedy text must not depend on who shares its decode
// step.
//
// WHY THIS EXISTS. The engine forms one compact membership across lanes at every round boundary
// (src/runtime/engine/scheduler.h, build_decode_membership). A compacted step shares its kernel
// launches, so if any kernel chooses a regime, a schedule or a split from a step-wide property, a
// request's output can depend on its neighbours. veloGB10 shipped exactly that defect: a decode
// step chose dense or sparse attention once for all rows from the longest request, so a short
// request that shared a step with a long one was computed by different kernels and its greedy text
// changed in three of three runs. Their fix was host-side: choose the regime per row from the row's
// own position, order short rows first, and do not let a speculative round straddle the switch.
//
// WHAT THIS ASSERTS. A prompt's greedy token ids, produced alone, must be identical to the same
// prompt's ids produced while a second, differently sized request is decoding beside it. The test
// also asserts that the two requests really did share decode steps, by reading the engine's
// decode_rounds and decode_row_rounds counters: without that check the comparison would pass
// vacuously on an engine that serialised the two requests.
//
// The 27B v3 artifact is needed, so the test skips without NINFER_TEST_ARTIFACT like the other
// _real tests.
//
// CURRENT STATE ON HARDWARE: this test FAILS, and that is the finding rather than a mistake. On the
// RTX 5090 with the 27B v3 artifact the short request's greedy tokens diverge from its alone run at
// index 7 of 192, deterministically across runs, and it fails identically with CUDA graphs
// disabled, so the cause is the compute path and not a graph key. The suspect is route selection by
// the step's token count: a solo decode step presents one row and takes the GEMV route, while a
// step with two rows takes a sliced-K or MMA route, so the same row is computed by a different
// kernel and a different reduction tree depending on who else is decoding. Filed as
// ninfer-gb10-<defect>; this test is its repro and the gate for the fix.

#include "ninfer/engine.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

struct Options {
    unsigned long_outputs  = 192;
    unsigned short_outputs = 16;
};

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    // The same context and capacity the speculative-invariant test uses, which the engine accepts
    // for two lanes; a larger explicit capacity is refused as outside the usable page range.
    options.max_context     = 2304;
    options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(2304 * 2);
    options.prefill_chunk   = 2304;
    options.max_concurrency = 2;
    // The product default, so the test covers the configuration that actually serves.
    options.use_cuda_graph = true;
    return options;
}

ninfer::RequestOptions request_options(unsigned outputs) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = outputs;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.sampling.seed           = 0;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;
    return request;
}

// A prompt long enough to give the second request a clearly different context length while the
// first is still decoding. The words are irrelevant; the length is the point.
std::string long_prompt() {
    std::string text;
    for (int index = 0; index < 25; ++index) {
        text += "The harbour records list the tide, the wind and the number of vessels cleared on ";
        text += "each day of the season, and the clerk copied every entry without correction. ";
    }
    return text;
}

struct Run {
    std::vector<ninfer::TokenId> tokens;
    std::uint64_t decode_rounds     = 0;
    std::uint64_t decode_row_rounds = 0;
};

Run run_alone(const char* artifact, const std::string& prompt, unsigned outputs) {
    ninfer::EngineOptions options = engine_options(artifact);
    options.max_concurrency       = 1;
    // The capacity must match the lane count: startup allows at most max_concurrency pages of the
    // logical context, so a two-lane capacity is refused by a one-lane engine.
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(options.max_context);
    ninfer::Engine engine(options);
    const auto result = engine.generate(engine.prepare_tokens(engine.tokenize_text(prompt)),
                                        request_options(outputs));
    require(result.generated_token_ids.size() == outputs, "the alone run did not honor the budget");
    const auto stats = engine.runtime_stats();
    return {result.generated_token_ids, stats.decode_rounds, stats.decode_row_rounds};
}

// Submits both prompts before waiting on either, so the engine can admit them together and form a
// compact step. Returns the long prompt's run and the short prompt's run in that order.
std::pair<Run, Run> run_paired(const char* artifact, const std::string& short_prompt,
                               const std::string& long_text, const Options& shape) {
    ninfer::Engine engine(engine_options(artifact));
    {
        const auto& description = engine.description();
        std::cerr << "resolved: max_concurrency=" << description.max_concurrency
                  << " device_state_capacity=" << description.device_state_capacity
                  << " kv_capacity=" << description.kv_capacity
                  << " host_context_capacity_bytes=" << description.host_context_capacity_bytes
                  << "\n";
    }
    std::vector<ninfer::GenerationHandle> handles;
    handles.push_back(engine.submit(engine.prepare_tokens(engine.tokenize_text(short_prompt)),
                                    request_options(shape.long_outputs)));
    handles.push_back(engine.submit(engine.prepare_tokens(engine.tokenize_text(long_text)),
                                    request_options(shape.short_outputs)));
    Run first;
    Run second;
    first.tokens            = handles[0].wait().generated_token_ids;
    second.tokens           = handles[1].wait().generated_token_ids;
    const auto stats        = engine.runtime_stats();
    first.decode_rounds     = stats.decode_rounds;
    first.decode_row_rounds = stats.decode_row_rounds;
    require(first.tokens.size() == shape.long_outputs,
            "the paired long run did not honor the budget");
    require(second.tokens.size() == shape.short_outputs,
            "the paired short run did not honor the budget");
    return {first, second};
}

void require_identical(const std::vector<ninfer::TokenId>& expected,
                       const std::vector<ninfer::TokenId>& actual, const char* label) {
    if (expected == actual) { return; }
    const std::size_t common     = std::min(expected.size(), actual.size());
    std::size_t first_difference = common;
    for (std::size_t index = 0; index < common; ++index) {
        if (expected[index] != actual[index]) {
            first_difference = index;
            break;
        }
    }
    throw std::runtime_error(
        std::string(label) + ": greedy tokens diverged from the alone run at " + "index " +
        std::to_string(first_difference) + " (alone " + std::to_string(expected.size()) +
        " tokens, paired " + std::to_string(actual.size()) + ")");
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cerr << "SKIP: NINFER_TEST_ARTIFACT is not set; the batch-neighbour test did not run. "
                     "Set it to a .ninfer file\n";
        return 77;
    }
    try {
        const Options shape;
        const std::string short_prompt =
            "Complete this sentence with exactly one short factual clause: "
            "The primary gas in Earth's atmosphere is";
        const std::string long_text = long_prompt();

        const auto short_alone = run_alone(artifact, short_prompt, shape.long_outputs);
        const auto long_alone  = run_alone(artifact, long_text, shape.short_outputs);

        const auto [short_paired, long_paired] =
            run_paired(artifact, short_prompt, long_text, shape);

        // The point of the test: the neighbour did not change either request's bytes.
        require_identical(short_alone.tokens, short_paired.tokens,
                          "short request with a neighbour");
        require_identical(long_alone.tokens, long_paired.tokens, "long request with a neighbour");

        // And the two requests really did share decode steps, so the comparison above was not
        // vacuous. decode_row_rounds counts the sum of the batch sizes of the decode rounds.
        if (long_paired.decode_row_rounds <= long_paired.decode_rounds) {
            std::cerr << "paired decode_rounds=" << long_paired.decode_rounds
                      << " decode_row_rounds=" << long_paired.decode_row_rounds << "\n";
            throw std::runtime_error(
                "the two requests never shared a decode step; the invariance claim is untested");
        }

        std::cout << "ok: greedy text is independent of the decode-step neighbour ("
                  << long_paired.decode_rounds << " decode rounds, "
                  << long_paired.decode_row_rounds << " row rounds)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fail: " << error.what() << "\n";
        return 1;
    }
}
