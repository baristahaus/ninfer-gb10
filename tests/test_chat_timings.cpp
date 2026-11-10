// The OpenAI chat `timings` block is the llama.cpp-compatible surface a client reads to judge the
// server. A block that is structurally present and always zero reads as a broken or unusably slow
// server, which is the failure this guards: the Flash-Next route published no wall bound, so every
// field rendered as 0.0 while the JSONL record had the real numbers.
#include "serve/generation_service.h"
#include "serve/openai_chat.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

namespace {

using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << "\n";
    return 1;
}

Json timings_for(const ninfer::RequestStats& stats, int prompt_tokens, int completion_tokens) {
    ninfer::serve::GenerationOutcome outcome;
    outcome.prompt_tokens     = prompt_tokens;
    outcome.completion_tokens = completion_tokens;
    outcome.finish_reason     = ninfer::FinishReason::StopToken;
    outcome.stats             = stats;
    const auto identity       = ninfer::serve::make_openai_chat_response_identity("test-model");
    return Json::parse(ninfer::serve::make_chat_completion_response(identity, outcome))
        .at("timings");
}

} // namespace

int main() {
    int failures = 0;

    // The wall bounds are the advertised quantity when the engine recorded them.
    {
        ninfer::RequestStats stats;
        stats.timings.prompt_wall_seconds     = 0.5;
        stats.timings.generation_wall_seconds = 2.0;
        // Phase measurements differ on purpose: the wall bound must win.
        stats.timings.prefill_seconds = 0.1;
        stats.timings.decode_seconds  = 0.2;
        const Json timings            = timings_for(stats, 100, 100);
        failures += check(timings.at("prompt_ms") == 500.0, "prompt_ms did not use the wall bound");
        failures += check(timings.at("predicted_ms") == 2000.0,
                          "predicted_ms did not use the wall bound");
        failures += check(timings.at("predicted_per_second") == 49.5,
                          "predicted_per_second is wrong for 99 intervals over 2 s");
        failures += check(timings.at("prompt_per_second") == 200.0,
                          "prompt_per_second is wrong for 100 tokens over 0.5 s");
    }

    // The shape the Flash-Next route produced: no wall bound recorded, phase measurements present.
    // Every field must still be populated rather than published as zero.
    {
        ninfer::RequestStats stats;
        stats.timings.prefill_seconds = 0.4;
        stats.timings.decode_seconds  = 3.0;
        const Json timings            = timings_for(stats, 20, 100);
        failures += check(timings.at("prompt_ms") == 400.0,
                          "a missing wall bound published a zero prompt_ms");
        failures += check(timings.at("predicted_ms") == 3000.0,
                          "a missing wall bound published a zero predicted_ms");
        failures += check(timings.at("predicted_per_second") == 33.0,
                          "a missing wall bound published a zero predicted_per_second");
        failures += check(timings.at("prompt_per_second") == 50.0,
                          "a missing wall bound published a zero prompt_per_second");
    }

    // A request with no recorded time at all stays zero rather than inventing a rate.
    {
        const Json timings = timings_for({}, 10, 2);
        failures += check(timings.at("prompt_ms") == 0.0 && timings.at("predicted_ms") == 0.0 &&
                              timings.at("predicted_per_second") == 0.0,
                          "an untimed request invented a rate");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
