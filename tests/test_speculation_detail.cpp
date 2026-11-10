// --log-speculation-detail: the option gating and the request_done emission. The per-step recording
// itself happens in the runtime commit loop, which needs a loaded Engine; this covers the contract
// that loop feeds: an empty sequence emits null, a complete one renders one character per step, a
// capped one is marked truncated, and the flag cannot be enabled without the JSONL log.
#include "product/config/options.h"
#include "serve/request_log.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer::serve;
using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << "\n";
    return 1;
}

ninfer::product::ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    return ninfer::product::parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

Json done_json(const ninfer::RequestStats& stats) {
    GenerationOutcome outcome;
    outcome.prompt_tokens   = 10;
    outcome.completion_tokens = 5;
    outcome.finish_reason   = ninfer::FinishReason::StopToken;
    outcome.stats           = stats;
    RequestLogContext context;
    context.id       = 1;
    context.protocol = "openai_chat_completions";
    context.model    = "test-model";
    return Json::parse(format_request_done_json("serve-test", 1, context, outcome));
}

Json detail_of(const ninfer::RequestStats& stats) {
    return done_json(stats).at("speculative").at("detail");
}

ninfer::RequestStats recorded() {
    ninfer::RequestStats stats;
    stats.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    stats.speculative.draft_window  = 3;
    stats.speculative.rounds        = 3;
    stats.speculative.fallback_steps = 1;
    stats.speculative.accepted_lengths = {2, 3, ninfer::kSpeculativeFallbackStep, 0};
    return stats;
}

} // namespace

int main() {
    int failures = 0;

    // The flag is off by default.
    const auto off = parse({"ninfer-serve", "model.ninfer"});
    failures += check(!off.log_speculation_detail && off.speculation_detail_max_steps == 4096,
                      "speculation detail is not off by default with a 4096-step bound");

    // It needs the JSONL request log, because that is where the detail lands.
    bool rejected_without_log = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--log-speculation-detail"});
    } catch (const std::invalid_argument&) { rejected_without_log = true; }
    failures += check(rejected_without_log,
                      "--log-speculation-detail was accepted without --request-log-jsonl");

    const auto enabled = parse({"ninfer-serve", "model.ninfer", "--log-speculation-detail",
                                "--request-log-jsonl", "requests.jsonl"});
    failures += check(enabled.log_speculation_detail && enabled.speculation_detail_max_steps == 4096,
                      "the enabled option did not keep its default bound");
    const auto capped = parse({"ninfer-serve", "model.ninfer", "--log-speculation-detail",
                               "--request-log-jsonl", "requests.jsonl",
                               "--log-speculation-detail-max-steps", "7"});
    failures += check(capped.speculation_detail_max_steps == 7, "the step bound was not applied");

    // A request that recorded nothing emits null: this is the budget-0 case, and it is what every
    // request looks like with the option off.
    ninfer::RequestStats none;
    none.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
    none.speculative.draft_window = 3;
    failures += check(detail_of(none).is_null(), "an unrecorded request emitted a detail object");

    // A complete sequence: one character per step, '-' for a step that verified no draft.
    const auto detail = detail_of(recorded());
    failures += check(detail.at("accepted_lengths") == "23-0",
                      "the recorded sequence did not render one character per step");
    failures += check(detail.at("draft_window") == 3, "the detail lost the draft window");
    failures += check(detail.at("accepted_lengths_truncated") == false,
                      "a complete sequence was marked truncated");

    // A sequence shorter than the request's decode steps was cut by the bound.
    auto truncated                  = recorded();
    truncated.speculative.rounds    = 300;
    truncated.speculative.fallback_steps = 2;
    failures += check(detail_of(truncated).at("accepted_lengths_truncated") == true,
                      "a capped sequence was not marked truncated");

    // A length beyond the draft window is not a value any backend produces; the detail is dropped
    // rather than reported as a wrong digit.
    auto impossible                     = recorded();
    impossible.speculative.accepted_lengths = {4};
    failures += check(detail_of(impossible).is_null(),
                      "a length beyond the draft window was reported");

    // The rest of the speculative block is unchanged by the detail.
    const auto full = done_json(recorded()).at("speculative");
    failures += check(full.at("backend") == "mtp" && full.at("rounds") == 3 &&
                          full.at("fallback_steps") == 1,
                      "the speculative counters changed");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
