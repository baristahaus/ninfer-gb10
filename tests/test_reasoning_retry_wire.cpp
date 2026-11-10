// --reasoning-close-retry wiring: the option is off unless given, a request is only eligible when
// the option is on and the prompt starts in thinking with no media, and the request_done record
// reports what the retry did. The planner's own trigger cases are covered by
// tests/test_reasoning_retry.cpp; this file covers the gate in front of it.
#include "product/config/options.h"
#include "serve/generation_service.h"
#include "serve/reasoning_retry.h"
#include "serve/request_log.h"

#include <nlohmann/json.hpp>

#include <iostream>
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

ninfer::PromptSummary summary(bool starts_in_reasoning, bool has_media) {
    return ninfer::PromptSummary{.starts_in_reasoning = starts_in_reasoning,
                                 .prompt_tokens       = 128,
                                 .has_media           = has_media};
}

Json done_json(const GenerationOutcome& outcome) {
    RequestLogContext context;
    context.id       = 1;
    context.protocol = "openai_chat_completions";
    context.model    = "test-model";
    return Json::parse(format_request_done_json("serve-test", 1, context, outcome));
}

} // namespace

int main() {
    int failures = 0;

    // Off unless given, and --no-reasoning-close-retry turns it back off.
    const auto defaults = parse({"ninfer-serve", "model.ninfer"});
    failures += check(!defaults.reasoning_close_retry, "the retry is not off by default");
    const auto enabled = parse({"ninfer-serve", "model.ninfer", "--reasoning-close-retry"});
    failures += check(enabled.reasoning_close_retry, "--reasoning-close-retry did not enable it");
    const auto disabled = parse({"ninfer-serve", "model.ninfer", "--reasoning-close-retry",
                                 "--no-reasoning-close-retry"});
    failures += check(!disabled.reasoning_close_retry,
                      "--no-reasoning-close-retry did not turn it off");

    // Eligibility: the option, a thinking start and no media. This is the gate that keeps the whole
    // path off when the option is off, whatever the request looks like.
    failures += check(reasoning_retry_eligible(true, summary(true, false)),
                      "an eligible request was rejected");
    for (const auto& [enabled_flag, starts, media, label] :
         std::vector<std::tuple<bool, bool, bool, const char*>>{
             {false, true, false, "the option is off"},
             {true, false, false, "the prompt does not start in thinking"},
             {true, true, true, "the prompt carries media"},
             {false, false, true, "everything is off"}}) {
        failures += check(!reasoning_retry_eligible(enabled_flag, summary(starts, media)), label);
    }

    // The planner behind the gate produces the expected trigger for an eligible request, and no
    // trigger for a request that does not start in thinking.
    ReasoningGuardTokens tokens;
    tokens.close_suffix = {198, 248069, 271};
    tokens.think_close  = {248069};
    const std::vector<ninfer::TokenId> stopped{248046};
    const ReasoningRetryInput in_reasoning{.starts_in_reasoning      = true,
                                           .finish_reason            = ninfer::FinishReason::StopToken,
                                           .generated                = stopped,
                                           .content                  = "",
                                           .tool_calls               = 0,
                                           .thinking_control_applied = false,
                                           .requested_output_tokens  = 1000};
    failures += check(plan_reasoning_retry(in_reasoning, tokens).trigger ==
                          ReasoningRetryTrigger::StopInReasoning,
                      "the planner no longer triggers on a first-token stop");
    ReasoningRetryInput not_in_reasoning = in_reasoning;
    not_in_reasoning.starts_in_reasoning = false;
    failures += check(plan_reasoning_retry(not_in_reasoning, tokens).trigger ==
                          ReasoningRetryTrigger::None,
                      "the planner triggered without a thinking start");
    failures += check(!reasoning_retry_eligible(true, summary(false, false)),
                      "a request the planner refuses passed the gate");

    // The canonical constants are the ones the Qwen frontends render.
    failures += check(kReasoningCloseSerialization == "\n</think>\n\n" &&
                          kReasoningThinkClose == "</think>",
                      "the canonical reasoning-close text changed");

    // The record reports nothing when the retry did not fire, and the trigger name when it did.
    GenerationOutcome untouched;
    failures += check(done_json(untouched).at("result").at("reasoning_retry").is_null(),
                      "a request without a retry reported one");
    GenerationOutcome retried;
    retried.reasoning_retry.trigger             = ReasoningRetryTrigger::StopInReasoning;
    retried.reasoning_retry.kept_tokens         = 0;
    retried.reasoning_retry.injected_tokens     = 3;
    retried.reasoning_retry.continuation_prompt = 131;
    retried.reasoning_retry.continuation_completion = 12;
    const auto retried_json = done_json(retried).at("result").at("reasoning_retry");
    failures += check(retried_json.at("trigger") == "stop_in_reasoning" &&
                          retried_json.at("injected_tokens") == 3 &&
                          retried_json.at("continuation_completion_tokens") == 12 &&
                          retried_json.at("error").is_null(),
                      "the retry outcome was not reported");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
