// SPDX-FileCopyrightText: ValerioDolci/capy-flash
// SPDX-License-Identifier: Apache-2.0
// plan_reasoning_retry (serve --reasoning-close-retry): when a reply is continued, with which
// tokens, and when it is left alone. Ported from capy-flash tests/test_reasoning_retry.cpp (release
// commit c2d9e4ba); the caller-stop case uses this Engine's FinishReason::StopString instead of
// capy's stop-token-id membership test.
#include "serve/reasoning_retry.h"

#include <iostream>
#include <vector>

namespace {

using ninfer::FinishReason;
using ninfer::TokenId;
using ninfer::serve::plan_reasoning_retry;
using ninfer::serve::ReasoningGuardTokens;
using ninfer::serve::ReasoningRetryInput;
using ninfer::serve::ReasoningRetryTrigger;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << "\n";
    return 1;
}

// 248069 is </think>, 198/271 are the surrounding newlines; the other ids are illustrative.
ReasoningGuardTokens qwen_tokens() {
    ReasoningGuardTokens tokens;
    tokens.close_suffix = {198, 248069, 271};
    tokens.think_close  = {248069};
    return tokens;
}

ReasoningRetryInput stopped(const std::vector<TokenId>& generated) {
    return ReasoningRetryInput{.starts_in_reasoning      = true,
                               .finish_reason            = FinishReason::StopToken,
                               .generated                = generated,
                               .content                  = "",
                               .tool_calls               = 0,
                               .thinking_control_applied = false,
                               .requested_output_tokens  = 32000};
}

} // namespace

int main() {
    int failures      = 0;
    const auto tokens = qwen_tokens();

    // The first token after "<think>\n" is the end-of-turn token.
    {
        const std::vector<TokenId> generated{248046};
        const auto plan = plan_reasoning_retry(stopped(generated), tokens);
        failures += check(plan.trigger == ReasoningRetryTrigger::StopInReasoning &&
                              plan.appended == tokens.close_suffix && plan.kept_tokens == 0 &&
                              plan.injected_tokens == 3 && plan.remaining_output_tokens == 31997U,
                          "a first-token stop is not closed and continued with the close suffix");
    }
    // Stop after some reasoning: the reasoning is kept, then the close.
    {
        const std::vector<TokenId> generated{100, 101, 102, 248044};
        const auto plan = plan_reasoning_retry(stopped(generated), tokens);
        failures +=
            check(plan.trigger == ReasoningRetryTrigger::StopInReasoning &&
                      plan.appended == std::vector<TokenId>({100, 101, 102, 198, 248069, 271}) &&
                      plan.kept_tokens == 3,
                  "a stop inside the reasoning does not keep the reasoning before the close");
    }
    // Stop right after "</think>" with blank content: resumed from the same point, nothing added.
    {
        const std::vector<TokenId> generated{100, 248069, 271, 248046};
        auto input      = stopped(generated);
        input.content   = "\n";
        const auto plan = plan_reasoning_retry(input, tokens);
        failures += check(plan.trigger == ReasoningRetryTrigger::EmptyAfterClose &&
                              plan.appended == std::vector<TokenId>({100, 248069, 271}) &&
                              plan.injected_tokens == 0,
                          "an empty reply after </think> is not resumed from the same point");
    }
    // Left alone: a normal answer, a tool call, a length finish, a caller stop, no thinking, an
    // applied thinking budget, no room after the close, missing guard tokens.
    {
        const std::vector<TokenId> generated{100, 248069, 271, 300, 248046};
        auto answer    = stopped(generated);
        answer.content = "Done.";
        failures +=
            check(plan_reasoning_retry(answer, tokens).trigger == ReasoningRetryTrigger::None,
                  "a reply with content was retried");
        auto tool       = stopped(generated);
        tool.tool_calls = 1;
        failures += check(plan_reasoning_retry(tool, tokens).trigger == ReasoningRetryTrigger::None,
                          "a reply with a tool call was retried");
    }
    {
        const std::vector<TokenId> generated{100, 101};
        auto length          = stopped(generated);
        length.finish_reason = FinishReason::OutputLimit;
        failures +=
            check(plan_reasoning_retry(length, tokens).trigger == ReasoningRetryTrigger::None,
                  "a reply cut by the output limit was retried");
        // Our equivalent of capy's caller-stop-token case: a caller stop string is StopString.
        auto caller_stop          = stopped(generated);
        caller_stop.finish_reason = FinishReason::StopString;
        failures += check(plan_reasoning_retry(caller_stop, tokens).trigger ==
                              ReasoningRetryTrigger::None,
                          "a caller stop string was treated as the model's end of turn");
    }
    {
        const std::vector<TokenId> generated{248046};
        auto plain                = stopped(generated);
        plain.starts_in_reasoning = false;
        failures +=
            check(plan_reasoning_retry(plain, tokens).trigger == ReasoningRetryTrigger::None,
                  "a reply without thinking was retried");
        auto budget                     = stopped(generated);
        budget.thinking_control_applied = true;
        failures +=
            check(plan_reasoning_retry(budget, tokens).trigger == ReasoningRetryTrigger::None,
                  "a reply whose thinking budget was applied was retried");
        auto tight                    = stopped(generated);
        tight.requested_output_tokens = 3;
        failures +=
            check(plan_reasoning_retry(tight, tokens).trigger == ReasoningRetryTrigger::None,
                  "a retry was planned without room for one token after the close");
        auto open                    = stopped(generated);
        open.requested_output_tokens = 0;
        const auto plan              = plan_reasoning_retry(open, tokens);
        failures += check(plan.trigger == ReasoningRetryTrigger::StopInReasoning &&
                              !plan.remaining_output_tokens,
                          "an unbounded request did not leave the continuation budget unset");
        auto empty_suffix = tokens;
        empty_suffix.close_suffix.clear();
        failures += check(plan_reasoning_retry(stopped(generated), empty_suffix).trigger ==
                              ReasoningRetryTrigger::None,
                          "a plan was made without a close suffix");
        auto empty_close = tokens;
        empty_close.think_close.clear();
        failures += check(plan_reasoning_retry(stopped(generated), empty_close).trigger ==
                              ReasoningRetryTrigger::None,
                          "a plan was made without a thinking-close marker");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
