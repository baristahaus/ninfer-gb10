#include "serve/reasoning_retry.h"

#include <algorithm>
#include <cctype>

namespace ninfer::serve {
namespace {

bool blank(std::string_view text) noexcept {
    return std::all_of(text.begin(), text.end(),
                       [](char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; });
}

bool contains(std::span<const TokenId> haystack, std::span<const TokenId> needle) {
    if (needle.empty() || needle.size() > haystack.size()) { return false; }
    return std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end()) !=
           haystack.end();
}

} // namespace

const char* reasoning_retry_trigger_name(ReasoningRetryTrigger trigger) noexcept {
    switch (trigger) {
    case ReasoningRetryTrigger::None: return "none";
    case ReasoningRetryTrigger::StopInReasoning: return "stop_in_reasoning";
    case ReasoningRetryTrigger::EmptyAfterClose: return "empty_after_close";
    }
    return "none";
}

bool reasoning_retry_eligible(bool enabled, const ninfer::PromptSummary& summary) noexcept {
    return enabled && summary.starts_in_reasoning && !summary.has_media;
}

ReasoningRetryPlan plan_reasoning_retry(const ReasoningRetryInput& input,
                                        const ReasoningGuardTokens& tokens) {
    ReasoningRetryPlan plan;
    if (!input.starts_in_reasoning || input.finish_reason != FinishReason::StopToken ||
        input.tool_calls != 0 || !blank(input.content) || input.thinking_control_applied ||
        input.generated.empty() || tokens.close_suffix.empty() || tokens.think_close.empty()) {
        return plan;
    }
    // The terminal token is not part of the continuation: the reply is replayed without it, so the
    // model re-decides that position instead of seeing its own end-of-turn.
    const std::span<const TokenId> kept = input.generated.first(input.generated.size() - 1U);
    const bool closed                   = contains(kept, tokens.think_close);
    const std::size_t injected          = closed ? 0U : tokens.close_suffix.size();
    if (input.requested_output_tokens != 0) {
        const std::uint64_t used = kept.size() + injected;
        // At least one model token must fit after the close.
        if (used + 1U > input.requested_output_tokens) { return plan; }
        plan.remaining_output_tokens =
            static_cast<std::uint32_t>(input.requested_output_tokens - used);
    }
    plan.trigger = closed ? ReasoningRetryTrigger::EmptyAfterClose
                          : ReasoningRetryTrigger::StopInReasoning;
    plan.appended.assign(kept.begin(), kept.end());
    if (!closed) {
        plan.appended.insert(plan.appended.end(), tokens.close_suffix.begin(),
                             tokens.close_suffix.end());
    }
    plan.kept_tokens     = static_cast<std::uint32_t>(kept.size());
    plan.injected_tokens = static_cast<std::uint32_t>(injected);
    return plan;
}

} // namespace ninfer::serve
