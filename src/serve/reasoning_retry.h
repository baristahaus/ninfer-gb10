// SPDX-FileCopyrightText: ValerioDolci/capy-flash
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Reasoning-close retry planning. Ported from capy-flash `src/serve/reasoning_retry.{h,cpp}`
// (release commit c2d9e4ba "capy-flash v0.7.7: two engine-wide failures after a reasoning-close
// retry", capy HEAD 3d4b4b76). The planner is unchanged in behaviour; the surrounding types are
// adapted to this tree, see below.
//
// The failure this plans around is a model behaviour, not a path property: Qwen3.8-Flash-Next can
// draw its end-of-turn token while the thinking phase is still open - as the very first token after
// "<think>\n", or after some reasoning. The reply then has blank content and no tool call, and an
// agent harness stops. The remedy is to close the thinking with the canonical "\n</think>\n\n" and
// let the model continue once from the reply so far. A reply that stops right after "</think>" with
// blank content is resumed once from the same point with a different seed. A reply that does not end
// this way is untouched.
//
// Adaptations against capy:
//  - `ReasoningGuardTokens` lives here instead of in `include/ninfer/types.h`: our public Engine has
//    no `reasoning_guard_tokens()` accessor yet (capy's does), so the token ids are supplied by the
//    caller. The struct is engine-owned data in capy; this is the seam where that accessor belongs.
//  - The membership test of the last generated token in `stop_token_ids` is dropped. The planner
//    already requires `FinishReason::StopToken`, which this Engine sets only for the model's own
//    terminal token (a caller stop string reports `FinishReason::StopString`), so the extra test
//    would re-derive a fact the finish reason already carries.
//  - `thinking_control_applied` is kept: our `RequestStats::thinking.applied`
//    (`include/ninfer/observability/execution.h`) is the same fact capy reads.

#include "ninfer/types.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// The canonical close the Qwen frontends render, and the thinking-closing marker the model emits.
// Keep in step with kCanonicalReasoningCloseSerialization in
// src/models/common/qwen/frontend/chat_template.h (model-side, not visible to serve).
inline constexpr std::string_view kReasoningCloseSerialization = "\n</think>\n\n";
inline constexpr std::string_view kReasoningThinkClose        = "</think>";

enum class ReasoningRetryTrigger : std::uint8_t {
    None,
    StopInReasoning, // terminal token while thinking was open (no "</think>" in the reply)
    EmptyAfterClose, // terminal token after "</think>" with blank content and no tool call
};

[[nodiscard]] const char* reasoning_retry_trigger_name(ReasoningRetryTrigger trigger) noexcept;

// Engine-owned vocabulary facts the plan needs. In capy these come from
// `Engine::reasoning_guard_tokens()`; here the caller supplies them.
struct ReasoningGuardTokens {
    // Token ids of the canonical close suffix, "\n</think>\n\n".
    std::vector<TokenId> close_suffix;
    // Token ids of the thinking-closing marker, "</think>".
    std::vector<TokenId> think_close;
};

struct ReasoningRetryInput {
    bool starts_in_reasoning   = false;
    FinishReason finish_reason = FinishReason::None;
    std::span<const TokenId> generated; // the reply's token ids, terminal token included
    std::string_view content;
    std::size_t tool_calls        = 0;
    bool thinking_control_applied = false;
    // Request output budget (0 = the server default, not bounded here).
    std::uint32_t requested_output_tokens = 0;
};

struct ReasoningRetryPlan {
    ReasoningRetryTrigger trigger = ReasoningRetryTrigger::None;
    // Tokens appended to the chat prompt: the reply without its terminal token, then the close
    // suffix for StopInReasoning.
    std::vector<TokenId> appended;
    std::uint32_t kept_tokens     = 0; // reply tokens kept (terminal token dropped)
    std::uint32_t injected_tokens = 0; // close-suffix tokens
    // Output budget left for the continuation when the request set one.
    std::optional<std::uint32_t> remaining_output_tokens;
};

[[nodiscard]] ReasoningRetryPlan plan_reasoning_retry(const ReasoningRetryInput& input,
                                                      const ReasoningGuardTokens& tokens);

// True when the service should keep this request's continuation material: the option is on, the
// prompt starts in thinking, and it carries no media. A request that fails this is never retried,
// so the whole path stays off when the option is off.
[[nodiscard]] bool reasoning_retry_eligible(bool enabled,
                                            const ninfer::PromptSummary& summary) noexcept;

} // namespace ninfer::serve
