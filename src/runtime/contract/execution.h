#pragma once

#include "runtime/contract/request.h"
#include "runtime/contract/timing.h"
#include <compare>
#include <span>

namespace ninfer::runtime {

struct LaneId {
    std::uint32_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(LaneId, LaneId) noexcept  = default;
    [[nodiscard]] friend constexpr auto operator<=>(LaneId, LaneId) noexcept = default;
};

enum class ConsumeStatus : std::uint8_t {
    Consumed,
    InvariantMismatch,
};

enum class CommitDisposition : std::uint8_t {
    Active,
    Finishable,
    CancelledReleased,
};

// The product Engine only needs statistics for rows whose sequence is released by commit.
// Direct diagnostic callers may temporarily request cumulative snapshots for every row.
enum class CommitObservation : std::uint8_t {
    ReleasedRowsOnly,
    AllRows,
};

struct CommitDecision {
    std::uint32_t accepted_tokens = 0;
    bool terminal                 = false;
    bool cancelled                = false;
    // Copied unchanged from the corresponding OutputDecision; still relative to this row's
    // accepted span.
    std::optional<std::uint32_t> prefix_execution_split_after;
};

struct BeginSummary {
    std::uint32_t prompt_tokens        = 0;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::Root;

    [[nodiscard]] friend constexpr bool operator==(BeginSummary, BeginSummary) noexcept = default;
};

// Target-model probability reports for one round's published tokens, laid out exactly like the
// token span they describe: element i belongs to token i, and the alternative ranks for token i
// occupy [i*ranks, (i+1)*ranks). Non-owning views into the Program's pinned egress buffer, valid
// only between that round's decode and commit, like the token span itself. Every span is empty when
// the engine was loaded without EngineOptions::token_logprobs.
struct RoundTokenScores {
    std::span<const float> chosen;
    std::span<const std::int32_t> top_ids;
    std::span<const float> top_logprobs;
    std::int32_t ranks                   = 0;
    [[nodiscard]] bool empty() const noexcept { return chosen.empty(); }
};

struct GeneratedRound {
    std::span<const TokenId> tokens;
    RoundTokenScores scores;
};

struct BatchedGeneratedRound {
    std::span<const TokenId> tokens;
    std::span<const std::int32_t> row_counts;
    std::uint32_t row_stride = 1;
    RoundTokenScores scores;
    ExecutionTiming timing;
};

struct PrefillStepResult {
    BeginSummary summary;
    GeneratedRound round;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    ExecutionTiming timing;
};

struct RoundBudget {
    std::uint32_t generated_tokens_remaining = 0;
};

} // namespace ninfer::runtime
