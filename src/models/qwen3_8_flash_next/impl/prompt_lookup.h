#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next {

// Prompt lookup for MTP rounds: when a sequence's last tokens occurred earlier in its prompt or
// output, the tokens that followed that occurrence are offered as the round's drafts instead of
// the MTP layer's. Operations work quotes its context (paths, identifiers, log lines, code being
// edited), so a long match often continues verbatim.
//
// The verify window has the same width either way, so a round costs the same with either source
// and the choice is the one with more expected accepted drafts:
//
//   MTP     E = the mean accepted MTP drafts per round (EMA over this sequence's rounds)
//   lookup  E(k) = q + q^2 + ... + q^k for a k-token proposal, q = the acceptance rate of lookup
//           drafts whose match was about as long (four match-length buckets, decayed counts,
//           priors worth four drafts)
//
// Lookup is taken only when its E exceeds the MTP's by a margin. Verification decides every
// token, and drafts are one-hot under both sources, so greedy output is unchanged and sampled
// output keeps its distribution.
class PromptLookup {
public:
    static constexpr std::uint32_t kKeyTokens  = 3;  // a match covers at least the last 3 tokens
    static constexpr std::uint32_t kMaxMatch   = 32; // longer matches share the top bucket
    static constexpr std::uint32_t kMaxDrafts  = 8;
    static constexpr std::size_t kBuckets      = 4;
    static constexpr double kMargin            = 0.05;

    struct Proposal {
        std::array<std::int32_t, kMaxDrafts> tokens{};
        std::uint32_t count = 0;
        std::uint32_t match = 0; // matched suffix length, kKeyTokens..kMaxMatch
    };

    // Forgets the history and the learned rates; a lane starts a new sequence.
    void reset();

    // Indexes the history tokens not yet indexed. `history` is the sequence's whole ledger, which
    // only grows while the sequence lives.
    void sync(std::span<const std::int32_t> history);

    // The tokens that followed the most recent earlier occurrence of the longest indexed suffix
    // match, at most `max_tokens`. Count zero means no match of at least kKeyTokens.
    [[nodiscard]] Proposal propose(std::span<const std::int32_t> history,
                                   std::uint32_t max_tokens) const;

    // Whether `proposal` (count >= 1) is expected to commit more drafts than the MTP's
    // `mtp_extent` drafts would.
    [[nodiscard]] bool prefer(const Proposal& proposal, std::uint32_t mtp_extent) const;

    void observe_lookup(std::uint32_t match, std::uint32_t drafted, std::uint32_t accepted);
    void observe_mtp(std::uint32_t drafted, std::uint32_t accepted);

    [[nodiscard]] double lookup_rate(std::uint32_t match) const;
    [[nodiscard]] double mtp_expected(std::uint32_t extent) const;

private:
    static constexpr std::uint32_t kWays = 4;
    struct Slot {
        std::uint64_t key = 0; // zero marks an empty slot
        std::array<std::uint32_t, kWays> ends{}; // positions of a key's last token, newest first
        std::uint32_t count = 0;
    };

    [[nodiscard]] static std::uint64_t key_at(std::span<const std::int32_t> history,
                                              std::size_t end);
    [[nodiscard]] static std::size_t bucket(std::uint32_t match);
    [[nodiscard]] const Slot* find(std::uint64_t key) const;
    Slot& insert(std::uint64_t key);
    void grow();

    std::vector<Slot> table_;
    std::size_t used_    = 0;
    std::size_t indexed_ = 0; // history tokens indexed

    std::array<double, kBuckets> lookup_accepted_{};
    std::array<double, kBuckets> lookup_rejected_{};
    double mtp_accepted_ = 0.0; // EMA of accepted MTP drafts per round
    double mtp_drafted_  = 0.0; // EMA of MTP drafts per round
    bool mtp_observed_   = false;
};

} // namespace ninfer::models::qwen3_8_flash_next
