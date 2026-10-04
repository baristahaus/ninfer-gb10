#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next {

// Prompt lookup for MTP rounds: when a sequence's last tokens occurred earlier in its prompt or
// output, the tokens that followed that occurrence are offered as the round's drafts instead of
// the MTP layer's. Editing and reviewing pasted scripts and quoting logs repeat the context
// verbatim, so a long match often continues.
//
// PromptLookupIndex is per request: it indexes that request's ledger and proposes drafts.
// PromptLookupPolicy is per Program: acceptance rates describe the workload, and a request drafts
// from lookup too rarely to learn them alone (measured: about two lookup rounds per request).
//
// The verify window has the same width either way, so a round costs the same with either source
// and the choice is the one with more expected accepted drafts:
//
//   MTP     E = the mean accepted MTP drafts per round (EMA over the Program's rounds)
//   lookup  E(k) = q + q^2 + ... + q^k for a k-token proposal, q = the acceptance rate of lookup
//           drafts whose match was about as long (four match-length buckets, decayed counts,
//           priors worth four drafts)
//
// Lookup is taken only when its E exceeds the MTP's by a margin. A bucket's evidence also fades
// while its proposals are passed over, so a workload that starts copying is tried again.
// Verification decides every token, and drafts are one-hot under both sources, so greedy output
// is unchanged and sampled output keeps its distribution.
inline constexpr std::uint32_t kPromptLookupMaxDrafts = 8;

struct PromptLookupProposal {
    std::array<std::int32_t, kPromptLookupMaxDrafts> tokens{};
    std::uint32_t count = 0;
    std::uint32_t match = 0; // matched suffix length, PromptLookupIndex::kKeyTokens..kMaxMatch
};

class PromptLookupIndex {
public:
    static constexpr std::uint32_t kKeyTokens = 3;  // a match covers at least the last 3 tokens
    static constexpr std::uint32_t kMaxMatch  = 32; // longer matches share the top bucket

    // Forgets the history; a lane starts a new request.
    void reset();

    // Indexes the history tokens not yet indexed. `history` is the request's whole ledger, which
    // only grows while the request lives.
    void sync(std::span<const std::int32_t> history);

    // The tokens that followed the most recent earlier occurrence of the longest indexed suffix
    // match, at most `max_tokens`. Count zero means no match of at least kKeyTokens.
    [[nodiscard]] PromptLookupProposal propose(std::span<const std::int32_t> history,
                                               std::uint32_t max_tokens) const;

private:
    static constexpr std::uint32_t kWays = 4;
    struct Slot {
        std::uint64_t key = 0;                   // zero marks an empty slot
        std::array<std::uint32_t, kWays> ends{}; // positions of a key's last token, newest first
        std::uint32_t count = 0;
    };

    [[nodiscard]] static std::uint64_t key_at(std::span<const std::int32_t> history,
                                              std::size_t end);
    [[nodiscard]] const Slot* find(std::uint64_t key) const;
    Slot& insert(std::uint64_t key);
    void grow();

    std::vector<Slot> table_;
    std::size_t used_    = 0;
    std::size_t indexed_ = 0; // history tokens indexed
};

class PromptLookupPolicy {
public:
    static constexpr std::size_t kBuckets = 4;
    static constexpr double kMargin       = 0.05;

    // Whether `proposal` (count >= 1) is expected to commit more drafts than the MTP's
    // `mtp_extent` drafts would.
    [[nodiscard]] bool prefer(const PromptLookupProposal& proposal,
                              std::uint32_t mtp_extent) const;

    void observe_lookup(std::uint32_t match, std::uint32_t drafted, std::uint32_t accepted);
    void observe_mtp(std::uint32_t drafted, std::uint32_t accepted);
    // A proposal of this match length was passed over for the MTP's drafts.
    void observe_skipped(std::uint32_t match);

    [[nodiscard]] double lookup_rate(std::uint32_t match) const;
    [[nodiscard]] double mtp_expected(std::uint32_t extent) const;

private:
    [[nodiscard]] static std::size_t bucket(std::uint32_t match);

    std::array<double, kBuckets> lookup_accepted_{};
    std::array<double, kBuckets> lookup_rejected_{};
    double mtp_accepted_ = 0.0; // EMA of accepted MTP drafts per round
    double mtp_drafted_  = 0.0; // EMA of MTP drafts per round
    bool mtp_observed_   = false;
};

} // namespace ninfer::models::qwen3_8_flash_next
