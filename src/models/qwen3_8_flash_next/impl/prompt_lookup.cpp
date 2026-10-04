#include "models/qwen3_8_flash_next/impl/prompt_lookup.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_8_flash_next {
namespace {

// Before a bucket has data: the longer the match, the likelier its continuation. Worth four
// drafts, so a few real rounds override them.
constexpr std::array<double, PromptLookup::kBuckets> kPriorRate{0.75, 0.88, 0.93, 0.96};
constexpr double kPriorWeight = 4.0;
constexpr double kDecay       = 0.97; // older lookup rounds fade
constexpr double kMtpAlpha    = 0.05; // EMA weight of a new MTP round
// Before any MTP round: about the per-draft acceptance measured on GB10 at K=1..3.
constexpr double kMtpPriorRate = 0.75;

// Expected accepted drafts of `count` drafts each accepted with `rate` after the one before.
[[nodiscard]] double geometric(double rate, std::uint32_t count) noexcept {
    double expected = 0.0;
    double run      = 1.0;
    for (std::uint32_t i = 0; i < count; ++i) {
        run *= rate;
        expected += run;
    }
    return expected;
}

[[nodiscard]] std::uint64_t mix(std::uint64_t x) noexcept {
    x ^= x >> 33U;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33U;
    x *= 0xc4ceb9fe1a85ec53ULL;
    return x ^ (x >> 33U);
}

} // namespace

void PromptLookup::reset() {
    table_.clear();
    used_    = 0;
    indexed_ = 0;
    lookup_accepted_.fill(0.0);
    lookup_rejected_.fill(0.0);
    mtp_accepted_ = 0.0;
    mtp_drafted_  = 0.0;
    mtp_observed_ = false;
}

std::uint64_t PromptLookup::key_at(std::span<const std::int32_t> history, std::size_t end) {
    const auto a = static_cast<std::uint32_t>(history[end - 2]);
    const auto b = static_cast<std::uint32_t>(history[end - 1]);
    const auto c = static_cast<std::uint32_t>(history[end]);
    return mix(mix(a * 0x9E3779B97F4A7C15ULL) ^ mix(b + 0x632BE59BD9B4E019ULL) ^
               (static_cast<std::uint64_t>(c) << 1U)) |
           1ULL;
}

std::size_t PromptLookup::bucket(std::uint32_t match) {
    return match < 6 ? 0 : match < 12 ? 1 : match < 24 ? 2 : 3;
}

const PromptLookup::Slot* PromptLookup::find(std::uint64_t key) const {
    if (table_.empty()) { return nullptr; }
    const std::size_t mask = table_.size() - 1;
    for (std::size_t i = key & mask;; i = (i + 1) & mask) {
        const Slot& slot = table_[i];
        if (slot.key == key) { return &slot; }
        if (slot.key == 0) { return nullptr; }
    }
}

PromptLookup::Slot& PromptLookup::insert(std::uint64_t key) {
    if ((used_ + 1) * 2 > table_.size()) { grow(); }
    const std::size_t mask = table_.size() - 1;
    for (std::size_t i = key & mask;; i = (i + 1) & mask) {
        Slot& slot = table_[i];
        if (slot.key == key) { return slot; }
        if (slot.key == 0) {
            slot.key = key;
            ++used_;
            return slot;
        }
    }
}

void PromptLookup::grow() {
    std::vector<Slot> old = std::move(table_);
    table_.assign(std::max<std::size_t>(4096, old.size() * 2), Slot{});
    const std::size_t mask = table_.size() - 1;
    for (const Slot& slot : old) {
        if (slot.key == 0) { continue; }
        std::size_t i = slot.key & mask;
        while (table_[i].key != 0) { i = (i + 1) & mask; }
        table_[i] = slot;
    }
}

void PromptLookup::sync(std::span<const std::int32_t> history) {
    if (history.size() < indexed_) {
        throw std::logic_error("prompt lookup history shrank while its sequence lived");
    }
    for (std::size_t end = std::max<std::size_t>(indexed_, kKeyTokens - 1); end < history.size();
         ++end) {
        Slot& slot = insert(key_at(history, end));
        for (std::uint32_t way = kWays - 1; way > 0; --way) { slot.ends[way] = slot.ends[way - 1]; }
        slot.ends[0] = static_cast<std::uint32_t>(end);
        slot.count   = std::min(slot.count + 1, kWays);
    }
    indexed_ = history.size();
}

PromptLookup::Proposal PromptLookup::propose(std::span<const std::int32_t> history,
                                             std::uint32_t max_tokens) const {
    Proposal proposal;
    if (history.size() != indexed_) {
        throw std::logic_error("prompt lookup proposes from an unsynchronized history");
    }
    if (history.size() < kKeyTokens + 1 || max_tokens == 0) { return proposal; }
    const std::size_t current = history.size() - 1;
    const Slot* slot          = find(key_at(history, current));
    if (slot == nullptr) { return proposal; }
    std::size_t best_end    = 0;
    std::uint32_t best_size = 0;
    for (std::uint32_t way = 0; way < slot->count; ++way) {
        const std::size_t end = slot->ends[way];
        if (end >= current) { continue; } // the current suffix itself
        std::uint32_t size = 0;
        while (size < kMaxMatch && size <= end && history[end - size] == history[current - size]) {
            ++size;
        }
        if (size > best_size) { // newest first, so ties keep the more recent occurrence
            best_size = size;
            best_end  = end;
        }
    }
    if (best_size < kKeyTokens) { return proposal; } // a hash collision, not a match
    proposal.match = best_size;
    // The continuation may run into the current suffix (periodic text); every token up to
    // `current` is history.
    const std::uint32_t limit = std::min(max_tokens, kMaxDrafts);
    for (std::size_t at = best_end + 1; at <= current && proposal.count < limit; ++at) {
        proposal.tokens[proposal.count++] = history[at];
    }
    return proposal;
}

double PromptLookup::lookup_rate(std::uint32_t match) const {
    const std::size_t b = bucket(match);
    return (lookup_accepted_[b] + kPriorWeight * kPriorRate[b]) /
           (lookup_accepted_[b] + lookup_rejected_[b] + kPriorWeight);
}

double PromptLookup::mtp_expected(std::uint32_t extent) const {
    if (extent == 0) { return 0.0; }
    if (!mtp_observed_) { return geometric(kMtpPriorRate, extent); }
    // Rounds draft the full window except near the budget or context end; a shorter extent keeps
    // the same share of its drafts.
    return mtp_drafted_ <= static_cast<double>(extent)
               ? mtp_accepted_
               : mtp_accepted_ * static_cast<double>(extent) / mtp_drafted_;
}

bool PromptLookup::prefer(const Proposal& proposal, std::uint32_t mtp_extent) const {
    if (proposal.count == 0) { return false; }
    return geometric(lookup_rate(proposal.match), proposal.count) >
           mtp_expected(mtp_extent) * (1.0 + kMargin);
}

void PromptLookup::observe_lookup(std::uint32_t match, std::uint32_t drafted,
                                  std::uint32_t accepted) {
    if (drafted == 0 || accepted > drafted) {
        throw std::logic_error("prompt lookup observed an invalid round");
    }
    const std::size_t b  = bucket(match);
    lookup_accepted_[b] = kDecay * lookup_accepted_[b] + accepted;
    lookup_rejected_[b] = kDecay * lookup_rejected_[b] + (accepted < drafted ? 1.0 : 0.0);
}

void PromptLookup::observe_mtp(std::uint32_t drafted, std::uint32_t accepted) {
    if (drafted == 0 || accepted > drafted) {
        throw std::logic_error("prompt lookup observed an invalid MTP round");
    }
    if (!mtp_observed_) { // the prior is the EMA's starting point, so one round cannot swing it
        mtp_accepted_ = geometric(kMtpPriorRate, drafted);
        mtp_drafted_  = drafted;
        mtp_observed_ = true;
    }
    mtp_accepted_ = (1.0 - kMtpAlpha) * mtp_accepted_ + kMtpAlpha * accepted;
    mtp_drafted_  = (1.0 - kMtpAlpha) * mtp_drafted_ + kMtpAlpha * drafted;
}

} // namespace ninfer::models::qwen3_8_flash_next
