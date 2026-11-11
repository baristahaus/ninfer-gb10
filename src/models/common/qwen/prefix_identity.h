#pragma once

// Compact host identity for the inputs represented by resident model state.

#include "models/common/qwen/frontend/prepared_prompt.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::models::qwen {

template <class Prompt>
using PromptVisionItems = std::remove_reference_t<decltype(std::declval<Prompt&>().vision_items)>;

template <class Prompt>
using PromptVisionItem = typename PromptVisionItems<Prompt>::value_type;

template <class Prompt>
class ResidentPrefixIdentityT {
public:
    void reserve(std::size_t tokens);
    void clear() noexcept;
    void assign(const Prompt& prompt);
    void swap(ResidentPrefixIdentityT& other) noexcept;
    void prepare_generated(std::size_t count, std::int32_t rope_delta,
                           std::optional<std::uint32_t> execution_split_after = std::nullopt);
    void append_generated(std::size_t count, std::int32_t rope_delta,
                          std::optional<std::uint32_t> execution_split_after = std::nullopt);
    void truncate(std::size_t tokens);

    [[nodiscard]] std::size_t size() const noexcept { return token_types_.size(); }

    [[nodiscard]] std::span<const std::uint32_t> execution_frontiers() const noexcept {
        return rewrite_execution_frontiers_;
    }

    [[nodiscard]] bool matches(const Prompt& prompt, std::size_t count) const;
    [[nodiscard]] bool equals(const ResidentPrefixIdentityT& other) const;
    [[nodiscard]] bool prefix_equals(const ResidentPrefixIdentityT& other, std::size_t count) const;

private:
    std::vector<std::uint8_t> token_types_;
    std::array<std::vector<std::int32_t>, 3> positions_;
    std::vector<PromptVisionItem<Prompt>> vision_items_;
    std::vector<std::uint32_t> rewrite_execution_frontiers_;
    // Exact proof for the initial text run: type zero and position[axis][i] = origin[axis] + i.
    std::size_t regular_prefix_ = 0;
    std::array<std::int32_t, 3> position_origins_{};
};

// One rolling digest per token frontier. This is only a content shortlist: exact token and
// ResidentPrefixIdentity comparison remains authoritative for reuse. Keeping it separate from the
// exact identity avoids retaining hash-only state in immutable capture backings.
template <class Prompt>
class PrefixShortlistDigestsT {
public:
    void reserve(std::size_t tokens);
    void clear() noexcept;
    void assign(const Prompt& prompt);
    void swap(PrefixShortlistDigestsT& other) noexcept;
    void prepare_generated(std::size_t count, std::int32_t rope_delta,
                           std::optional<std::uint32_t> execution_split_after = std::nullopt);
    void append_generated(std::span<const TokenId> tokens, std::int32_t rope_delta,
                          std::optional<std::uint32_t> execution_split_after = std::nullopt);
    void truncate(std::size_t tokens);

    [[nodiscard]] std::size_t size() const noexcept {
        return digests_.empty() ? 0 : digests_.size() - 1U;
    }

    [[nodiscard]] std::array<std::uint64_t, 2> at(std::size_t frontier) const;

private:
    std::vector<std::array<std::uint64_t, 2>> digests_;
};

// A borrowed view of one complete immutable recovery point. Backing identity is a fast proof
// only after the frontier, domain and digest checks; it never equates physical State/KV owners.
template <class Prompt>
struct RecoveryIdentityViewT {
    std::uint32_t frontier;
    std::uint32_t domain;
    std::span<const TokenId> tokens;
    const ResidentPrefixIdentityT<Prompt>& positions;
    const PrefixShortlistDigestsT<Prompt>& digests;
    const void* backing;
};

template <class Prompt>
[[nodiscard]] bool recovery_prefix_matches(const RecoveryIdentityViewT<Prompt>& earlier,
                                           const RecoveryIdentityViewT<Prompt>& later);

template <class Prompt>
[[nodiscard]] bool prefix_matches(const Prompt& prompt, std::span<const TokenId> resident_tokens,
                                  const ResidentPrefixIdentityT<Prompt>& resident_identity,
                                  std::size_t count);

using ResidentPrefixIdentity = ResidentPrefixIdentityT<PreparedPromptData>;
using PrefixShortlistDigests = PrefixShortlistDigestsT<PreparedPromptData>;
using RecoveryIdentityView   = RecoveryIdentityViewT<PreparedPromptData>;

} // namespace ninfer::models::qwen
