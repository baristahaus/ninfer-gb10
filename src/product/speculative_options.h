#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "unknown";
}

// `--draft-tokens auto|N`. `auto` selects the per-round MTP draft count.
inline void parse_draft_tokens(std::string_view value, SpeculativeOptions& options) {
    if (value == "auto") {
        options.draft_tokens          = 0;
        options.adaptive_draft_tokens = true;
        return;
    }
    std::size_t used = 0;
    unsigned long parsed = 0;
    try {
        parsed = std::stoul(std::string(value), &used);
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid --draft-tokens value: " + std::string(value));
    }
    if (used != value.size() || parsed == 0 || parsed > 15) {
        throw std::invalid_argument("invalid --draft-tokens value: " + std::string(value));
    }
    options.draft_tokens          = static_cast<std::uint32_t>(parsed);
    options.adaptive_draft_tokens = false;
}

// Largest per-round MTP draft count an adaptive target may select.
inline constexpr std::uint32_t kAdaptiveMtpMaximumDraftTokens = 7;

// The largest draft count a configuration can use in one round.
[[nodiscard]] inline std::uint32_t maximum_draft_tokens(const SpeculativeOptions& options) {
    return options.adaptive_draft_tokens ? kAdaptiveMtpMaximumDraftTokens : options.draft_tokens;
}

// MTP without an explicit draft count uses the adaptive draft count.
inline void apply_speculative_cli_defaults(SpeculativeOptions& options) {
    if (options.backend == SpeculativeBackend::Mtp && options.draft_tokens == 0) {
        options.adaptive_draft_tokens = true;
    }
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.adaptive_draft_tokens ||
            options.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "--draft-tokens and --lm-head-draft require --spec mtp|dflash|dflash2");
        }
        return;
    case SpeculativeBackend::Mtp:
        if (options.adaptive_draft_tokens ? options.draft_tokens != 0
                                          : options.draft_tokens == 0 || options.draft_tokens > 7) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens auto or 1..7");
        }
        return;
    case SpeculativeBackend::DFlash:
        if (options.adaptive_draft_tokens || options.draft_tokens == 0 ||
            options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        return;
    case SpeculativeBackend::DFlash2:
        if (options.adaptive_draft_tokens || options.draft_tokens == 0 ||
            options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,15]");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninfer::product
