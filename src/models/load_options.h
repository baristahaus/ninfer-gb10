#pragma once

#include "ninfer/types.h"

#include <string_view>

namespace ninfer::models {

struct LoadOptions {
    EnginePurpose purpose          = EnginePurpose::Generation;
    bool vision                    = false;
    SpeculativeBackend speculative = SpeculativeBackend::None;
    ProposalHead proposal_head     = ProposalHead::Full;
    // EngineOptions::rope_yarn_factor, 0 (or 1) = off.
    float rope_yarn_factor         = 0.0F;
    // EngineOptions::rope_original_max_position, 0 = the artifact's native window.
    std::uint32_t rope_original_max_position = 0;
    // The execution group this load is for: `width` 1 and `rank` 0 are the single-device lane and
    // are what every existing path gets. A load for `rank` at `width` installs the model's shard
    // transform on the binder, so the definition it reads holds that rank's bindings.
    int width = 1;
    int rank  = 0;

    bool operator==(const LoadOptions&) const = default;

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool dflash2() const noexcept {
        return speculative == SpeculativeBackend::DFlash2;
    }

    [[nodiscard]] bool masked_draft() const noexcept { return dflash() || dflash2(); }

    [[nodiscard]] bool proposal_enabled() const noexcept {
        return purpose == EnginePurpose::Generation && speculative != SpeculativeBackend::None &&
               proposal_head == ProposalHead::Optimized;
    }

    [[nodiscard]] std::string_view speculative_component() const noexcept {
        switch (speculative) {
        case SpeculativeBackend::None:
            return {};
        case SpeculativeBackend::Mtp:
            return "mtp";
        case SpeculativeBackend::DFlash:
            return "dflash";
        case SpeculativeBackend::DFlash2:
            return "dflash2";
        }
        return {};
    }
};

[[nodiscard]] constexpr bool is_masked_draft_backend(SpeculativeBackend backend) noexcept {
    return backend == SpeculativeBackend::DFlash || backend == SpeculativeBackend::DFlash2;
}

[[nodiscard]] inline LoadOptions load_options(const EngineOptions& options) noexcept {
    return {.purpose          = options.purpose,
            .vision           = options.enable_vision,
            .speculative      = options.speculative.backend,
            .proposal_head    = options.speculative.proposal_head,
            .rope_yarn_factor = options.rope_yarn_factor,
            .rope_original_max_position = options.rope_original_max_position,
            // Empty `devices` means the single `device` field, so the width is 1 and the load is
            // exactly the load it was before groups existed.
            .width = options.devices.empty() ? 1 : static_cast<int>(options.devices.size()),
            .rank  = 0};
}

} // namespace ninfer::models
