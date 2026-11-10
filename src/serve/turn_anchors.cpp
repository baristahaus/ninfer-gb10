// Turn-anchor marker injection (serve/turn_anchors.h). The protocol markers own the
// explicit-marker budget; the anchors yield to it. Protocol markers are already in
// input.context_cache.markers when this runs (see GenerationService::prepare).
#include "serve/turn_anchors.h"

namespace ninfer::serve {

void add_turn_anchor_markers(PromptInput& input, std::uint32_t turn_anchors) {
    const std::size_t present = input.context_cache.markers.size();
    if (present >= kMaximumExplicitPromptCacheMarkers) { return; }
    const auto budget = static_cast<std::uint32_t>(kMaximumExplicitPromptCacheMarkers - present);
    if (turn_anchors > budget) { turn_anchors = budget; }
    // Fewer than two messages means there is no boundary before the final message to anchor.
    if (turn_anchors == 0 || input.messages.size() < 2) { return; }
    const std::size_t last_boundary = input.messages.size() - 1U;
    for (std::size_t count = 0; count < turn_anchors && last_boundary > count; ++count) {
        input.context_cache.markers.push_back(PromptCacheMarker{
            .after_message_count = static_cast<std::uint32_t>(last_boundary - count),
            .kind                = PromptCacheMarkerKind::PrivateLongAnchor,
            .evidence            = SharedCandidateEvidence::None,
            .location            = PromptCacheMarkerLocation::MessageBoundary,
        });
    }
}

} // namespace ninfer::serve
