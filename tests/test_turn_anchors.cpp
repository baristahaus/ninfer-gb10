// Turn-anchor marker injection (serve/turn_anchors.h). Host-only: the decision is a pure function
// of the message count, the requested anchor count and the markers already present, so it needs no
// Engine, no artifact and no GPU.
//
// What this covers: which anchors the serving layer asks the Engine to retain, and the budget rule
// against the protocol's own cache_control markers. What it does NOT cover: the Engine's retention
// and replacement of anchors at the max_long_anchors_per_continuation bound, which lives in the
// Program's checkpoint transactions (src/models/qwen3_8_flash_next/impl/runtime/program_impl.h,
// `anchor_limit`) and cannot run without a GPU and a loaded model.
#include "serve/turn_anchors.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::PromptCacheMarker;
using ninfer::PromptCacheMarkerKind;
using ninfer::PromptCacheMarkerLocation;
using ninfer::PromptInput;
using ninfer::SharedCandidateEvidence;

int failures = 0;
int checks   = 0;

void expect(bool condition, const std::string& label) {
    ++checks;
    if (condition) { return; }
    std::cerr << "FAIL: " << label << '\n';
    ++failures;
}

PromptInput with_messages(std::size_t count) {
    PromptInput input;
    input.messages.resize(count);
    return input;
}

PromptCacheMarker protocol_marker(std::uint32_t after_message_count) {
    return PromptCacheMarker{
        .after_message_count = after_message_count,
        .kind                = PromptCacheMarkerKind::SharedStablePrefix,
        .evidence            = SharedCandidateEvidence::ExplicitBoundary,
        .location            = PromptCacheMarkerLocation::MessageBoundary,
    };
}

// The frontend's validity rule for a message-boundary marker (see
// src/models/common/qwen/frontend/frontend.cpp, "context cache message marker is invalid"). The
// injected anchors must satisfy it, or prepare() throws and the request fails.
bool frontend_accepts(const PromptCacheMarker& marker, std::size_t message_count) {
    return marker.after_message_count <= message_count && marker.leading_instruction_bytes == 0 &&
           marker.after_tool_count == 0 && marker.after_message_part_count == 0 &&
           marker.location == PromptCacheMarkerLocation::MessageBoundary;
}

std::vector<std::uint32_t> boundaries(const PromptInput& input) {
    std::vector<std::uint32_t> out;
    for (const PromptCacheMarker& marker : input.context_cache.markers) {
        out.push_back(marker.after_message_count);
    }
    return out;
}

// The default value must change nothing: an existing request's marker vector is untouched.
void test_default_is_off() {
    PromptInput input = with_messages(4);
    ninfer::serve::add_turn_anchor_markers(input, 0);
    expect(input.context_cache.markers.empty(), "default: no marker is added");

    // A request that already carries protocol markers keeps exactly those.
    PromptInput with_protocol = with_messages(4);
    with_protocol.context_cache.markers.push_back(protocol_marker(2));
    ninfer::serve::add_turn_anchor_markers(with_protocol, 0);
    expect(with_protocol.context_cache.markers.size() == 1, "default: protocol markers stay");
    expect(with_protocol.context_cache.markers.front().kind ==
               PromptCacheMarkerKind::SharedStablePrefix,
           "default: the protocol marker is not rewritten");
}

// N = 1 anchors the boundary before the final message: the branch point a cancelled or edited
// final turn resumes from.
void test_one_anchor() {
    PromptInput input = with_messages(3);
    ninfer::serve::add_turn_anchor_markers(input, 1);
    expect(input.context_cache.markers.size() == 1, "one: exactly one marker");
    if (input.context_cache.markers.size() == 1) {
        const PromptCacheMarker& marker = input.context_cache.markers.front();
        expect(marker.after_message_count == 2, "one: the boundary before the final message");
        expect(marker.kind == PromptCacheMarkerKind::PrivateLongAnchor, "one: a private anchor");
        expect(marker.location == PromptCacheMarkerLocation::MessageBoundary,
               "one: a message-boundary marker");
        expect(marker.evidence == SharedCandidateEvidence::None, "one: no shared-candidate evidence");
        expect(frontend_accepts(marker, 3), "one: the frontend accepts the marker");
    }
}

// N > 1 walks back through the preceding boundaries, newest first, so an Engine that keeps the
// first max_long_anchors_per_continuation anchors keeps the newest turns.
void test_several_anchors_newest_first() {
    PromptInput input = with_messages(5);
    ninfer::serve::add_turn_anchor_markers(input, 3);
    expect(boundaries(input) == std::vector<std::uint32_t>({4, 3, 2}),
           "several: three anchors, newest boundary first");
    bool all_private = true;
    for (const PromptCacheMarker& marker : input.context_cache.markers) {
        all_private = all_private && marker.kind == PromptCacheMarkerKind::PrivateLongAnchor &&
                      frontend_accepts(marker, 5);
    }
    expect(all_private, "several: every anchor is a valid private long anchor");
}

// More anchors than there are boundaries before the final message: clamp, never repeat a boundary
// and never anchor after the final message.
void test_clamped_to_boundaries() {
    PromptInput input = with_messages(4);
    ninfer::serve::add_turn_anchor_markers(input, 10);
    expect(boundaries(input) == std::vector<std::uint32_t>({3, 2, 1}),
           "clamp: one anchor per boundary before the final message");

    PromptInput two = with_messages(2);
    ninfer::serve::add_turn_anchor_markers(two, 10);
    expect(boundaries(two) == std::vector<std::uint32_t>({1}), "clamp: a two-message conversation");

    // A single message has no boundary before it: nothing to anchor.
    PromptInput one = with_messages(1);
    ninfer::serve::add_turn_anchor_markers(one, 3);
    expect(one.context_cache.markers.empty(), "clamp: one message has no boundary");

    PromptInput none = with_messages(0);
    ninfer::serve::add_turn_anchor_markers(none, 3);
    expect(none.context_cache.markers.empty(), "clamp: no messages, no anchors");
}

// The protocol's cache_control breakpoints own the explicit-marker budget: the anchors yield, and
// the protocol markers keep their place and their order.
void test_protocol_budget_wins() {
    PromptInput input = with_messages(6);
    input.context_cache.markers.push_back(protocol_marker(1));
    input.context_cache.markers.push_back(protocol_marker(3));
    ninfer::serve::add_turn_anchor_markers(input, 4);
    expect(input.context_cache.markers.size() == 4, "budget: two protocol + two anchors");
    expect(input.context_cache.markers[0].after_message_count == 1 &&
               input.context_cache.markers[1].after_message_count == 3,
           "budget: the protocol markers stay first and unchanged");
    expect(input.context_cache.markers[2].after_message_count == 5 &&
               input.context_cache.markers[3].after_message_count == 4,
           "budget: the anchors take the remaining budget, newest first");
    bool kinds_ok = true;
    for (std::size_t index = 2; index < input.context_cache.markers.size(); ++index) {
        kinds_ok = kinds_ok && input.context_cache.markers[index].kind ==
                                   PromptCacheMarkerKind::PrivateLongAnchor;
    }
    expect(kinds_ok, "budget: the appended markers are private anchors");

    // A client that fills the budget exactly must not be rejected because the server wanted an
    // anchor: the frontend throws above four markers.
    PromptInput full = with_messages(6);
    for (std::uint32_t index = 0; index < ninfer::kMaximumExplicitPromptCacheMarkers; ++index) {
        full.context_cache.markers.push_back(protocol_marker(index + 1));
    }
    ninfer::serve::add_turn_anchor_markers(full, 4);
    expect(full.context_cache.markers.size() == ninfer::kMaximumExplicitPromptCacheMarkers,
           "budget: a full budget takes no anchor and stays within the limit");
    expect(boundaries(full) == std::vector<std::uint32_t>({1, 2, 3, 4}),
           "budget: the full budget is untouched");
}

// Idempotence of the ordering rule: the anchors are always a descending suffix of the boundaries.
void test_ordering_invariant() {
    for (std::size_t messages = 2; messages <= 7; ++messages) {
        for (std::uint32_t anchors = 0; anchors <= 5; ++anchors) {
            PromptInput input = with_messages(messages);
            ninfer::serve::add_turn_anchor_markers(input, anchors);
            const auto got = boundaries(input);
            bool ok        = got.size() <= anchors;
            for (std::size_t index = 0; index < got.size(); ++index) {
                ok = ok && got[index] == static_cast<std::uint32_t>(messages - 1 - index);
            }
            expect(ok, "ordering: anchors are the newest boundaries, in order");
        }
    }
}

} // namespace

int main() {
    test_default_is_off();
    test_one_anchor();
    test_several_anchors_newest_first();
    test_clamped_to_boundaries();
    test_protocol_budget_wins();
    test_ordering_invariant();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "turn anchors: " << checks << " checks passed\n";
    return 0;
}
