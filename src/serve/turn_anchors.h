// SPDX-FileCopyrightText: ValerioDolci/capy-flash
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Turn anchors (--turn-anchors N). Ported from capy-flash (ValerioDolci/capy-flash v0.9.0,
// src/serve/turn_anchors.{h,cpp}, Apache-2.0); see PROVENANCE.md.
//
// A same-session follow-up consumes the previous turn's continuation, and its own turn-closure
// capture replaces the previous closure, so once turn N+1 is running no checkpoint is left at the
// end of turn N: a cancelled or edited turn re-prefilled the whole conversation. A private
// long-anchor opportunity at the message boundary before the final message (and, with N > 1,
// before the messages preceding it) makes the follow-up's prefill capture that branch point; the
// Engine keeps and replaces it like an explicit anchor, within
// ContextCacheOptions::max_long_anchors_per_continuation.
//
// Protocol markers stay first: the Engine's marker order follows this vector's order, and the
// anchors are the first to go when the protocol markers already fill the explicit-marker budget.
// A client that sends the maximum number of Anthropic cache_control breakpoints must not be
// rejected because the serving layer appended its own anchor on top.

#include "ninfer/types.h"

#include <cstdint>

namespace ninfer::serve {

// Appends up to `turn_anchors` private long-anchor markers to `input.context_cache.markers`,
// after any markers already there, without exceeding kMaximumExplicitPromptCacheMarkers.
// A value of 0 appends nothing, which is the default and leaves the request unchanged.
void add_turn_anchor_markers(PromptInput& input, std::uint32_t turn_anchors);

} // namespace ninfer::serve
