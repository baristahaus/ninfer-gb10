#pragma once

// json_output - tolerant post-generation cleaning for prompt-guided JSON
// response_format. NInfer performs no constrained decoding; this tolerant
// cleaner runs after generation: paired thinking leaks are dropped,
// markdown code fences are unwrapped, and the outermost JSON object is
// returned verbatim.
//
// Provenance (2026-09-27, Daphne): the original file was new-and-untracked
// in the v3-leg worktree on .24 and was lost in the 2026-09-27 re-image;
// the committed port tree references it (generation_service.cpp,
// tests/test_openai_schema.cpp) but omitted it from commit 85afc967. The
// body below is the tested desk original `extract_json_output`
// (daphne/ninfer-GB10 src/serve/generation_service.cpp; refs gb10/main and
// gitea/sm121a-tune, identical in both), relocated to the fork's header
// layout, with exactly one behavior change pinned by the fork tests: an
// UNCLOSED <think> fragment is left in place (the desk original erased it
// to the end of the string). Verified 2026-09-27 on the Mac: the 12-case
// harness (worklog/variants/json-extract-check) and the committed 8-case
// test_json_output_extract both pass.

#include <string>

namespace ninfer::serve::json_output {

// Tolerant-clean generated content into the JSON object. A complete object
// is returned verbatim; values containing the marker substring "think" and
// a newline are never damaged.
inline std::string extract(const std::string& text) {
    std::string s = text;

    // Strip thinking blocks (<think>...</think>) if any leaked into content.
    for (;;) {
        const auto open = s.find("<think>");
        if (open == std::string::npos) { break; }
        const auto close = s.find("</think>", open);
        if (close == std::string::npos) {
            break; // unclosed fragment: left in place, never erased to the end
        }
        s.erase(open, close + 8 - open);
    }

    // Strip markdown code fences: ```json\n...\n``` or ```\n...\n```
    const auto fence = s.find("```");
    if (fence != std::string::npos) {
        std::size_t start = fence + 3;
        // Skip optional language tag (json, etc.) on the same line.
        if (start < s.size() && s[start] != '\n') {
            const auto nl = s.find('\n', start);
            if (nl != std::string::npos) { start = nl + 1; }
        } else if (start < s.size()) {
            ++start; // skip the newline after ```
        }
        const auto close = s.rfind("```");
        if (close > start) {
            s = s.substr(start, close - start);
        }
    }

    // Find the outermost { ... } pair.
    const auto first = s.find('{');
    const auto last  = s.rfind('}');
    if (first != std::string::npos && last != std::string::npos && last > first) {
        return s.substr(first, last - first + 1);
    }
    return s;
}

} // namespace ninfer::serve::json_output
