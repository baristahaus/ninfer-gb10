#pragma once

// json_output - tolerant post-generation cleaning for prompt-guided JSON response_format.
// NInfer performs no constrained decoding, so generated content may surround the JSON value
// with whitespace, a markdown fence, a leaked thinking block, or prose. The cleaner returns the
// first well-formed JSON object or array it can recover, byte-for-byte as generated.

#include <nlohmann/json.hpp>

#include <cstddef>
#include <string>
#include <string_view>

namespace ninfer::serve::json_output {
namespace detail {

inline std::string_view trim(std::string_view text) noexcept {
    constexpr std::string_view kSpace = " \t\r\n";
    const std::size_t first           = text.find_first_not_of(kSpace);
    if (first == std::string_view::npos) { return {}; }
    return text.substr(first, text.find_last_not_of(kSpace) - first + 1);
}

// A structured JSON value: an object or array that parses completely.
inline bool is_json_value(std::string_view text) {
    if (text.empty() || (text.front() != '{' && text.front() != '[')) { return false; }
    return nlohmann::json::accept(text.begin(), text.end());
}

// Drops closed <think>...</think> blocks. An unclosed opening tag is left in place.
inline std::string strip_thinking(std::string_view text) {
    constexpr std::string_view kOpen  = "<think>";
    constexpr std::string_view kClose = "</think>";
    std::string out;
    std::size_t position = 0;
    for (;;) {
        const std::size_t open = text.find(kOpen, position);
        if (open == std::string_view::npos) { break; }
        const std::size_t close = text.find(kClose, open + kOpen.size());
        if (close == std::string_view::npos) { break; }
        out.append(text.substr(position, open - position));
        position = close + kClose.size();
    }
    out.append(text.substr(position));
    return out;
}

// Body of the first markdown fence: after the opening fence line, before the last fence.
inline std::string_view fence_body(std::string_view text) {
    constexpr std::string_view kFence = "```";
    const std::size_t open            = text.find(kFence);
    if (open == std::string_view::npos) { return {}; }
    const std::size_t line_end = text.find('\n', open + kFence.size());
    if (line_end == std::string_view::npos) { return {}; }
    const std::size_t close = text.rfind(kFence);
    if (close <= line_end) { return {}; }
    return text.substr(line_end + 1, close - line_end - 1);
}

// First opening bracket whose span to a later matching closing bracket parses. Closers are
// tried from the right so the outermost value wins over a nested one.
inline std::string_view embedded_value(std::string_view text) {
    constexpr int kMaxAttempts = 256;
    int attempts               = 0;
    for (std::size_t open = text.find_first_of("{["); open != std::string_view::npos;
         open             = text.find_first_of("{[", open + 1)) {
        const char closer = text[open] == '{' ? '}' : ']';
        for (std::size_t close = text.rfind(closer); close != std::string_view::npos && close > open;
             close             = close == 0 ? std::string_view::npos : text.rfind(closer, close - 1)) {
            if (++attempts > kMaxAttempts) { return {}; }
            const std::string_view candidate = text.substr(open, close - open + 1);
            if (is_json_value(candidate)) { return candidate; }
        }
    }
    return {};
}

} // namespace detail

// Returns the JSON object or array carried by generated content, or the content unchanged when
// none can be recovered.
inline std::string extract(const std::string& text) {
    const std::string_view whole = detail::trim(text);
    if (detail::is_json_value(whole)) { return std::string(whole); }

    const std::string visible = detail::strip_thinking(text);
    const std::string_view fenced = detail::trim(detail::fence_body(visible));
    if (detail::is_json_value(fenced)) { return std::string(fenced); }

    const std::string_view embedded = detail::embedded_value(visible);
    if (!embedded.empty()) { return std::string(embedded); }
    return text;
}

} // namespace ninfer::serve::json_output
