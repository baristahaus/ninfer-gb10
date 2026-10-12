#pragma once

#include <httplib.h>

namespace ninfer::serve {

// Browser clients send these in a preflight. The OpenAI and Anthropic JavaScript SDKs add the
// x-stainless-* identification headers, and the organization and browser-access headers when the
// page opts in. A wildcard is not allowed once a request carries Authorization, so each name is
// listed; a client header missing here fails its preflight.
inline constexpr const char* kCorsAllowHeaders =
    "Authorization, Content-Type, X-API-Key, anthropic-version, anthropic-beta, "
    "anthropic-dangerous-direct-browser-access, anthropic-user-profile-id, openai-beta, "
    "openai-organization, openai-project, x-stainless-arch, x-stainless-lang, x-stainless-os, "
    "x-stainless-package-version, x-stainless-retry-count, x-stainless-runtime, "
    "x-stainless-runtime-version, x-stainless-timeout";

// Lets browsers cache a preflight answer instead of repeating it before every request.
inline constexpr const char* kCorsMaxAgeSeconds = "600";

// Headers on every response: the static CORS grant, applied before routing so that error and
// streamed responses carry them too.
inline void apply_cors_defaults(httplib::Server& server) {
    server.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Expose-Headers", "x-request-id, request-id"},
        {"Access-Control-Allow-Headers", kCorsAllowHeaders},
        {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"},
    });
}

// The answer to any OPTIONS request, given without credentials so the real request can carry the
// API key. Chromium's Private Network Access asks a public page's preflight to acknowledge a
// private-network target, so it is echoed when requested.
inline void answer_cors_preflight(const httplib::Request& request, httplib::Response& response) {
    response.status = 204;
    response.set_header("Access-Control-Max-Age", kCorsMaxAgeSeconds);
    if (request.has_header("Access-Control-Request-Private-Network")) {
        response.set_header("Access-Control-Allow-Private-Network", "true");
    }
}

} // namespace ninfer::serve
