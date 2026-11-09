#include "serve/security.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <stdexcept>

#include <sys/random.h>

namespace ninfer::serve {
namespace {

constexpr char kHexDigits[] = "0123456789abcdef";

// The host part of `host[:port]` or `[v6]:port`, without the brackets or a trailing dot.
std::string_view host_without_port(std::string_view host) noexcept {
    if (host.empty()) { return host; }
    if (host.front() == '[') {
        const std::size_t close = host.find(']');
        return close == std::string_view::npos ? host.substr(1) : host.substr(1, close - 1);
    }
    // A `host:port` pair has exactly one colon; a bare IPv6 address has several.
    const std::size_t colon = host.rfind(':');
    if (colon != std::string_view::npos && host.find(':') == colon) {
        host = host.substr(0, colon);
    }
    if (!host.empty() && host.back() == '.') { host.remove_suffix(1); }
    return host;
}

// True for a dotted-quad IPv4 literal whose first octet is 127 (127.0.0.0/8). A prefix test is
// not enough: `127.0.0.1.attacker.example` is a name an attacker controls that starts with `127.`,
// and accepting it would defeat the rebinding guard.
bool is_loopback_v4(std::string_view host) noexcept {
    std::size_t index  = 0;
    int octets         = 0;
    while (true) {
        std::size_t value  = 0;
        std::size_t digits = 0;
        while (index < host.size() && host[index] >= '0' && host[index] <= '9') {
            value = value * 10 + static_cast<std::size_t>(host[index] - '0');
            if (value > 255) { return false; }
            ++digits;
            ++index;
        }
        if (digits == 0 || digits > 3) { return false; }
        if (octets == 0 && value != 127) { return false; }
        ++octets;
        if (index == host.size()) { break; }
        if (host[index] != '.') { return false; }
        ++index;
        if (index == host.size()) { return false; }
        if (octets == 4) { return false; }
    }
    return octets == 4;
}

bool is_loopback_name(std::string_view host) noexcept {
    if (host == "localhost" || host == "::1") { return true; }
    return is_loopback_v4(host);
}

} // namespace

bool secure_equals(std::string_view a, std::string_view b) noexcept {
    // volatile keeps the fold out of a vectorised or short-circuiting rewrite; the loop always
    // runs over the longer operand so the work does not depend on where a difference is.
    const std::size_t length          = std::max(a.size(), b.size());
    volatile unsigned char difference = 0;
    for (std::size_t i = 0; i < length; ++i) {
        const unsigned char left  = i < a.size() ? static_cast<unsigned char>(a[i]) : 0U;
        const unsigned char right = i < b.size() ? static_cast<unsigned char>(b[i]) : 0U;
        difference = static_cast<unsigned char>(difference | (left ^ right));
    }
    difference = static_cast<unsigned char>(difference | (a.size() != b.size() ? 1U : 0U));
    return difference == 0;
}

std::string secure_hex(std::size_t bytes) {
    std::array<unsigned char, 64> raw{};
    if (bytes > raw.size()) { bytes = raw.size(); }
    std::size_t filled = 0;
    while (filled < bytes) {
        const ssize_t got = ::getrandom(raw.data() + filled, bytes - filled, 0);
        if (got < 0) {
            if (errno == EINTR) { continue; }
            throw std::runtime_error("the kernel CSPRNG is unavailable");
        }
        if (got == 0) { throw std::runtime_error("the kernel CSPRNG returned no bytes"); }
        filled += static_cast<std::size_t>(got);
    }
    std::string result(2 * bytes, '0');
    for (std::size_t i = 0; i < bytes; ++i) {
        result[2 * i]     = kHexDigits[raw[i] >> 4];
        result[2 * i + 1] = kHexDigits[raw[i] & 0x0FU];
    }
    return result;
}

bool is_loopback_host(std::string_view host) noexcept {
    return is_loopback_name(host_without_port(host));
}

bool refuse_loopback_foreign_request(std::string_view bind_host, std::string_view host_header,
                                     std::string_view origin_header,
                                     bool cross_origin_enabled) noexcept {
    if (!is_loopback_name(host_without_port(bind_host))) { return false; }
    // DNS rebinding: the name resolves to loopback, so the browser sends the attacker's host.
    // This applies whatever the CORS setting is; it is not an origin question.
    if (!host_header.empty() && !is_loopback_host(host_header)) { return true; }
    // An operator who passed --cors asked for browser origins, so the Origin check yields.
    if (origin_header.empty() || cross_origin_enabled) { return false; }
    // Sandboxed documents and file:// pages send `Origin: null`.
    if (origin_header == "null") { return true; }
    const std::size_t scheme = origin_header.find("://");
    const std::string_view authority =
        scheme == std::string_view::npos ? origin_header : origin_header.substr(scheme + 3);
    return !is_loopback_host(authority);
}

} // namespace ninfer::serve
