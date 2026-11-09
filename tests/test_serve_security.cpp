// Host-only contract tests for the serving security primitives: the constant-time credential
// comparison, the kernel-CSPRNG identifiers, and the loopback request guard.

#include "serve/security.h"

#include <iostream>
#include <string>
#include <string_view>

namespace {

using namespace ninfer::serve;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int test_secure_equals() {
    int failures = 0;
    failures += check(secure_equals("", ""), "two empty strings are equal");
    failures += check(secure_equals("sk-local-key", "sk-local-key"), "identical strings are equal");
    failures += check(!secure_equals("sk-local-key", "sk-local-kez"), "a one-byte difference differs");
    failures += check(!secure_equals("sk-local-key", "sk-local-ke"), "a prefix is not equal");
    failures += check(!secure_equals("sk-local-ke", "sk-local-key"), "a prefix is not equal reversed");
    failures += check(!secure_equals("", "sk-local-key"), "empty differs from non-empty");
    failures += check(!secure_equals("sk-local-key", ""), "non-empty differs from empty");
    // A difference in the first byte and a difference in the last must both be reported.
    failures += check(!secure_equals("Xk-local-key", "sk-local-key"), "first-byte difference");
    failures += check(!secure_equals("sk-local-keX", "sk-local-key"), "last-byte difference");
    // Embedded NULs are compared, not treated as terminators.
    const std::string with_nul("a\0b", 3);
    const std::string other_nul("a\0c", 3);
    failures += check(secure_equals(with_nul, with_nul), "identical strings with an embedded NUL");
    failures += check(!secure_equals(with_nul, other_nul), "embedded NUL compared, not skipped");
    return failures;
}

int test_secure_hex() {
    int failures = 0;
    const std::string sixteen = secure_hex(8);
    failures += check(sixteen.size() == 16, "8 bytes render as 16 hex characters");
    failures += check(sixteen.find_first_not_of("0123456789abcdef") == std::string::npos,
                      "identifiers are lowercase hex only");

    const std::string thirty_two = secure_hex(16);
    failures += check(thirty_two.size() == 32, "16 bytes render as 32 hex characters");

    failures += check(secure_hex(0).empty(), "zero bytes render as an empty string");

    // Two draws from a CSPRNG must not repeat. The chance of a collision here is negligible; the
    // check exists so a future replacement with a deterministic generator is caught.
    failures += check(secure_hex(8) != secure_hex(8), "successive identifiers differ");

    // The cap keeps the fixed buffer honest rather than silently truncating an oversized request.
    failures += check(secure_hex(1000).size() == 128, "an oversized request is capped at 64 bytes");
    return failures;
}

int test_is_loopback_host() {
    int failures = 0;
    for (const std::string_view loopback : {"localhost", "127.0.0.1", "127.1.2.3", "::1", "[::1]",
                                            "localhost:8000", "127.0.0.1:3080", "[::1]:3080",
                                            "localhost."}) {
        failures += check(is_loopback_host(loopback),
                          "loopback host accepted: " + std::string(loopback));
    }
    for (const std::string_view foreign : {"", "0.0.0.0", "example.com", "example.com:8000",
                                           "192.168.1.10", "10.0.0.5:8000", "127.0.0.1.example.com",
                                           "localhost.example.com", "[2001:db8::1]", "127.0.0"}) {
        failures += check(!is_loopback_host(foreign),
                          "non-loopback host refused: " + std::string(foreign));
    }
    return failures;
}

int test_loopback_guard() {
    struct Case {
        std::string_view bind;
        std::string_view host;
        std::string_view origin;
        bool cors;
        bool refused;
        const char* why;
    };
    const Case cases[] = {
        // A server on a routable address is not guarded: the operator chose to expose it.
        {"0.0.0.0", "example.com", "https://example.com", false, false, "routable bind is not guarded"},
        {"", "example.com", "https://example.com", false, false, "empty bind is not guarded"},
        // A loopback server with a loopback Host is the normal local client.
        {"127.0.0.1", "127.0.0.1:3080", "", false, false, "loopback host header allowed"},
        {"127.0.0.1", "localhost:3080", "", false, false, "localhost host header allowed"},
        {"127.0.0.1", "[::1]:3080", "", false, false, "IPv6 loopback host header allowed"},
        {"127.0.0.1", "", "", false, false, "a missing Host header is allowed"},
        // DNS rebinding: the browser sends the attacker's name to a loopback server.
        {"127.0.0.1", "attacker.example:3080", "", false, true, "foreign host header refused"},
        {"127.0.0.1", "127.0.0.1.attacker.example", "", false, true, "suffixed loopback name refused"},
        {"127.0.0.1", "127.0.0", "", false, true, "a truncated quad is not loopback"},
        {"127.0.0.1", "127.0.0.1.1", "", false, true, "a five-part quad is not loopback"},
        {"127.0.0.1", "127.999.0.1", "", false, true, "an out-of-range octet is not loopback"},
        // Cross-origin browser reads, with --cors off.
        {"127.0.0.1", "127.0.0.1", "https://attacker.example", false, true, "foreign origin refused"},
        {"127.0.0.1", "127.0.0.1", "null", false, true, "sandboxed origin refused"},
        {"127.0.0.1", "127.0.0.1", "http://localhost:5173", false, false, "loopback origin allowed"},
        {"127.0.0.1", "127.0.0.1", "http://127.0.0.1:5173", false, false, "IPv4 loopback origin allowed"},
        {"127.0.0.1", "127.0.0.1", "http://[::1]:5173", false, false, "IPv6 loopback origin allowed"},
        // A non-browser client sends no Origin; the Host check still applies.
        {"127.0.0.1", "attacker.example", "", false, true, "foreign host without origin refused"},
        // --cors is the operator's explicit opt-in to browser origins: the Origin check yields,
        // the Host check does not, because rebinding defence does not depend on the caller.
        {"127.0.0.1", "127.0.0.1", "https://app.example", true, false, "cors waives the origin check"},
        {"127.0.0.1", "127.0.0.1", "null", true, false, "cors waives the sandboxed origin check"},
        {"127.0.0.1", "attacker.example", "https://app.example", true, true, "cors does not waive rebinding"},
    };
    int failures = 0;
    for (const Case& c : cases) {
        const bool refused =
            refuse_loopback_foreign_request(c.bind, c.host, c.origin, c.cors);
        failures += check(refused == c.refused,
                          std::string("loopback guard: ") + c.why +
                              " (bind=" + std::string(c.bind) + " host=" + std::string(c.host) +
                              " origin=" + std::string(c.origin) +
                              " cors=" + (c.cors ? "on" : "off") + ")");
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_secure_equals();
    failures += test_secure_hex();
    failures += test_is_loopback_host();
    failures += test_loopback_guard();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
