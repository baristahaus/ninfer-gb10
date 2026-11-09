#pragma once

// Security primitives shared by the serving layer: a constant-time comparison for credential
// checks, identifiers drawn from the kernel CSPRNG, and the loopback request guard that stops DNS
// rebinding and cross-origin browser requests.

#include <cstddef>
#include <string>
#include <string_view>

namespace ninfer::serve {

// Compares two strings without an early exit. The running time depends on the lengths, not on the
// position of the first difference, so a caller cannot learn a credential one byte at a time.
[[nodiscard]] bool secure_equals(std::string_view a, std::string_view b) noexcept;

// `bytes` bytes from the kernel CSPRNG as lowercase hex (2 * bytes characters). Throws
// std::runtime_error when the kernel refuses to supply entropy.
[[nodiscard]] std::string secure_hex(std::size_t bytes);

// True when `host` names the loopback interface: `localhost`, an address in `127.0.0.0/8`, or
// `::1`. The argument may carry a port and an IPv6 bracket (`localhost:8000`, `[::1]:8000`).
[[nodiscard]] bool is_loopback_host(std::string_view host) noexcept;

// True when a server bound to `bind_host` must refuse this request: it listens only on loopback
// and the request either is addressed to a foreign `Host` (DNS rebinding) or carries a browser
// `Origin` that is not loopback (a page in the user's browser reading the local server).
// Empty header values are treated as absent.
[[nodiscard]] bool refuse_loopback_foreign_request(std::string_view bind_host,
                                                   std::string_view host_header,
                                                   std::string_view origin_header) noexcept;

} // namespace ninfer::serve
