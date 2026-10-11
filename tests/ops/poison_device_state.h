#pragma once

#include <cstddef>
#include <cstdint>

namespace ninfer::test {

// Leaves `pattern` in every SM's shared memory, so a later kernel that reads shared memory it
// never wrote sees that value instead of whatever an earlier test kernel happened to leave.
void poison_shared_memory(std::uint32_t pattern);
// Fills device memory, e.g. a workspace, with one byte value before an Op runs over it.
void poison_device_bytes(void* data, std::size_t bytes, unsigned char value);

} // namespace ninfer::test
