#pragma once

#include "core/platform.h"

#include <iostream>

namespace ninfer::test {

// A Host context tier copies KV and state images out of device memory to free it. This target has a
// single physical memory pool shared by the CPU and the GPU, so there is no separate host tier to
// copy to: the engine rejects a non-zero capacity by design (src/models/common/context/storage.cpp,
// NINFER_SHARED_PHYSICAL_MEMORY). A test that asks for one must skip here, rather than fail on the
// target's own and correct refusal.
inline bool skip_without_host_tier(const char* test) {
#if NINFER_SHARED_PHYSICAL_MEMORY
    std::cout << "skip: " << test
              << " exercises a Host context tier, which this target does not have"
                 " (NINFER_SHARED_PHYSICAL_MEMORY: one pool shared by CPU and GPU)\n";
    return true;
#else
    (void)test;
    return false;
#endif
}

} // namespace ninfer::test
