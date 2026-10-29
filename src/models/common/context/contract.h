#pragma once

#include "core/spill_kv_store.h"
#include "models/common/context/directory/history.h"
#include "models/common/context/directory/state.h"

#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

namespace ninfer::models::context {

struct StateClear {
    std::int32_t slot = -1;
};

struct StateCopy {
    std::int32_t source      = -1;
    std::int32_t destination = -1;
    // Zero bytes copies the complete image. A field group is relative to one image.
    std::size_t offset = 0;
    std::size_t bytes  = 0;
};

struct PageCopy {
    PageSlot source;
    PageSlot destination;
};

struct PageTransfer {
    StateCopyDirection direction = StateCopyDirection::DeviceToHost;
    HostExtent host;
    std::vector<PageSlot> pages;
};

struct MappingUpdate {
    TableRow row;
    std::uint32_t logical_begin = 0;
    std::vector<PageSlot> pages;
};

// Plain host copies against the engine's spill file, executed synchronously inside the
// migration: spill-outs run at the boundary that frees Host capacity, page-ins at admission.
// The spill store pointer rides on ContextActions because the session owns the store and the
// executor only borrows it for the duration of the actions.
struct StateSpillOut {
    HostExtent host;            // source bytes in the pinned backing
    SpillKVStore::Extent spill; // destination extent, written by the executor
};

struct StateSpillIn {
    SpillKVStore::Extent spill; // source extent in the engine's spill store
    HostExtent host;            // destination bytes in the pinned backing
};

using ContextAction = std::variant<StateClear, StateCopy, StateTransferDescription, PageCopy,
                                   PageTransfer, MappingUpdate, StateSpillOut, StateSpillIn>;

// Owns all descriptors and mapping arrays. Source leases and destination owners remain in the
// corresponding control-side transaction until its execution dependency has retired.
struct ContextActions {
    std::vector<ContextAction> operations;
    SpillKVStore* spill_store = nullptr;

    [[nodiscard]] bool empty() const noexcept { return operations.empty(); }
};

} // namespace ninfer::models::context
