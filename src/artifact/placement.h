#pragma once

#include "artifact/encoded_slice.h"
#include "core/layout.h"

#include <span>
#include <vector>

namespace ninfer::artifact {

class Reader;

struct PreparedResource {
    ObjectHandle object;
    std::vector<std::byte> data;
};

// Moving this owner preserves the addresses borrowed by parsed frontend resources.
struct PreparedResources {
    PreparedResources()                                        = default;
    PreparedResources(PreparedResources&&) noexcept            = default;
    PreparedResources& operator=(PreparedResources&&) noexcept = default;
    PreparedResources(const PreparedResources&)                = delete;
    PreparedResources& operator=(const PreparedResources&)     = delete;

    const Reader* source = nullptr;
    std::vector<PreparedResource> objects;
    std::uint64_t read_bytes        = 0;
    std::uint64_t owned_value_bytes = 0;
};

enum class HostAccess { CpuOnly, CudaTransfer };

struct PlacementRequest {
    ObjectHandle object;
    Device device;
    HostAccess host_access = HostAccess::CpuOnly;
};

// SP-4 tiered routed bank: the hot experts (the `new_order` prefix, expert id order)
// materialize as a compact hot mini-bank on the device; the warm experts (the `new_order`
// suffix) live in a pinned-host warm mini-bank that the materializer allocates and fills.
// Parts address their expert's compacted position in the matching mini-bank.
struct TierPlanInput {
    ObjectHandle object;
    std::vector<std::uint32_t> new_order;
    std::uint32_t hot_experts = 0;
    std::uint64_t alignment   = 256;
};

// SP-4 adaptive cache: one uniform slot count across every tiered bank — a promoted
// expert's paired banks must carry matching slots, and one slot costs its per-expert bytes
// in every bank. The device budget covers the whole plan: the plain device objects and the
// tiered hot mini-banks consume first, and the remaining bytes buy as many whole expert
// slices as fit, never exceeding the smallest bank's warm count.
struct TierBudget {
    std::uint64_t device_bytes = 0;
    std::uint32_t slot_cap     = 0;
};

struct TierPlacementRequest {
    ObjectHandle object;
    std::uint64_t offset      = 0;
    std::uint64_t bytes       = 0; // device mini-bank bytes: hot experts plus the cache tail.
    std::uint64_t alignment   = 256;
    std::uint32_t hot_experts = 0;
    std::uint32_t cache_slots = 0;
    std::vector<std::uint32_t> new_order;
};

struct TensorPlacement {
    StorageId storage;
    LayoutRegion region;
};

struct TensorRequest {
    EncodedTensor tensor;
    Device device;
    HostAccess host_access = HostAccess::CpuOnly;
    // Nonzero domains keep separately packed groups from sharing individual matrices.
    std::uint64_t storage_domain = 0;
};

struct MaterializationPlan {
    const Reader* source     = nullptr;
    std::size_t object_count = 0;
    Device device;
    std::size_t device_capacity_bytes = 0;
    std::size_t host_capacity_bytes   = 0;
    std::size_t host_alignment        = 256;
    std::size_t registered_host_bytes = 0; // Aligned prefix of the Host backing.
    std::vector<EncodedTensor> tensors;
    std::vector<StorageId> requests;
    std::vector<std::optional<StorageId>> objects;
    std::vector<TensorPlacement> device_objects;
    // SP-4 tiered banks placed after the plain device objects.
    std::vector<TierPlacementRequest> tier_objects;
    // Host placements are relative to their respective registered/ordinary regions.
    std::vector<TensorPlacement> ordinary_host_objects;
    std::vector<TensorPlacement> registered_host_objects;
    PreparedResources resources;
};

// Preserve the source page offset so aligned interiors can be read directly into Host backing.
[[nodiscard]] LayoutRegion plan_host_tensor_region(LayoutBuilder&, const Reader&, ObjectHandle);

// Deduplicates complete parents independently at each requested location. Tensor placement
// never reads payloads; only already-prepared resource bytes move through this plan.
[[nodiscard]] MaterializationPlan plan_materialization(const Reader& reader,
                                                       std::span<const PlacementRequest> requests,
                                                       PreparedResources resources = {});

// The tier-aware form: the plain placements plan first, then every tiered bank gains a
// compact hot mini-bank plus its budget-derived cache tail.
[[nodiscard]] MaterializationPlan
plan_materialization(const Reader& reader, std::span<const PlacementRequest> requests,
                     std::span<const TierPlanInput> tiers, TierBudget budget,
                     PreparedResources resources = {});

[[nodiscard]] MaterializationPlan plan_materialization(const Reader&,
                                                       std::span<const TensorRequest>,
                                                       PreparedResources resources = {});

[[nodiscard]] MaterializationPlan plan_materialization(const Reader&,
                                                       std::span<const TensorRequest> requests,
                                                       std::span<const TierPlanInput> tiers,
                                                       TierBudget budget,
                                                       PreparedResources resources = {});

} // namespace ninfer::artifact
