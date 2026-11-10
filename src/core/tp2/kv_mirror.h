#pragma once

// tp2 rank-1 mirrors: the physical mutations rank 0 issues on its KV page pool, KV execution
// tables and StateImage pool are replayed at the same index on rank 1's pool, on rank 1's stream.
//
// The contract (capy-flash docs/maintainer/tensor-parallel.md §6, src/core/tp2/paged_kv_cache_tp2.inc):
//
//   * attach a mirror before the origin's first mutation: both replicas must start from the same
//     empty addressing, and a mirror is refused while any execution row is bound;
//   * rank 1 never allocates and never acquires on its own: rank 0's pool owns every page and row
//     decision, and rank 1's replica is addressed by the same physical indices;
//   * a replica is index-identical, not byte-identical: at tp 2 rank 1 carries its own KV-head
//     shard, so only page counts and per-page plane geometry must agree;
//   * every replay runs on rank 1's stream. Both ranks issue the same sequence in the same order,
//     which is the whole ordering argument: no cross-stream synchronization is involved;
//   * Host transfers are rejected while a mirror is attached, because tp 2 has no Host KV or state
//     tier. A path that would spill or restore state must be disabled at option normalization.
//
// These wrappers are the drop-in mutation surface for a mirrored rank-0 pool. Nothing here reads or
// rewrites the origin's bookkeeping; the origin's own methods remain the authority for it.

#include "core/device.h"
#include "core/device_kv_page_pool.h"
#include "core/state_image.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::tp2 {

// Where rank 1's replicas live. `device` owns the stream facts and outlives every mirror; `stream`
// and `transfer_stream` are borrowed from it.
struct MirrorTarget {
    const DeviceContext* device  = nullptr;
    cudaStream_t stream          = nullptr;
    cudaStream_t transfer_stream = nullptr;
    // Orders rank 1's transfer-stream work for a context transfer that both ranks must finish.
    // Recorded by the owning Program, awaited on the origin side as well.
    CudaCompletionEvent* transfer_order = nullptr;

    [[nodiscard]] bool valid() const noexcept { return device != nullptr && stream != nullptr; }
};

// Validates that `mirror` can replay `origin`'s physical page mutations: equal page capacity,
// equal plane count and equal per-page plane geometry. Total plane bytes may differ, because rank 1
// carries its own KV-head shard. Throws std::invalid_argument.
void require_matching_kv_mirror(const DeviceKVPagePool& origin, const DeviceKVPagePool& mirror);

// Validates that `mirror` can replay `origin`'s slot mutations: equal image stride, slot count and
// field layout. Pure: both storages may be host-backed, which is what the host-only test uses.
void require_matching_state_image_mirror(const StateImageStorage& origin,
                                         const StateImageStorage& mirror);

// Rank 0's KV page pool with rank 1's replica attached. Construct once, before any page is zeroed.
class MirroredKVPagePool {
public:
    MirroredKVPagePool(DeviceKVPagePool& origin, DeviceKVPagePool& mirror, MirrorTarget where);

    MirroredKVPagePool(const MirroredKVPagePool&)            = delete;
    MirroredKVPagePool& operator=(const MirroredKVPagePool&) = delete;

    [[nodiscard]] const MirrorTarget& target() const noexcept { return where_; }
    [[nodiscard]] DeviceKVPagePool& origin_pool() noexcept { return origin_; }
    [[nodiscard]] DeviceKVPagePool& mirror_pool() noexcept { return mirror_; }

    void zero_pages(std::span<const DeviceKVPageHandle> pages, cudaStream_t stream = nullptr);
    void copy_page(DeviceKVPageHandle source, DeviceKVPageHandle destination,
                   cudaStream_t stream = nullptr);

    // tp 2 has no Host KV tier: rank 1 has no Host copy of KV, so a spill or restore would leave
    // the ranks' addressing divergent. These throw std::logic_error instead of half-copying.
    void copy_to_host(std::span<const DeviceKVPageHandle> source, HostKVAllocationView destination,
                      cudaStream_t stream = nullptr) const;
    void copy_from_host(HostKVAllocationConstView source,
                        std::span<const DeviceKVPageHandle> destination,
                        cudaStream_t stream = nullptr) const;

private:
    DeviceKVPagePool& origin_;
    DeviceKVPagePool& mirror_;
    MirrorTarget where_;
};

// Rank 0's KV execution tables with rank 1's attached. Rank 1 never acquires on its own: its lease
// is taken here, mirror first, and dropped with rank 0's, so there is one release path.
class MirroredKVExecutionTables {
public:
    MirroredKVExecutionTables(MirroredKVPagePool& pages, KVExecutionTablePool& origin,
                              KVExecutionTablePool& mirror);

    MirroredKVExecutionTables(const MirroredKVExecutionTables&)            = delete;
    MirroredKVExecutionTables& operator=(const MirroredKVExecutionTables&) = delete;

    // Takes rank 1's lease on `row` first, then rank 0's. A failure on either side leaves both
    // pools unchanged and returns no lease.
    [[nodiscard]] KVExecutionRowLease acquire(std::int32_t row);
    // Drops rank 0's lease and the paired rank-1 lease. False when the lease was already stale, in
    // which case the paired row was dropped with the earlier release.
    bool release(KVExecutionRowLease lease);

    void publish(KVExecutionRowHandle row, std::uint32_t logical_begin,
                 std::span<const DeviceKVPageHandle> pages, cudaStream_t stream = nullptr);
    void publish_repeated(KVExecutionRowHandle row, DeviceKVPageHandle page, std::uint32_t count,
                          cudaStream_t stream = nullptr);

    // Rank 1's lease of `row`; the caller reads rank 1's row view through the mirror pool.
    [[nodiscard]] const KVExecutionRowLease& mirror_row(std::int32_t row) const;

private:
    [[nodiscard]] KVExecutionRowHandle require_mirror_handle(std::int32_t row) const;

    MirroredKVPagePool& pages_;
    KVExecutionTablePool& origin_;
    KVExecutionTablePool& mirror_;
    MirrorTarget where_;
    std::vector<std::optional<KVExecutionRowLease>> rows_;
};

// Rank 0's StateImage pool with rank 1's replica attached. Every slot operation is replayed at the
// same slot index on rank 1. The origin's execution view stays the caller's; rank 1's is derived
// from the mirror target.
class MirroredStateImagePool {
public:
    MirroredStateImagePool(StateImageStorage& origin, StateImageStorage& mirror, MirrorTarget where);

    MirroredStateImagePool(const MirroredStateImagePool&)            = delete;
    MirroredStateImagePool& operator=(const MirroredStateImagePool&) = delete;

    [[nodiscard]] const MirrorTarget& target() const noexcept { return where_; }
    [[nodiscard]] StateImageStorage& origin_storage() noexcept { return origin_; }
    [[nodiscard]] StateImageStorage& mirror_storage() noexcept { return mirror_; }
    // Rank 1's execution view for its own stream; the mirror target must be attached.
    [[nodiscard]] CudaExecutionView mirror_execution() const;

    void clear(std::int32_t slot, CudaExecutionView origin_execution);
    void clear_all(CudaExecutionView origin_execution);
    void copy(std::int32_t source, std::int32_t destination, CudaExecutionView origin_execution);
    void copy_group(LayoutRegion group, std::int32_t source, std::int32_t destination,
                    CudaExecutionView origin_execution);

    // tp 2 has no Host state tier; see MirroredKVPagePool::copy_to_host.
    void copy_to_host(std::int32_t source, MemorySpan destination,
                      CudaExecutionView origin_execution) const;
    void copy_from_host(ConstMemorySpan source, std::int32_t destination,
                        CudaExecutionView origin_execution) const;

private:
    StateImageStorage& origin_;
    StateImageStorage& mirror_;
    MirrorTarget where_;
};

} // namespace ninfer::tp2
