// Implements: src/core/tp2/kv_mirror.h
#include "core/tp2/kv_mirror.h"

#include "core/host_kv_arena.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::tp2 {
namespace {

[[noreturn]] void throw_host_transfer_unavailable(const char* what) {
    throw std::logic_error(std::string(what) +
                           " is unavailable while a tp2 rank-1 mirror is attached: tp 2 has no "
                           "Host KV or state tier");
}

void require_target(const MirrorTarget& where, const char* what) {
    if (!where.valid()) {
        throw std::invalid_argument(std::string(what) + " requires rank 1's device and stream");
    }
}

// Runs `replay` with rank 1's device current. Stream-ordered copies and fills route by stream, so
// the scope is not strictly required; it keeps allocation and registration paths safe if a replay
// ever grows one.
template <typename Replay>
void on_mirror_device(const MirrorTarget& where, Replay&& replay) {
    const CudaDeviceScope scope(where.device->device());
    replay();
}

} // namespace

void require_matching_kv_mirror(const DeviceKVPagePool& origin, const DeviceKVPagePool& mirror) {
    if (&origin == &mirror) {
        throw std::invalid_argument("Paged KV mirror is the origin pool");
    }
    if (origin.capacity_pages() != mirror.capacity_pages()) {
        throw std::invalid_argument("Paged KV mirror pool page capacity differs");
    }
    if (origin.geometry().page_tokens != mirror.geometry().page_tokens ||
        origin.geometry().device_plane_order != mirror.geometry().device_plane_order ||
        origin.geometry().planes.size() != mirror.geometry().planes.size() ||
        origin.plane_count() != mirror.plane_count()) {
        throw std::invalid_argument("Paged KV mirror pool geometry differs");
    }
    // Per-page addressing must agree; the planes' total bytes may differ because rank 1 carries its
    // own KV-head shard.
    for (std::size_t index = 0; index < origin.plane_count(); ++index) {
        const Tensor& source  = origin.plane(index);
        const Tensor& replica = mirror.plane(index);
        if (origin.geometry().device_plane_order == PagedKVPlaneOrder::PageMajor) {
            if (source.nb[3] != replica.nb[3]) {
                throw std::invalid_argument("Paged KV mirror pool plane page stride differs");
            }
        } else if (source.nb[2] != replica.nb[2] || source.nb[3] != replica.nb[3] ||
                   source.ne[3] != replica.ne[3]) {
            throw std::invalid_argument("Paged KV mirror pool head-major plane geometry differs");
        }
    }
}

void require_matching_state_image_mirror(const StateImageStorage& origin,
                                         const StateImageStorage& mirror) {
    if (&origin == &mirror) {
        throw std::invalid_argument("StateImage mirror is the origin storage");
    }
    if (origin.slot_count() != mirror.slot_count()) {
        throw std::invalid_argument("StateImage mirror slot count differs");
    }
    if (origin.image_bytes() != mirror.image_bytes()) {
        throw std::invalid_argument("StateImage mirror image stride differs");
    }
    const StateImageLayout& source  = origin.layout();
    const StateImageLayout& replica = mirror.layout();
    if (source.fields.size() != replica.fields.size()) {
        throw std::invalid_argument("StateImage mirror field count differs");
    }
    for (std::size_t index = 0; index < source.fields.size(); ++index) {
        const StateFieldLayout& a = source.fields[index];
        const StateFieldLayout& b = replica.fields[index];
        if (a.tensor.region.offset != b.tensor.region.offset ||
            a.tensor.region.bytes != b.tensor.region.bytes || a.layers != b.layers ||
            a.layer_stride != b.layer_stride || a.slot_axis != b.slot_axis) {
            throw std::invalid_argument("StateImage mirror field layout differs");
        }
    }
}

// --- MirroredKVPagePool -------------------------------------------------------------------

MirroredKVPagePool::MirroredKVPagePool(DeviceKVPagePool& origin, DeviceKVPagePool& mirror,
                                       MirrorTarget where)
    : origin_(origin), mirror_(mirror), where_(where) {
    require_target(where_, "Paged KV mirror");
    require_matching_kv_mirror(origin_, mirror_);
}

void MirroredKVPagePool::zero_pages(std::span<const DeviceKVPageHandle> pages,
                                    cudaStream_t stream) {
    origin_.zero_pages(pages, stream);
    if (pages.empty()) { return; }
    // The same contiguous physical runs on rank 1, in the origin's order.
    std::size_t begin = 0;
    while (begin < pages.size()) {
        std::size_t end = begin + 1;
        const std::int32_t first = origin_.physical_page(pages[begin]);
        while (end < pages.size() &&
               origin_.physical_page(pages[end]) ==
                   origin_.physical_page(pages[end - 1]) + 1) {
            ++end;
        }
        const auto count = static_cast<std::uint32_t>(end - begin);
        on_mirror_device(where_, [&] { mirror_.zero_page_run(first, count, where_.stream); });
        begin = end;
    }
}

void MirroredKVPagePool::copy_page(DeviceKVPageHandle source, DeviceKVPageHandle destination,
                                   cudaStream_t stream) {
    origin_.copy_page(source, destination, stream);
    const std::int32_t source_index      = origin_.physical_page(source);
    const std::int32_t destination_index = origin_.physical_page(destination);
    if (source_index == destination_index) { return; }
    on_mirror_device(where_, [&] {
        mirror_.copy_page_run(source_index, destination_index, 1, where_.stream);
    });
}

void MirroredKVPagePool::copy_to_host(std::span<const DeviceKVPageHandle> source,
                                      HostKVAllocationView destination,
                                      cudaStream_t stream) const {
    (void)source;
    (void)destination;
    (void)stream;
    throw_host_transfer_unavailable("Paged KV Device-to-Host copy");
}

void MirroredKVPagePool::copy_from_host(HostKVAllocationConstView source,
                                        std::span<const DeviceKVPageHandle> destination,
                                        cudaStream_t stream) const {
    (void)source;
    (void)destination;
    (void)stream;
    throw_host_transfer_unavailable("Paged KV Host-to-Device restore");
}

// --- MirroredKVExecutionTables ------------------------------------------------------------

MirroredKVExecutionTables::MirroredKVExecutionTables(MirroredKVPagePool& pages,
                                                     KVExecutionTablePool& origin,
                                                     KVExecutionTablePool& mirror)
    : pages_(pages), origin_(origin), mirror_(mirror), where_(pages.target()),
      rows_(static_cast<std::size_t>(origin.row_count())) {
    if (&origin == &mirror) { throw std::invalid_argument("Paged KV mirror table is the origin"); }
    if (origin.logical_page_capacity() != mirror.logical_page_capacity() ||
        origin.row_count() != mirror.row_count()) {
        throw std::invalid_argument("Paged KV mirror execution table geometry differs");
    }
    // Both replicas must start from the same empty addressing (capy paged_kv_cache_tp2.inc:89).
    if (origin.any_row_bound() || mirror.any_row_bound()) {
        throw std::logic_error(
            "Paged KV mirror cannot be attached while an execution row is bound");
    }
}

KVExecutionRowLease MirroredKVExecutionTables::acquire(std::int32_t row) {
    if (row < 0 || static_cast<std::size_t>(row) >= rows_.size()) {
        throw std::out_of_range("Paged KV mirror execution row is out of range");
    }
    auto& slot = rows_[static_cast<std::size_t>(row)];
    if (slot.has_value()) {
        throw std::logic_error("Paged KV mirror execution row is already bound");
    }
    // Mirror first: if rank 0's acquire refuses, rank 1's lease is released and neither pool
    // changed.
    KVExecutionRowLease mirror_lease = mirror_.acquire(row);
    KVExecutionRowLease origin_lease = [&] {
        try {
            return origin_.acquire(row);
        } catch (...) {
            (void)mirror_lease.release();
            throw;
        }
    }();
    slot.emplace(std::move(mirror_lease));
    return origin_lease;
}

bool MirroredKVExecutionTables::release(KVExecutionRowLease lease) {
    const std::int32_t row = lease.row_index();
    const bool released    = lease.release();
    if (!released) { return false; }
    auto& slot = rows_.at(static_cast<std::size_t>(row));
    if (slot.has_value()) {
        (void)slot->release();
        slot.reset();
    }
    return true;
}

void MirroredKVExecutionTables::publish(KVExecutionRowHandle row, std::uint32_t logical_begin,
                                        std::span<const DeviceKVPageHandle> pages,
                                        cudaStream_t stream) {
    origin_.publish(row, logical_begin, pages, stream);
    const KVExecutionRowHandle mirror_row_handle = require_mirror_handle(row.row_index());
    std::vector<std::int32_t> indices(pages.size());
    for (std::size_t index = 0; index < pages.size(); ++index) {
        indices[index] = pages_.origin_pool().physical_page(pages[index]);
    }
    on_mirror_device(where_, [&] {
        mirror_.publish_physical(mirror_row_handle, logical_begin, indices, where_.stream);
    });
}

void MirroredKVExecutionTables::publish_repeated(KVExecutionRowHandle row, DeviceKVPageHandle page,
                                                 std::uint32_t count, cudaStream_t stream) {
    origin_.publish_repeated(row, page, count, stream);
    const KVExecutionRowHandle mirror_row_handle = require_mirror_handle(row.row_index());
    const std::int32_t physical                  = pages_.origin_pool().physical_page(page);
    std::vector<std::int32_t> indices(count, physical);
    on_mirror_device(where_, [&] {
        mirror_.publish_physical(mirror_row_handle, 0, indices, where_.stream);
    });
}

const KVExecutionRowLease& MirroredKVExecutionTables::mirror_row(std::int32_t row) const {
    if (row < 0 || static_cast<std::size_t>(row) >= rows_.size()) {
        throw std::out_of_range("Paged KV mirror execution row is out of range");
    }
    const auto& slot = rows_[static_cast<std::size_t>(row)];
    if (!slot.has_value()) {
        throw std::logic_error("Paged KV mirror execution row is not bound");
    }
    return *slot;
}

KVExecutionRowHandle MirroredKVExecutionTables::require_mirror_handle(std::int32_t row) const {
    return mirror_row(row).handle();
}

// --- MirroredStateImagePool ---------------------------------------------------------------

MirroredStateImagePool::MirroredStateImagePool(StateImageStorage& origin, StateImageStorage& mirror,
                                               MirrorTarget where)
    : origin_(origin), mirror_(mirror), where_(where) {
    require_target(where_, "StateImage mirror");
    require_matching_state_image_mirror(origin_, mirror_);
}

CudaExecutionView MirroredStateImagePool::mirror_execution() const {
    require_target(where_, "StateImage mirror replay");
    return where_.device->execution_view(where_.stream);
}

void MirroredStateImagePool::clear(std::int32_t slot, CudaExecutionView origin_execution) {
    origin_.clear(slot, origin_execution);
    mirror_.clear(slot, mirror_execution());
}

void MirroredStateImagePool::clear_all(CudaExecutionView origin_execution) {
    origin_.clear_all(origin_execution);
    mirror_.clear_all(mirror_execution());
}

void MirroredStateImagePool::copy(std::int32_t source, std::int32_t destination,
                                  CudaExecutionView origin_execution) {
    origin_.copy(source, destination, origin_execution);
    mirror_.copy(source, destination, mirror_execution());
}

void MirroredStateImagePool::copy_group(LayoutRegion group, std::int32_t source,
                                        std::int32_t destination,
                                        CudaExecutionView origin_execution) {
    origin_.copy_group(group, source, destination, origin_execution);
    mirror_.copy_group(group, source, destination, mirror_execution());
}

void MirroredStateImagePool::copy_to_host(std::int32_t source, MemorySpan destination,
                                          CudaExecutionView origin_execution) const {
    (void)source;
    (void)destination;
    (void)origin_execution;
    throw_host_transfer_unavailable("StateImage Device-to-Host copy");
}

void MirroredStateImagePool::copy_from_host(ConstMemorySpan source, std::int32_t destination,
                                            CudaExecutionView origin_execution) const {
    (void)source;
    (void)destination;
    (void)origin_execution;
    throw_host_transfer_unavailable("StateImage Host-to-Device restore");
}

} // namespace ninfer::tp2
