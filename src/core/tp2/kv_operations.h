#pragma once

// The KV funnel: the one place a Program's and a logical KV store's physical KV mutations go
// through, so a width-2 run replays them on rank 1 without editing a call site.
//
// This is a core contract, not a model fact: both families' KV owners mutate the same two core
// pools, so the funnel lives beside the mirror it drives (`core/tp2/kv_mirror.h`) rather than in a
// family's impl directory.
//
// Two halves, because the two owners own different pools: `LogicalKVPageStore` has a page pool and
// no execution tables; `KVAddressSpaceStore` has both. Each half holds either the plain pool or the
// mirrored pair, chosen at construction:
//
//   KVPageOperations(DeviceKVPagePool&)                  width 1, the plain pool
//   KVPageOperations(tp2::MirroredKVPagePool&)           width 2, constructed by the per-rank plan
//   KVTableOperations(KVExecutionTablePool&)             width 1
//   KVTableOperations(tp2::MirroredKVExecutionTables&)   width 2
//
// Width 1 forwards to the plain pool, so every call site keeps its exact behaviour. Width 2 forwards
// to the mirror, which runs the origin operation and replays it on rank 1's stream. The call sites
// are the same in both cases — that is the point of the funnel.
//
// Header-only: `logical_kv_store.h` is entirely inline and this header sits beside it, so the funnel
// adds no translation unit and no build-graph edge.
//
// The row lease is owned: `KVTableOperations::acquire` returns a `KVTableRowLease`, which drops the
// paired rank-1 row on release. A bare `KVExecutionRowLease` would release only rank 0's row and
// leak rank 1's, because its destructor calls back into the plain pool.

#include "core/device_kv_page_pool.h"
#include "core/tp2/kv_mirror.h"

#include <cstdint>
#include <span>
#include <utility>

namespace ninfer::tp2 {

class KVTableOperations;

// Owned execution-row lease. Move-only; releasing it releases rank 1's row too when it came from a
// mirrored table pool.
class KVTableRowLease {
public:
    KVTableRowLease() noexcept = default;
    ~KVTableRowLease();

    KVTableRowLease(KVTableRowLease&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), lease_(std::move(other.lease_)) {}

    KVTableRowLease& operator=(KVTableRowLease&& other) noexcept {
        if (this != &other) {
            (void)release();
            owner_ = std::exchange(other.owner_, nullptr);
            lease_ = std::move(other.lease_);
        }
        return *this;
    }

    KVTableRowLease(const KVTableRowLease&)            = delete;
    KVTableRowLease& operator=(const KVTableRowLease&) = delete;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr && lease_.valid(); }

    [[nodiscard]] KVExecutionRowHandle handle() const noexcept { return lease_.handle(); }

    [[nodiscard]] std::int32_t row_index() const noexcept { return lease_.row_index(); }

    // The underlying core lease, for consumers that take one by reference (a PagedKVCache's
    // execution view, for example). It carries rank 0's row only; release this wrapper, not that.
    [[nodiscard]] const KVExecutionRowLease& lease() const noexcept { return lease_; }

    // Releases rank 0's row and, when mirrored, rank 1's. False when the lease was already stale or
    // empty.
    bool release() noexcept;

private:
    friend class KVTableOperations;

    KVTableRowLease(KVTableOperations& owner, KVExecutionRowLease lease) noexcept
        : owner_(&owner), lease_(std::move(lease)) {}

    KVTableOperations* owner_ = nullptr;
    KVExecutionRowLease lease_;
};

// The execution-table half of the funnel: acquire, publish, publish_repeated and the geometry
// queries a caller needs before those.
class KVTableOperations {
public:
    explicit KVTableOperations(KVExecutionTablePool& tables) noexcept : plain_(&tables) {}
    explicit KVTableOperations(tp2::MirroredKVExecutionTables& tables) noexcept
        : mirrored_(&tables) {}

    [[nodiscard]] bool mirrored() const noexcept { return mirrored_ != nullptr; }

    // Rank 0's geometry; rank 1 shares it, because the mirror addresses the same rows and pages.
    [[nodiscard]] std::uint32_t logical_page_capacity() const noexcept {
        return mirrored_ != nullptr ? mirrored_->logical_page_capacity()
                                    : plain_->logical_page_capacity();
    }

    [[nodiscard]] std::int32_t row_count() const noexcept {
        return mirrored_ != nullptr ? mirrored_->row_count() : plain_->row_count();
    }

    [[nodiscard]] KVTableRowLease acquire(std::int32_t row) {
        if (mirrored_ != nullptr) {
            return KVTableRowLease(*this, mirrored_->acquire(row));
        }
        return KVTableRowLease(*this, plain_->acquire(row));
    }

    // Called by KVTableRowLease. Releasing a lease from a different funnel is a caller error, not a
    // silent cross-pool release.
    void release(KVExecutionRowLease lease) noexcept {
        if (mirrored_ != nullptr) {
            (void)mirrored_->release(std::move(lease));
        } else {
            (void)lease.release();
        }
    }

    void publish(KVExecutionRowHandle row, std::uint32_t logical_begin,
                 std::span<const DeviceKVPageHandle> pages, cudaStream_t stream = nullptr) {
        if (mirrored_ != nullptr) {
            mirrored_->publish(row, logical_begin, pages, stream);
        } else {
            plain_->publish(row, logical_begin, pages, stream);
        }
    }

    void publish_repeated(KVExecutionRowHandle row, DeviceKVPageHandle page, std::uint32_t count,
                          cudaStream_t stream = nullptr) {
        if (mirrored_ != nullptr) {
            mirrored_->publish_repeated(row, page, count, stream);
        } else {
            plain_->publish_repeated(row, page, count, stream);
        }
    }

private:
    KVExecutionTablePool* plain_               = nullptr;
    tp2::MirroredKVExecutionTables* mirrored_  = nullptr;
};

inline KVTableRowLease::~KVTableRowLease() { (void)release(); }

inline bool KVTableRowLease::release() noexcept {
    if (owner_ == nullptr) { return false; }
    KVTableOperations* owner = owner_;
    owner_                   = nullptr;
    KVExecutionRowLease lease = std::move(lease_);
    owner->release(std::move(lease));
    return true;
}

// The page-pool half of the funnel: the two physical page mutations the Program issues.
class KVPageOperations {
public:
    explicit KVPageOperations(DeviceKVPagePool& pages) noexcept : plain_(&pages) {}
    explicit KVPageOperations(tp2::MirroredKVPagePool& pages) noexcept : mirrored_(&pages) {}

    [[nodiscard]] bool mirrored() const noexcept { return mirrored_ != nullptr; }

    void copy_page(DeviceKVPageHandle source, DeviceKVPageHandle destination,
                   cudaStream_t stream = nullptr) const {
        if (mirrored_ != nullptr) {
            mirrored_->copy_page(source, destination, stream);
        } else {
            plain_->copy_page(source, destination, stream);
        }
    }

    void zero_pages(std::span<const DeviceKVPageHandle> pages,
                    cudaStream_t stream = nullptr) const {
        if (mirrored_ != nullptr) {
            mirrored_->zero_pages(pages, stream);
        } else {
            plain_->zero_pages(pages, stream);
        }
    }

private:
    DeviceKVPagePool* plain_              = nullptr;
    tp2::MirroredKVPagePool* mirrored_    = nullptr;
};

} // namespace ninfer::tp2
