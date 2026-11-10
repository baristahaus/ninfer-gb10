// The KV funnel: dispatch at width 1, and the two Program capture sites that now go through it.
//
// The compile-time half is the point of the funnel for B3: the handle must be constructible from the
// plain pool *or* the mirrored pair, so the per-rank plan only has to pass the mirrored one in. The
// host-only half proves the width-1 handle forwards to the plain pool without a device: a page
// pool over pageable host memory rejects its own bad arguments before any CUDA call, and those
// rejections are what the funnel surfaces. The two-device-free runtime half needs a GPU because a
// KVExecutionTablePool allocates pinned host memory in its constructor.

#include "core/tp2/kv_operations.h"
#include "models/qwen3_8_flash_next/impl/runtime/logical_kv_store.h"

#include "core/buffer.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace detail = ninfer::models::qwen3_8_flash_next::detail;

namespace {

int fail(const std::string& message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver ||
           err == cudaErrorStubLibrary;
}

template <typename Exception, typename Fn>
int expect_throws(Fn&& fn, const std::string& label) {
    try {
        fn();
    } catch (const Exception&) { return 0; }
    std::cerr << label << " did not throw the expected exception\n";
    return 1;
}

// B3 supplies the mirrored pair; the funnel must accept either form without a call site changing.
static_assert(std::is_constructible_v<ninfer::tp2::KVPageOperations, ninfer::DeviceKVPagePool&>);
static_assert(std::is_constructible_v<ninfer::tp2::KVPageOperations, ninfer::tp2::MirroredKVPagePool&>);
static_assert(std::is_constructible_v<ninfer::tp2::KVTableOperations, ninfer::KVExecutionTablePool&>);
static_assert(
    std::is_constructible_v<ninfer::tp2::KVTableOperations, ninfer::tp2::MirroredKVExecutionTables&>);
static_assert(std::is_constructible_v<ninfer::tp2::KVTableRowLease>);
static_assert(std::is_move_constructible_v<ninfer::tp2::KVTableRowLease>);
static_assert(!std::is_copy_constructible_v<ninfer::tp2::KVTableRowLease>);

struct PlannedPools {
    ninfer::DeviceKVPagePoolLayout pages;
    ninfer::KVExecutionTableLayout tables;
    std::size_t bytes = 0;
};

PlannedPools plan_pools(std::uint32_t physical_pages, std::uint32_t logical_pages,
                        std::int32_t rows) {
    const ninfer::KVPageGeometry geometry{
        .planes = {{.dtype          = ninfer::DType::I8,
                    .leading_extent = 4,
                    .head_extent    = 2,
                    .alignment      = 256,
                    .tokens_per_row = 4}}};
    ninfer::LayoutBuilder builder;
    PlannedPools planned;
    planned.pages = ninfer::plan_device_kv_page_pool(
        builder, {.page_group_count = physical_pages, .geometry = geometry});
    planned.tables = ninfer::plan_kv_execution_tables(
        builder, {.logical_page_capacity = logical_pages, .table_rows = rows});
    planned.bytes = builder.finish(256);
    return planned;
}

// --- host-only: the width-1 page handle forwards to the plain pool ---------------------------

int exercise_width_one_page_dispatch() {
    int failures = 0;
    const PlannedPools plan = plan_pools(2, 2, 1);
    // A page pool binds tensors over its backing; it performs no CUDA call, so pageable host memory
    // is enough to prove dispatch through the pool's own validation.
    std::unique_ptr<void, void (*)(void*)> backing(
        std::aligned_alloc(256, (plan.bytes + 255) / 256 * 256), &std::free);
    if (backing == nullptr) { return fail("host backing allocation failed"); }
    ninfer::DeviceKVPagePool pool(
        ninfer::MemorySpan{backing.get(), plan.bytes, ninfer::Device::cpu()}, plan.pages);

    ninfer::tp2::KVPageOperations operations(pool);
    failures += (operations.mirrored() ? fail("width-1 page handle reports mirrored") : 0);

    std::optional<ninfer::DeviceKVPageReservation> reservation = pool.reserve(2);
    if (!reservation.has_value()) { return fail("two-page reservation failed"); }
    std::vector<ninfer::DeviceKVPageLease> leases;
    leases.reserve(2); // materialize refuses to grow the destination past its capacity
    pool.materialize(*reservation, 2, leases);
    const std::vector<ninfer::DeviceKVPageHandle> pages{leases[0].handle(), leases[1].handle()};

    // The pool's own duplicate check runs before any CUDA call, so a throw here proves the handle
    // forwarded to the plain pool rather than doing nothing.
    failures += expect_throws<std::invalid_argument>(
        [&] {
            const std::array<ninfer::DeviceKVPageHandle, 2> duplicated{pages[0], pages[0]};
            operations.zero_pages(duplicated);
        },
        "funnel zero of a duplicated page");
    // A self-copy is a no-op in the plain pool: no CUDA call, no throw.
    operations.copy_page(pages[0], pages[0]);
    // A handle of another pool is rejected by the plain pool's handle validation.
    const PlannedPools other_plan = plan_pools(2, 2, 1);
    std::unique_ptr<void, void (*)(void*)> other_backing(
        std::aligned_alloc(256, (other_plan.bytes + 255) / 256 * 256), &std::free);
    if (other_backing == nullptr) { return fail("second host backing allocation failed"); }
    ninfer::DeviceKVPagePool other_pool(
        ninfer::MemorySpan{other_backing.get(), other_plan.bytes, ninfer::Device::cpu()},
        other_plan.pages);
    std::optional<ninfer::DeviceKVPageReservation> other_reservation = other_pool.reserve(1);
    if (!other_reservation.has_value()) { return fail("foreign page reservation failed"); }
    std::vector<ninfer::DeviceKVPageLease> other_leases;
    other_leases.reserve(1);
    other_pool.materialize(*other_reservation, 1, other_leases);
    failures += expect_throws<std::invalid_argument>(
        [&] { operations.copy_page(other_leases[0].handle(), pages[0]); },
        "funnel copy of a handle from another pool");
    return failures;
}

// --- GPU: the two capture sites, through the store funnels ----------------------------------

int exercise_capture_site_dispatch(ninfer::DeviceContext& device) {
    int failures = 0;
    const PlannedPools plan = plan_pools(2, 4, 1);
    ninfer::DeviceBuffer arena(device.device(), plan.bytes);
    ninfer::DeviceKVPagePool pool(arena.span(), plan.pages);
    ninfer::KVExecutionTablePool tables(arena.span(), plan.tables, pool);

    detail::LogicalKVPageStore pages(pool, 4);
    detail::KVAddressSpaceStore addresses(pages, tables, 2, 4);
    failures += (addresses.logical_page_capacity() == 4 ? 0
                                                        : fail("funnel geometry is not the pool's"));

    std::optional<detail::KVAddressSpaceHandle> allocation = addresses.create_active(1, 0);
    if (!allocation.has_value()) { return fail("capture row activation failed"); }
    addresses.materialize_to_tokens(*allocation, 1, device.stream());

    // The site `program_impl.h:11413` shape: repeat the row's private page across its table.
    const ninfer::DeviceKVPageHandle physical = addresses.physical_page(*allocation, 0);
    addresses.publish_repeated(addresses.execution_row(*allocation).handle(), physical,
                               addresses.logical_page_capacity(), device.stream());
    device.synchronize();
    const ninfer::Tensor row = tables.row(addresses.execution_row(*allocation).handle());
    std::vector<std::int32_t> mapping(4, -1);
    if (cudaMemcpy(mapping.data(), row.data, mapping.size() * sizeof(std::int32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return fail("execution-row readback failed");
    }
    const std::int32_t expected = pool.physical_page(physical);
    failures += (std::all_of(mapping.begin(), mapping.end(),
                             [expected](std::int32_t value) { return value == expected; })
                     ? 0
                     : fail("funnelled publish_repeated did not repeat the page across the row"));

    // The site `program_impl.h:11455` shape: zero the row's pages through the address store.
    const std::size_t page_bytes = static_cast<std::size_t>(pool.plane(0).nb[3]);
    if (cudaMemset(static_cast<std::byte*>(pool.plane(0).data) +
                       static_cast<std::ptrdiff_t>(expected) * page_bytes,
                   0x5a, page_bytes) != cudaSuccess) {
        return fail("page pattern upload failed");
    }
    const std::array<ninfer::DeviceKVPageHandle, 1> zero_target{physical};
    addresses.zero_pages(zero_target, device.stream());
    device.synchronize();
    std::vector<std::byte> zeroed(page_bytes, std::byte{0xee});
    if (cudaMemcpy(zeroed.data(),
                   static_cast<const std::byte*>(pool.plane(0).data) +
                       static_cast<std::ptrdiff_t>(expected) * page_bytes,
                   page_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return fail("page readback failed");
    }
    failures += (std::all_of(zeroed.begin(), zeroed.end(),
                             [](std::byte value) { return value == std::byte{0}; })
                     ? 0
                     : fail("funnelled zero_pages did not clear the page"));

    // The lease is owned by the funnel: releasing it releases the execution row, so the same row
    // can be activated again.
    if (!addresses.release(*allocation)) { return fail("address release failed"); }
    std::optional<detail::KVAddressSpaceHandle> again = addresses.create_active(1, 0);
    failures += (again.has_value() ? 0 : fail("funnelled lease did not release the execution row"));
    if (again.has_value()) { (void)addresses.release(*again); }
    return failures;
}

} // namespace

int main() {
    int failures = exercise_width_one_page_dispatch();

    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device; the runtime dispatch cases did not run\n";
        return failures == 0 ? 77 : 1;
    }
    if (count_err != cudaSuccess) {
        return fail(std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(count_err));
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return failures == 0 ? 77 : 1;
    }

    try {
        ninfer::DeviceContext device(0);
        failures += exercise_capture_site_dispatch(device);
    } catch (const std::exception& error) {
        return fail(std::string("KV funnel dispatch threw: ") + error.what());
    }

    if (failures != 0) { std::cerr << failures << " KV funnel checks failed\n"; }
    return failures == 0 ? 0 : 1;
}
