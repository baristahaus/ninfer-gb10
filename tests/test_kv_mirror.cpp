// tp2 rank-1 mirrors: the geometry contract and the replay path.
//
// The host-only cases run anywhere: they cover the StateImage layout contract and the wrapper's
// refusal of an unattached target, neither of which touches CUDA. The two-device cases are skipped
// without a GPU (SKIP_RETURN_CODE 77) and are the real replay check.

#include "core/tp2/kv_mirror.h"

#include "core/buffer.h"
#include "core/device.h"
#include "core/host_kv_arena.h"
#include "core/paged_kv_cache.h"
#include "core/state_image.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int fail(const std::string& message) {
    std::cerr << message << '\n';
    return 1;
}

// A host with no driver, or one linked against the toolkit's stub, has no device to run on. The
// stub case matters: it is how a GPU-less CI builder links at all, so it must skip rather than fail.
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

template <typename Fn>
int expect_ok(Fn&& fn, const std::string& label) {
    try {
        fn();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << label << " threw: " << error.what() << '\n';
        return 1;
    }
}

// --- host-only: StateImage layout contract ------------------------------------------------

// One StateImage over pageable host memory. `extra_fields` changes the field inventory so the
// contract can be shown to reject a mirror whose slots would not line up.
struct HostImage {
    std::unique_ptr<void, void (*)(void*)> data{nullptr, &std::free};
    ninfer::StateImageLayout layout;
    std::unique_ptr<ninfer::StateImageStorage> storage;
};

HostImage make_host_image(std::int32_t slots, int extra_fields) {
    ninfer::LayoutBuilder backing;
    ninfer::StateImageBuilder builder;
    (void)builder.add(ninfer::DType::BF16, {4}, 1, "field");
    if (extra_fields > 0) { (void)builder.add(ninfer::DType::BF16, {4}, 1, "extra"); }
    HostImage image;
    image.layout = builder.finish(backing, slots);
    const std::size_t bytes = image.layout.storage.bytes;
    void* memory            = std::aligned_alloc(256, (bytes + 255) / 256 * 256);
    if (memory == nullptr) { throw std::bad_alloc(); }
    image.data.reset(memory);
    image.storage = std::make_unique<ninfer::StateImageStorage>(
        ninfer::MemorySpan{memory, bytes, ninfer::Device::cpu()}, image.layout);
    return image;
}

int exercise_state_image_contract() {
    int failures = 0;
    HostImage origin = make_host_image(2, 0);
    HostImage same   = make_host_image(2, 0);
    HostImage wider  = make_host_image(3, 0);
    HostImage extra  = make_host_image(2, 1);

    failures += expect_ok(
        [&] {
            ninfer::tp2::require_matching_state_image_mirror(*origin.storage, *same.storage);
        },
        "matching StateImage mirror");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            ninfer::tp2::require_matching_state_image_mirror(*origin.storage, *origin.storage);
        },
        "StateImage mirror of itself");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            ninfer::tp2::require_matching_state_image_mirror(*origin.storage, *wider.storage);
        },
        "StateImage mirror with another slot count");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            ninfer::tp2::require_matching_state_image_mirror(*origin.storage, *extra.storage);
        },
        "StateImage mirror with another field inventory");

    // The wrapper refuses an unattached target before it reaches any CUDA call, so a host-backed
    // pair can prove the refusal here.
    failures += expect_throws<std::invalid_argument>(
        [&] {
            ninfer::tp2::MirroredStateImagePool mirrored(*origin.storage, *same.storage,
                                                         ninfer::tp2::MirrorTarget{});
            (void)mirrored;
        },
        "StateImage mirror without a target");

    const ninfer::tp2::MirrorTarget unattached{};
    failures += (unattached.valid() ? fail("empty mirror target reported valid") : 0);
    return failures;
}

// --- two devices: the replay path ---------------------------------------------------------

struct PlannedPools {
    ninfer::DeviceKVPagePoolLayout pages;
    ninfer::KVExecutionTableLayout tables;
    std::size_t bytes = 0;
};

PlannedPools plan_pools(std::uint32_t physical_pages, std::uint32_t logical_pages,
                        std::int32_t rows, const ninfer::KVPageGeometry& geometry) {
    ninfer::LayoutBuilder builder;
    PlannedPools planned;
    planned.pages = ninfer::plan_device_kv_page_pool(
        builder, {.page_group_count = physical_pages, .geometry = geometry});
    planned.tables = ninfer::plan_kv_execution_tables(
        builder, {.logical_page_capacity = logical_pages, .table_rows = rows});
    planned.bytes = builder.finish(256);
    return planned;
}

std::vector<std::int32_t> read_indices(const ninfer::Tensor& row, std::size_t count) {
    std::vector<std::int32_t> out(count, -1);
    if (cudaMemcpy(out.data(), row.data, count * sizeof(std::int32_t), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        throw std::runtime_error("execution-row readback failed");
    }
    return out;
}

std::vector<std::byte> read_bytes(const void* device, std::size_t bytes) {
    std::vector<std::byte> out(bytes);
    if (cudaMemcpy(out.data(), device, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("plane readback failed");
    }
    return out;
}

int exercise_page_and_row_replay(ninfer::DeviceContext& rank0, ninfer::DeviceContext& rank1) {
    int failures = 0;
    const ninfer::KVPageGeometry geometry{
        .planes = {{.dtype = ninfer::DType::I8,
                    .leading_extent = 4,
                    .head_extent = 2,
                    .alignment = 256,
                    .tokens_per_row = 4}}};
    const PlannedPools plan = plan_pools(2, 4, 1, geometry);

    ninfer::DeviceBuffer origin_arena(rank0.device(), plan.bytes);
    ninfer::DeviceBuffer mirror_arena(rank1.device(), plan.bytes);
    ninfer::DeviceKVPagePool origin(origin_arena.span(), plan.pages);
    ninfer::DeviceKVPagePool mirror(mirror_arena.span(), plan.pages);
    const ninfer::tp2::MirrorTarget where{.device = &rank1,
                                          .stream = rank1.stream(),
                                          .transfer_stream = rank1.transfer_stream()};
    ninfer::tp2::MirroredKVPagePool mirrored(origin, mirror, where);

    // Both replicas start empty; two pages are materialized on rank 0 only.
    std::optional<ninfer::DeviceKVPageReservation> reservation = origin.reserve(2);
    if (!reservation.has_value()) { return fail("two-page reservation failed"); }
    std::vector<ninfer::DeviceKVPageLease> leases;
    origin.materialize(*reservation, 2, leases);
    std::vector<ninfer::DeviceKVPageHandle> handles;
    for (const auto& lease : leases) { handles.push_back(lease.handle()); }

    mirrored.zero_pages(handles, rank0.stream());
    rank0.synchronize();
    rank1.synchronize();
    const std::size_t page_bytes = static_cast<std::size_t>(origin.plane(0).nb[3]);
    const std::vector<std::byte> zeroed = read_bytes(mirror.plane(0).data, 2 * page_bytes);
    failures += (std::any_of(zeroed.begin(), zeroed.end(),
                             [](std::byte value) { return value != std::byte{0}; })
                     ? fail("mirror pages are not zero after a replayed zero")
                     : 0);

    // A pattern into rank 0's first page, then the pool's own copy to the second page: the mirror
    // must hold the same bytes after the replay.
    std::vector<std::byte> pattern(page_bytes);
    for (std::size_t index = 0; index < pattern.size(); ++index) {
        pattern[index] = static_cast<std::byte>((index * 7 + 3) & 0xff);
    }
    if (cudaMemcpy(origin.plane(0).data, pattern.data(), page_bytes, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        return fail("pattern upload failed");
    }
    mirrored.copy_page(handles[0], handles[1], rank0.stream());
    rank0.synchronize();
    rank1.synchronize();
    const std::vector<std::byte> copied =
        read_bytes(static_cast<const std::byte*>(mirror.plane(0).data) + page_bytes, page_bytes);
    failures += (copied == pattern ? 0 : fail("replayed page copy differs on rank 1"));

    // Execution-row publication replays the same physical indices into rank 1's row.
    ninfer::DeviceBuffer origin_tables_arena(rank0.device(), plan.bytes);
    ninfer::DeviceBuffer mirror_tables_arena(rank1.device(), plan.bytes);
    ninfer::KVExecutionTablePool origin_tables(origin_tables_arena.span(), plan.tables, origin);
    ninfer::KVExecutionTablePool mirror_tables(mirror_tables_arena.span(), plan.tables, mirror);
    ninfer::tp2::MirroredKVExecutionTables tables(mirrored, origin_tables, mirror_tables);

    ninfer::KVExecutionRowLease row = tables.acquire(0);
    const std::vector<ninfer::DeviceKVPageHandle> mapping{handles[1], handles[0]};
    tables.publish(row.handle(), 0, mapping, rank0.stream());
    rank0.synchronize();
    rank1.synchronize();
    const std::vector<std::int32_t> expected{1, 0};
    failures += (read_indices(origin_tables.row(row.handle()), 2) == expected
                     ? 0
                     : fail("origin execution row differs from the published mapping"));
    failures += (read_indices(mirror_tables.row(tables.mirror_row(0).handle()), 2) == expected
                     ? 0
                     : fail("mirror execution row was not replayed"));

    failures += (tables.release(std::move(row)) ? 0 : fail("mirrored row release failed"));
    failures += expect_throws<std::logic_error>(
        [&] { (void)tables.mirror_row(0); }, "released mirror row stayed readable");

    // tp 2 has no Host tier: the wrapper refuses instead of half-copying.
    failures += expect_throws<std::logic_error>(
        [&] {
            mirrored.copy_to_host(handles, ninfer::HostKVAllocationView{}, rank0.stream());
        },
        "mirrored KV Host copy");
    failures += expect_throws<std::logic_error>(
        [&] {
            mirrored.copy_from_host(ninfer::HostKVAllocationConstView{}, handles, rank0.stream());
        },
        "mirrored KV Host restore");
    return failures;
}

int exercise_state_image_replay(ninfer::DeviceContext& rank0, ninfer::DeviceContext& rank1) {
    int failures = 0;
    ninfer::LayoutBuilder backing;
    ninfer::StateImageBuilder builder;
    (void)builder.add(ninfer::DType::BF16, {4}, 1, "field");
    const ninfer::StateImageLayout layout = builder.finish(backing, 2);
    const std::size_t bytes               = backing.finish(256);

    ninfer::DeviceBuffer origin_arena(rank0.device(), bytes);
    ninfer::DeviceBuffer mirror_arena(rank1.device(), bytes);
    ninfer::StateImageStorage origin(origin_arena.span(), layout);
    ninfer::StateImageStorage mirror(mirror_arena.span(), layout);
    const ninfer::tp2::MirrorTarget where{.device = &rank1,
                                          .stream = rank1.stream(),
                                          .transfer_stream = rank1.transfer_stream()};
    ninfer::tp2::MirroredStateImagePool mirrored(origin, mirror, where);

    const std::size_t image_bytes = layout.image_stride;
    std::vector<std::byte> pattern(image_bytes);
    for (std::size_t index = 0; index < pattern.size(); ++index) {
        pattern[index] = static_cast<std::byte>(0xa5);
    }
    if (cudaMemcpy(origin.image(0).data, pattern.data(), image_bytes, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        return fail("StateImage pattern upload failed");
    }

    mirrored.copy(0, 1, rank0.execution_view());
    rank0.synchronize();
    rank1.synchronize();
    failures += (read_bytes(mirror.image(1).data, image_bytes) == pattern
                     ? 0
                     : fail("replayed StateImage copy differs on rank 1"));

    mirrored.clear(0, rank0.execution_view());
    rank0.synchronize();
    rank1.synchronize();
    const std::vector<std::byte> cleared = read_bytes(mirror.image(0).data, image_bytes);
    failures += (std::any_of(cleared.begin(), cleared.end(),
                             [](std::byte value) { return value != std::byte{0}; })
                     ? fail("mirror StateImage slot is not zero after a replayed clear")
                     : 0);

    failures += expect_throws<std::logic_error>(
        [&] { mirrored.copy_to_host(0, ninfer::MemorySpan{}, rank0.execution_view()); },
        "mirrored StateImage Host copy");
    return failures;
}

} // namespace

int main() {
    int failures = exercise_state_image_contract();

    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device; the replay cases did not run\n";
        return failures == 0 ? 77 : 1;
    }
    if (count_err != cudaSuccess) {
        return fail(std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(count_err));
    }
    if (count < 2) {
        std::cout << "SKIP: tp2 replay needs two devices, found " << count << '\n';
        return failures == 0 ? 77 : 1;
    }

    try {
        ninfer::DeviceContext rank0(0);
        ninfer::DeviceContext rank1(1);
        failures += exercise_page_and_row_replay(rank0, rank1);
        failures += exercise_state_image_replay(rank0, rank1);
    } catch (const std::exception& error) {
        return fail(std::string("tp2 mirror replay threw: ") + error.what());
    }

    if (failures != 0) { std::cerr << failures << " tp2 mirror checks failed\n"; }
    return failures == 0 ? 0 : 1;
}
