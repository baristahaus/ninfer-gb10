// Flash-Next StateImage mirror: the index-identical contract and the replay path.
//
// The host-only cases run anywhere: they build the family pool over pageable host memory (its
// constructor performs no CUDA call) and check the contract that decides whether a mirror may
// attach. The two-device cases are the replay check and are skipped without two GPUs (77).
//
// Contract under test (review/phase3/state-image.md §3): a mirror replays the same slot indices on
// rank 1's own shard, so the SOURCE content must exist on both ranks before a replayed copy. The
// replay cases therefore seed a different pattern per rank and assert each rank's destination holds
// its own pattern — a cross-rank copy fails instead of passing.

#include "models/qwen3_8_flash_next/impl/state/state_image_mirror.h"

#include "core/buffer.h"
#include "core/device.h"
#include "core/layout.h"
#include "ninfer/models/qwen3_8_flash_next/state_image.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::models::qwen3_8_flash_next::DFlashLocalStateSpec;
using ninfer::models::qwen3_8_flash_next::StateImageDeviceLayout;
using ninfer::models::qwen3_8_flash_next::StateImageDevicePool;
using ninfer::models::qwen3_8_flash_next::StateImageSpec;
using ninfer::models::qwen3_8_flash_next::plan_state_image_device_pool;

int fail(const std::string& message) {
    std::cerr << message << '\n';
    return 1;
}

// A host with no driver, or one linked against the toolkit's stub, has no device to run on. The
// stub case matters: it is how a GPU-less builder links at all, so it must skip rather than fail.
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

StateImageSpec make_spec(std::uint32_t layers, std::int32_t slots, std::int32_t value_heads,
                         std::int32_t hidden, std::uint32_t dflash_layers, bool mtp_hidden) {
    StateImageSpec spec;
    spec.linear.layers        = layers;
    spec.linear.conv_channels = 8;
    spec.linear.conv_width    = 4;
    spec.linear.value_heads   = value_heads;
    spec.linear.value_head_dim = 4;
    spec.linear.key_head_dim  = 2;
    spec.linear.slot_count    = slots;
    spec.linear.conv_dtype    = ninfer::DType::BF16;
    spec.hidden               = hidden;
    if (mtp_hidden) { spec.mtp_hidden = 8; }
    if (dflash_layers > 0) {
        spec.dflash_local = DFlashLocalStateSpec{.layers   = dflash_layers,
                                                 .capacity = static_cast<std::uint32_t>(slots),
                                                 .kv_heads = 1,
                                                 .head_dim = 4};
    }
    return spec;
}

// One StateImage device pool over pageable host memory. The pool's constructor binds tensors and
// validates the layout only, so this exercises the contract without CUDA.
struct HostPool {
    std::unique_ptr<void, void (*)(void*)> data{nullptr, &std::free};
    StateImageDeviceLayout layout;
    std::unique_ptr<StateImageDevicePool> pool;
};

HostPool make_host_pool(const StateImageSpec& spec) {
    ninfer::LayoutBuilder builder;
    HostPool out;
    out.layout              = plan_state_image_device_pool(builder, spec);
    const std::size_t bytes = builder.finish(256);
    void* memory            = std::aligned_alloc(256, (bytes + 255) / 256 * 256);
    if (memory == nullptr) { throw std::bad_alloc(); }
    out.data.reset(memory);
    out.pool = std::make_unique<StateImageDevicePool>(
        ninfer::MemorySpan{memory, bytes, ninfer::Device::cpu()}, out.layout);
    return out;
}

int exercise_contract() {
    int failures = 0;
    const StateImageSpec base = make_spec(2, 2, 2, 8, 1, true);
    HostPool origin           = make_host_pool(base);
    HostPool same             = make_host_pool(base);
    HostPool wider            = make_host_pool(make_spec(2, 3, 2, 8, 1, true));
    HostPool no_mtp           = make_host_pool(make_spec(2, 2, 2, 8, 1, false));
    HostPool more_dflash      = make_host_pool(make_spec(2, 2, 2, 8, 2, true));
    // A rank-1 shard: same slots and components, narrower GDN and hidden widths.
    HostPool shard            = make_host_pool(make_spec(2, 2, 1, 4, 1, true));

    failures += expect_ok(
        [&] {
            ninfer::models::qwen3_8_flash_next::require_matching_state_image_device_pool(
                *origin.pool, *same.pool);
        },
        "matching StateImage pool mirror");
    failures += expect_ok(
        [&] {
            ninfer::models::qwen3_8_flash_next::require_matching_state_image_device_pool(
                *origin.pool, *shard.pool);
        },
        "index-identical mirror with narrower shard widths");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            ninfer::models::qwen3_8_flash_next::require_matching_state_image_device_pool(
                *origin.pool, *origin.pool);
        },
        "StateImage mirror of itself");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            ninfer::models::qwen3_8_flash_next::require_matching_state_image_device_pool(
                *origin.pool, *wider.pool);
        },
        "StateImage mirror with another slot count");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            ninfer::models::qwen3_8_flash_next::require_matching_state_image_device_pool(
                *origin.pool, *no_mtp.pool);
        },
        "StateImage mirror without the MTP-hidden component");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            ninfer::models::qwen3_8_flash_next::require_matching_state_image_device_pool(
                *origin.pool, *more_dflash.pool);
        },
        "StateImage mirror with another DFlash layer count");

    // The wrapper refuses an unattached target before any CUDA call.
    failures += expect_throws<std::invalid_argument>(
        [&] {
            ninfer::models::qwen3_8_flash_next::MirroredStateImageDevicePool mirrored(
                *origin.pool, *same.pool, ninfer::tp2::MirrorTarget{});
            (void)mirrored;
        },
        "StateImage mirror without a target");
    return failures;
}

// --- two devices: the replay path ---------------------------------------------------------

struct DevicePool {
    std::unique_ptr<ninfer::DeviceBuffer> backing;
    std::unique_ptr<StateImageDevicePool> pool;
};

DevicePool make_device_pool(ninfer::DeviceContext& context, const StateImageDeviceLayout& layout,
                            std::size_t bytes) {
    DevicePool out;
    out.backing = std::make_unique<ninfer::DeviceBuffer>(context.device(), bytes);
    out.pool    = std::make_unique<StateImageDevicePool>(out.backing->span(), layout);
    return out;
}

void fill_pattern(std::vector<std::byte>& bytes, unsigned char seed) {
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::byte>((seed + index * 7U) & 0xffU);
    }
}

bool same_bytes(const std::vector<std::byte>& left, const std::vector<std::byte>& right) {
    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin());
}

bool range_is_zero(const std::vector<std::byte>& bytes, std::size_t offset, std::size_t count) {
    if (offset > bytes.size() || count > bytes.size() - offset) { return false; }
    return std::all_of(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                       bytes.begin() + static_cast<std::ptrdiff_t>(offset + count),
                       [](std::byte value) { return value == std::byte{0}; });
}

int exercise_replay(ninfer::DeviceContext& rank0, ninfer::DeviceContext& rank1) {
    int failures = 0;
    const StateImageSpec spec = make_spec(2, 2, 2, 8, 1, true);
    ninfer::LayoutBuilder builder;
    const StateImageDeviceLayout layout = plan_state_image_device_pool(builder, spec);
    const std::size_t bytes             = builder.finish(256);

    DevicePool origin = make_device_pool(rank0, layout, bytes);
    DevicePool mirror = make_device_pool(rank1, layout, bytes);
    const ninfer::tp2::MirrorTarget where{.device          = &rank1,
                                          .stream          = rank1.stream(),
                                          .transfer_stream = rank1.transfer_stream()};
    ninfer::models::qwen3_8_flash_next::MirroredStateImageDevicePool mirrored(*origin.pool,
                                                                              *mirror.pool, where);

    const std::size_t image_bytes = layout.host.image_bytes;
    ninfer::PinnedHostBuffer image_rank0(image_bytes);
    ninfer::PinnedHostBuffer image_rank1(image_bytes);
    ninfer::PinnedHostBuffer readback_rank0(image_bytes);
    ninfer::PinnedHostBuffer readback_rank1(image_bytes);

    std::vector<std::byte> pattern_rank0(image_bytes);
    std::vector<std::byte> pattern_rank1(image_bytes);
    fill_pattern(pattern_rank0, 0x11);
    fill_pattern(pattern_rank1, 0x93);
    std::memcpy(image_rank0.data(), pattern_rank0.data(), image_bytes);
    std::memcpy(image_rank1.data(), pattern_rank1.data(), image_bytes);

    // The contract: each rank's source slot holds its OWN content before the replayed copy.
    origin.pool->copy_from_host({static_cast<const std::byte*>(image_rank0.data()), &layout.host}, 0,
                                rank0.stream());
    mirror.pool->copy_from_host({static_cast<const std::byte*>(image_rank1.data()), &layout.host}, 0,
                                rank1.stream());
    rank0.synchronize();
    rank1.synchronize();

    mirrored.copy_slot(0, 1, rank0.stream());
    rank0.synchronize();
    rank1.synchronize();

    // Read back with the same prefill so alignment gaps compare equal and only payload bytes differ.
    std::memcpy(readback_rank0.data(), pattern_rank0.data(), image_bytes);
    std::memcpy(readback_rank1.data(), pattern_rank1.data(), image_bytes);
    origin.pool->copy_to_host(1, {static_cast<std::byte*>(readback_rank0.data()), &layout.host},
                              rank0.stream());
    mirror.pool->copy_to_host(1, {static_cast<std::byte*>(readback_rank1.data()), &layout.host},
                              rank1.stream());
    rank0.synchronize();
    rank1.synchronize();

    std::vector<std::byte> destination_rank0(image_bytes);
    std::vector<std::byte> destination_rank1(image_bytes);
    std::memcpy(destination_rank0.data(), readback_rank0.data(), image_bytes);
    std::memcpy(destination_rank1.data(), readback_rank1.data(), image_bytes);

    failures += (same_bytes(destination_rank0, pattern_rank0)
                     ? 0
                     : fail("replayed StateImage copy differs on rank 0"));
    failures += (same_bytes(destination_rank1, pattern_rank1)
                     ? 0
                     : fail("replayed StateImage copy differs on rank 1"));
    // A cross-rank copy would make the two destinations equal; the contract forbids it.
    failures += (same_bytes(destination_rank0, destination_rank1)
                     ? fail("replayed StateImage copy crossed ranks")
                     : 0);

    // DFlash-local copy replays its own component only: compare that region between the slots.
    mirrored.zero_all(rank0.stream());
    rank0.synchronize();
    rank1.synchronize();
    origin.pool->copy_from_host({static_cast<const std::byte*>(image_rank0.data()), &layout.host}, 0,
                                rank0.stream());
    mirror.pool->copy_from_host({static_cast<const std::byte*>(image_rank1.data()), &layout.host}, 0,
                                rank1.stream());
    mirrored.copy_dflash_local(0, 1, rank0.stream());
    rank0.synchronize();
    rank1.synchronize();

    std::vector<std::byte> rank0_slot0(image_bytes);
    std::vector<std::byte> rank0_slot1(image_bytes);
    std::vector<std::byte> rank1_slot0(image_bytes);
    std::vector<std::byte> rank1_slot1(image_bytes);
    const auto read_slot = [&](StateImageDevicePool& pool, ninfer::DeviceContext& context,
                               std::int32_t slot, ninfer::PinnedHostBuffer& buffer,
                               std::vector<std::byte>& out) {
        std::memset(buffer.data(), 0xee, image_bytes);
        pool.copy_to_host(slot, {static_cast<std::byte*>(buffer.data()), &layout.host},
                          context.stream());
        context.synchronize();
        std::memcpy(out.data(), buffer.data(), image_bytes);
    };
    read_slot(*origin.pool, rank0, 0, readback_rank0, rank0_slot0);
    read_slot(*origin.pool, rank0, 1, readback_rank0, rank0_slot1);
    read_slot(*mirror.pool, rank1, 0, readback_rank1, rank1_slot0);
    read_slot(*mirror.pool, rank1, 1, readback_rank1, rank1_slot1);

    const auto same_region = [](const std::vector<std::byte>& left,
                                const std::vector<std::byte>& right, std::size_t offset,
                                std::size_t count) {
        return std::equal(left.begin() + static_cast<std::ptrdiff_t>(offset),
                          left.begin() + static_cast<std::ptrdiff_t>(offset + count),
                          right.begin() + static_cast<std::ptrdiff_t>(offset));
    };
    for (const auto* region : {&layout.host.dflash_local_k, &layout.host.dflash_local_v}) {
        if (region == nullptr || !region->has_value()) { continue; }
        const std::size_t offset = (*region)->offset;
        const std::size_t count  = (*region)->bytes;
        failures += (same_region(rank0_slot0, rank0_slot1, offset, count)
                         ? 0
                         : fail("replayed DFlash-local copy differs on rank 0"));
        failures += (same_region(rank1_slot0, rank1_slot1, offset, count)
                         ? 0
                         : fail("replayed DFlash-local copy differs on rank 1"));
    }

    // A replayed zero clears the continuation hidden on both ranks.
    mirrored.zero_slot(1, rank0.stream());
    rank0.synchronize();
    rank1.synchronize();
    for (ninfer::DeviceContext* context : {&rank0, &rank1}) {
        StateImageDevicePool& pool = context == &rank0 ? *origin.pool : *mirror.pool;
        auto* base = static_cast<std::byte*>(context == &rank0 ? readback_rank0.data()
                                                              : readback_rank1.data());
        std::memset(base, 0xff, image_bytes);
        pool.copy_to_host(1, {base, &layout.host}, context->stream());
        context->synchronize();
        std::vector<std::byte> cleared(image_bytes);
        std::memcpy(cleared.data(), base, image_bytes);
        failures += (range_is_zero(cleared, layout.host.continuation_hidden.offset,
                                   layout.host.continuation_hidden.bytes)
                         ? 0
                         : fail("replayed StateImage zero did not clear the hidden state"));
    }

    // tp 2 has no Host state tier.
    failures += expect_throws<std::logic_error>(
        [&] {
            mirrored.copy_to_host(0, {static_cast<std::byte*>(readback_rank0.data()), &layout.host},
                                  rank0.stream());
        },
        "mirrored StateImage Host copy");
    return failures;
}

} // namespace

int main() {
    int failures = exercise_contract();

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
        failures += exercise_replay(rank0, rank1);
    } catch (const std::exception& error) {
        return fail(std::string("StateImage mirror replay threw: ") + error.what());
    }

    if (failures != 0) { std::cerr << failures << " StateImage mirror checks failed\n"; }
    return failures == 0 ? 0 : 1;
}
