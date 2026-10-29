#include "core/device.h"
#include "models/common/context/directory/state.h"

#include <array>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace context = ninfer::models::context;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

template <class Exception, class Function>
void expect_throw(Function&& function, std::string_view message) {
    try {
        function();
    } catch (const Exception&) { return; }
    expect(false, message);
}

std::vector<std::byte> pattern(std::size_t bytes, unsigned seed) {
    std::vector<std::byte> result(bytes);
    for (std::size_t index = 0; index < bytes; ++index) {
        result[index] = static_cast<std::byte>((seed * 131U + static_cast<unsigned>(index)) & 0xffU);
    }
    return result;
}

void run() {
    const std::size_t image_bytes = 4096;
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "ninfer_state_spill_test.bin";
    std::error_code ignored;
    std::filesystem::remove(path, ignored);

    context::HostAllocator host(1, image_bytes * 4, image_bytes);
    context::StateDirectory states({16, 4, image_bytes}, &host);
    ninfer::SpillKVStore store(path, image_bytes * 8);

    const auto image = states.reserve_destination();
    expect(image.has_value(), "spill test image allocation failed");
    states.complete_initialization(*image);
    states.freeze(*image);
    expect(states.device_resident(*image), "fresh image lost its Device replica");

    // Host replica through the ordinary transfer path; the copy itself is the caller's concern.
    auto transfer = states.reserve_device_to_host(*image);
    expect(transfer.has_value(), "device-to-host reservation failed");
    transfer->publish(true);
    expect(states.host_resident(*image), "published transfer did not make the image Host resident");

    // The last full copy is host-only once the Device replica is dropped.
    expect(states.drop_device_replica(*image), "device replica drop failed");
    expect(!states.device_resident(*image) && states.host_resident(*image),
           "image is not host-only after the device drop");

    // Guards first: a device-resident image cannot spill, and a second spill is rejected.
    const auto second = states.reserve_destination();
    expect(second.has_value(), "second image allocation failed");
    states.complete_initialization(*second);
    states.freeze(*second);
    expect(!states.spill_host_replica(*second, {}),
           "device-resident image spilled without dropping its device replica");

    // The caller copies the host bytes into the store, then records the residency.
    const auto bytes = pattern(image_bytes, 7U);
    const auto extent = store.write(bytes.data(), bytes.size());
    const auto spilled_extent = extent;
    expect(states.spill_host_replica(*image, extent), "host-only image refused to spill");
    expect(!states.host_resident(*image) && states.spill_resident(*image) &&
               states.spill_extent(*image) == extent,
           "spill residency was not recorded");
    expect(!states.spill_host_replica(*image, extent), "double spill was accepted");
    expect(store.reserved_bytes() == image_bytes, "store accounting lost the spilled image");

    // Page-in: reserve, copy back, publish. The read-back must be bit-identical.
    auto reservation = states.reserve_spill_page_in(*image);
    expect(reservation.has_value(), "page-in reservation failed");
    std::vector<std::byte> back(image_bytes);
    store.read(extent, back.data(), back.size());
    expect(back == bytes, "spill round-trip is not bit-identical");
    expect(states.publish_spill_page_in(*image, std::move(*reservation)),
           "page-in publish failed");
    expect(states.host_resident(*image) && !states.spill_resident(*image) &&
               !states.spill_extent(*image).has_value(),
           "page-in did not restore host residency");
    store.release(spilled_extent);
    expect(!states.reserve_spill_page_in(*image).has_value(),
           "page-in was reserved without a spill extent");

    // Abort leaves the image unpinned and the reservation reusable.
    const auto re_spill_extent = store.write(bytes.data(), bytes.size());
    expect(states.spill_host_replica(*image, re_spill_extent),
           "re-spill after page-in failed");
    auto aborted = states.reserve_spill_page_in(*image);
    expect(aborted.has_value(), "second page-in reservation failed");
    states.abort_spill_page_in(*image, std::move(*aborted));
    expect(states.spill_resident(*image), "abort changed the residency");
    auto retried = states.reserve_spill_page_in(*image);
    expect(retried.has_value(), "reservation after abort failed");
    expect(states.publish_spill_page_in(*image, std::move(*retried)), "retry publish failed");

    // The owner releases the store extent before dropping the image; the directory record
    // clears with the image. After a published page-in the directory no longer holds the
    // extent, so the owner releases the extent it captured at spill time.
    store.release(re_spill_extent);
    expect(states.release(*image), "image release failed");
    expect(!states.spill_extent(*image).has_value(), "released image kept a spill record");

    store.release(states.spill_extent(*second).value_or(ninfer::SpillKVStore::Extent{}));
    expect(store.reserved_bytes() == 0, "store did not empty after the owner releases");

    std::filesystem::remove(path, ignored);
}

} // namespace

int main() {
    run();
    if (failures != 0) { std::cerr << "state spill failures: " << failures << '\n'; }
    return failures == 0 ? 0 : 1;
}
