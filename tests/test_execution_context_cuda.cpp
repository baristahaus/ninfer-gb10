// GPU contract tests for ExecutionContext: one device is the ordinary engine, two devices form a
// group, and the group's invariants hold. The width-2 cases are skipped when the machine has one
// device, so the test is useful on any box.

#include "core/device.h"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::ExecutionContext;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int device_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) { return 0; }
    return count;
}

int test_single_device() {
    ExecutionContext context(std::vector<int>{0});
    int failures = 0;
    failures += check(context.width() == 1, "a one-device context has width 1");
    failures += check(!context.parallel(), "a one-device context is not parallel");
    failures += check(context.primary().device().index() == 0, "rank 0 is the requested device");
    failures += check(context.device(0).device().index() == 0, "device(0) is the requested device");
    failures += check(context.compute_capability() == context.primary().compute_capability(),
                      "the group capability is the primary's");
    failures += check(context.primary().stream() != nullptr, "the primary owns a compute stream");
    return failures;
}

int test_two_devices() {
    ExecutionContext context(std::vector<int>{0, 1});
    int failures = 0;
    failures += check(context.width() == 2, "a two-device context has width 2");
    failures += check(context.parallel(), "a two-device context is parallel");
    failures += check(context.primary().device().index() == 0, "rank 0 is device 0");
    failures += check(context.device(1).device().index() == 1, "rank 1 is device 1");
    failures += check(context.primary().stream() != context.device(1).stream(),
                      "the two ranks own different streams");
    // The two 5070 Tis share sm_120, and the constructor would have thrown otherwise; assert the
    // shared capability explicitly so a future mixed pair fails here with a clear message.
    failures += check(context.device(1).compute_capability() == context.compute_capability(),
                      "both ranks share one compute capability");
    // Rank order is the caller's, not sorted: a group may name its primary anywhere.
    ExecutionContext reversed(std::vector<int>{1, 0});
    failures += check(reversed.primary().device().index() == 1,
                      "the primary is the first id the caller names");
    return failures;
}

int test_rejections() {
    int failures = 0;
    const auto rejects = [&failures](const std::vector<int>& ids, const std::string& what) {
        try {
            ExecutionContext context(ids);
            failures += check(false, what + " was accepted");
        } catch (const std::invalid_argument&) {
            // expected
        } catch (...) {
            failures += check(false, what + " threw the wrong exception type");
        }
    };
    rejects({}, "an empty device list");
    rejects({0, 0}, "a repeated device");
    rejects({0, -1}, "a negative device id");

    // A rank outside the group is rejected by the accessor, not silently clamped.
    ExecutionContext context(std::vector<int>{0});
    bool out_of_range = false;
    try {
        (void)context.device(1);
    } catch (const std::out_of_range&) {
        out_of_range = true;
    }
    failures += check(out_of_range, "device(rank) rejects a rank outside the group");
    return failures;
}

} // namespace

int main() {
    const int devices = device_count();
    if (devices < 1) {
        // Nothing in this test can run without a device.
        std::cerr << "SKIP: no CUDA device is available\n";
        return 77;
    }
    int failures = 0;
    failures += test_single_device();
    if (devices >= 2) {
        failures += test_two_devices();
    } else {
        std::cerr << "SKIP: the two-device cases did not run; one device is present\n";
    }
    failures += test_rejections();
    if (failures == 0) { std::cout << "ok (" << devices << " device(s))\n"; }
    // The two-device cases did not run. Report a skip, as the other two-device tests in this tree
    // do, so the suite's coverage gap is visible instead of being counted as a pass.
    if (failures == 0 && devices < 2) { return 77; }
    return failures == 0 ? 0 : 1;
}
