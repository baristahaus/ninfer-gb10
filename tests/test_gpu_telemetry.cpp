// serve/gpu_telemetry: the NVML clocks-event reason mapping, and the contract that a host without
// libnvidia-ml.so.1 reports no samples instead of failing. The real-GPU sampling path (clocks,
// power, energy delta, temperature) cannot be exercised on a host with no driver; this test asserts
// the documented fallback and runs the sampling path when NVML happens to be available.
#include "serve/gpu_telemetry.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::serve::clock_event_reason_names;
using ninfer::serve::GpuTelemetry;
using ninfer::serve::GpuTelemetryDevice;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << "\n";
    return 1;
}

bool names_are(std::uint64_t reasons, const std::vector<std::string>& expected) {
    return clock_event_reason_names(reasons) == expected;
}

} // namespace

int main() {
    int failures = 0;

    // Reason bits are named lowest first; an undefined bit falls back to its hexadecimal mask.
    failures += check(names_are(0, {}), "no reason bits did not map to an empty list");
    failures += check(names_are(0x1, {"gpu_idle"}), "the gpu_idle bit was not named");
    failures += check(names_are(0x4 | 0x20, {"sw_power_cap", "sw_thermal_slowdown"}),
                      "two reason bits were not named in bit order");
    failures += check(names_are(0x100, {"display_clock_setting"}),
                      "the highest defined reason bit was not named");
    failures += check(names_are(std::uint64_t{1} << 40, {"0x10000000000"}),
                      "an undefined reason bit was not named by its mask");
    failures += check(names_are(0x1 | (std::uint64_t{1} << 40), {"gpu_idle", "0x10000000000"}),
                      "a defined and an undefined bit were not reported together");

    // A bogus UUID never resolves to an NVML handle, so on a GPU-less host the library is absent
    // and the sampler reports nothing. On a host with NVML the same object still reports one
    // sample per configured device, with the unreadable fields empty.
    GpuTelemetry telemetry({GpuTelemetryDevice{.device = 0, .uuid = "GPU-not-a-real-uuid"}});
    const auto samples = telemetry.sample();
    if (!telemetry.available()) {
        failures += check(!samples.has_value(), "an unavailable NVML reported samples");
    } else {
        failures += check(samples.has_value() && samples->size() == 1,
                          "an available NVML did not report one sample per device");
        if (samples && samples->size() == 1) {
            failures += check(samples->front().device == 0, "the sample lost its device id");
            // The first call only sets the energy baseline, whatever the handle resolved to.
            failures += check(!samples->front().energy_joules.has_value(),
                              "the first sample reported an energy delta");
        }
    }

    // An empty device list is legal and must not crash either way.
    GpuTelemetry none({});
    const auto empty = none.sample();
    failures += check(!none.available() || (empty && empty->empty()),
                      "an empty device list did not report an empty sample list");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
