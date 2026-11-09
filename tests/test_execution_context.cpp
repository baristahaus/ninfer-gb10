// Host-only contract tests for the execution-group device list. The list is validated before any
// device is opened, so this runs without a GPU; constructing an ExecutionContext does open the
// devices and is covered by the integration tests instead.

#include "core/device.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::validate_execution_device_ids;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

template <typename Exception>
bool throws(const std::vector<int>& ids) {
    try {
        validate_execution_device_ids(ids);
    } catch (const Exception&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

int test_accepted_lists() {
    int failures = 0;
    for (const std::vector<int> ids : {std::vector<int>{0}, std::vector<int>{0, 1},
                                       std::vector<int>{0, 1, 2}, std::vector<int>{3},
                                       std::vector<int>{2, 0, 1}}) {
        bool accepted = true;
        try {
            validate_execution_device_ids(ids);
        } catch (...) {
            accepted = false;
        }
        failures += check(accepted, "an accepted device list was rejected (size " +
                                        std::to_string(ids.size()) + ")");
    }
    return failures;
}

int test_rejected_lists() {
    int failures = 0;
    failures += check(throws<std::invalid_argument>({}), "an empty device list is rejected");
    failures += check(throws<std::invalid_argument>({-1}), "a negative device id is rejected");
    failures += check(throws<std::invalid_argument>({0, -1}), "a negative id after a valid one");
    failures += check(throws<std::invalid_argument>({0, 0}), "a repeated device id is rejected");
    failures += check(throws<std::invalid_argument>({0, 1, 0}), "a repeated id at width 3");
    failures += check(throws<std::invalid_argument>({1, 1, 2}), "a repeated non-zero id");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_accepted_lists();
    failures += test_rejected_lists();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
