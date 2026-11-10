// The group-scoped resources of the two-Program lane: the rank rule (host-only, pure) and the owner
// (needs two devices).
//
// The owner is the thing that must exist exactly once. Its two properties are checked here: one
// instance shared by both Programs, and one partial buffer per rank on that rank's own device. The
// rank rule is pure, so it is checked without a device — it is the rule that decides which stream a
// rank's exchange lands on, and getting it wrong would send both ranks' exchanges to one stream.

#include "models/common/sharded_group.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

template <typename Exception, typename Fn>
int expect_throws(Fn&& fn, const std::string& label) {
    try {
        fn();
    } catch (const Exception&) { return 0; }
    std::cerr << label << " did not throw the expected exception\n";
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver ||
           err == cudaErrorStubLibrary;
}

// --- host-only: the rank rule -----------------------------------------------------------

int exercise_rank_rule() {
    int failures = 0;
    const std::array<int, 2> pair{0, 1};
    failures += check(ninfer::models::sharded_rank_of(pair, 0) == 0, "device 0 is not rank 0");
    failures += check(ninfer::models::sharded_rank_of(pair, 1) == 1, "device 1 is not rank 1");
    failures += check(ninfer::models::sharded_rank_of(pair, 2) == -1,
                      "a device outside the group got a rank");
    const std::array<int, 1> single{3};
    failures += check(ninfer::models::sharded_rank_of(single, 3) == 0,
                      "a one-device group's member is not rank 0");
    failures += check(ninfer::models::sharded_rank_of(single, 0) == -1,
                      "a foreign device got a rank in a one-device group");
    failures += check(ninfer::models::sharded_rank_of(std::span<const int>{}, 0) == -1,
                      "an empty group gave a rank");
    return failures;
}

// --- two devices: the owner -------------------------------------------------------------

int exercise_owner(ninfer::ExecutionContext& execution) {
    int failures = 0;
    ninfer::models::ShardedGroupOwner owner(execution, 8, 16);

    const ninfer::models::ShardedGroup* first  = &owner.group();
    const ninfer::models::ShardedGroup* second = &owner.group();
    failures += check(first == second, "the two Programs would not borrow one group");
    failures += check(first->execution == &execution, "the group does not name its execution group");
    failures += check(first->mailbox != nullptr, "the group has no mailbox");
    failures += check(first->partials[0].data != nullptr && first->partials[1].data != nullptr,
                      "a partial buffer is missing");
    failures += check(first->partials[0].data != first->partials[1].data,
                      "the two ranks share one partial buffer");
    failures += check(first->partials[0].ne[0] == 8 && first->partials[0].ne[1] == 16,
                      "a partial is not [rows, columns]");
    failures += check(first->partials[0].dtype == ninfer::DType::BF16,
                      "a partial is not BF16");
    failures += check(owner.partial_device(0) == execution.device(0).device(),
                      "rank 0's partial is not on rank 0's device");
    failures += check(owner.partial_device(1) == execution.device(1).device(),
                      "rank 1's partial is not on rank 1's device");

    // The two Programs derive their ranks from the same group and get different answers.
    std::array<int, 2> devices{execution.device(0).device().index(),
                               execution.device(1).device().index()};
    failures += check(ninfer::models::sharded_rank_of(devices, devices[0]) == 0 &&
                          ninfer::models::sharded_rank_of(devices, devices[1]) == 1,
                      "the two Programs do not derive different ranks from one group");
    return failures;
}

int exercise_single_device_refusal(ninfer::ExecutionContext& execution) {
    return expect_throws<std::invalid_argument>(
        [&] {
            ninfer::models::ShardedGroupOwner owner(execution, 8, 16);
            (void)owner;
        },
        "a one-device group built a two-Program collective");
}

} // namespace

int main() {
    int failures = exercise_rank_rule();

    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cerr << "SKIP: no usable CUDA device; the owner cases did not run; supply one with CUDA_VISIBLE_DEVICES (sm_121a on GB10, sm_120a on RTX 5090)\n";
        return failures == 0 ? 77 : 1;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count < 2) {
        std::cerr << "SKIP: the two-Program collective needs two devices. Set CUDA_VISIBLE_DEVICES to two devices, for example 0,1\n";
        return failures == 0 ? 77 : 1;
    }

    try {
        ninfer::ExecutionContext two(std::vector<int>{0, 1});
        failures += exercise_owner(two);
        ninfer::ExecutionContext one(std::vector<int>{0});
        failures += exercise_single_device_refusal(one);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: threw: " << error.what() << '\n';
        return 1;
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
