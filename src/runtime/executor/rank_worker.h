#pragma once

#include "core/device.h"
#include "models/common/sharded_group.h"
#include "runtime/executor/contract.h"

#include <memory>
#include <utility>

namespace ninfer::runtime {

template <class Types>
struct RankModelInstance {
    std::unique_ptr<typename Types::ModelShard> model;
    const typename Types::Parameters parameters;
    std::unique_ptr<typename Types::RankProgram> program;
    // The group-scoped resources of a two-GPU lane (the shared mailbox and the partial pair),
    // borrowed from the owner that declared them — they outlive every instance. Null on the
    // single-device lane.
    const models::ShardedGroup* sharded = nullptr;

    RankModelInstance(std::unique_ptr<typename Types::ModelShard> model,
                      std::shared_ptr<const typename Types::Profile> profile)
        : model(std::move(model)), parameters(*this->model, std::move(profile)) {}
};

// The device object remains at the same address from startup through shutdown. Programs may
// borrow its streams, and every allocation owner is destroyed before the device. The worker
// borrows the device rather than owning it: on the two-GPU lane the group owns every device, and
// the worker is one rank's user of it.
template <class Types>
class RankWorker {
public:
    using Protocol = typename Types::ExecutionTypes;

    RankWorker(DeviceContext& device, std::unique_ptr<RankModelInstance<Types>> instance)
        : device_(device), instance_(std::move(instance)) {}

    ~RankWorker() {
        device_.bind_to_current_thread_noexcept();
        instance_.reset();
    }

    // Binds this worker's own rank. A group-wide bind leaves the last rank current, so per-rank
    // work calls this immediately before touching the rank (see LocalExecutor::bind_rank).
    void bind_to_current_thread() { device_.bind_to_current_thread(); }

    [[nodiscard]] Device device() const noexcept { return device_.device(); }

    [[nodiscard]] const char* device_name() const noexcept { return device_.properties().name; }

    [[nodiscard]] typename Types::RankProgram& program() noexcept { return *instance_->program; }

    [[nodiscard]] const typename Types::RankProgram& program() const noexcept {
        return *instance_->program;
    }

private:
    DeviceContext& device_;
    std::unique_ptr<RankModelInstance<Types>> instance_;
};

} // namespace ninfer::runtime
