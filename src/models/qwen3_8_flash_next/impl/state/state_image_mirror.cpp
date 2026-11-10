// Implements: src/models/qwen3_8_flash_next/impl/state/state_image_mirror.h
#include "models/qwen3_8_flash_next/impl/state/state_image_mirror.h"

#include "core/device.h"

#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_8_flash_next {
namespace {

void require_target(const tp2::MirrorTarget& where) {
    if (!where.valid()) {
        throw std::invalid_argument(
            "StateImage mirror requires rank 1's device and stream");
    }
}

// Runs `replay` with rank 1's device current. Stream-ordered copies and fills route by stream, so
// the scope is not strictly required; it keeps any future allocation or registration path safe.
template <typename Replay>
void on_mirror_device(const tp2::MirrorTarget& where, Replay&& replay) {
    const CudaDeviceScope scope(where.device->device());
    replay();
}

[[noreturn]] void throw_host_transfer_unavailable(const char* what) {
    throw std::logic_error(std::string(what) +
                           " is unavailable while a tp2 rank-1 mirror is attached: tp 2 has no "
                           "Host state tier");
}

} // namespace

void require_matching_state_image_device_pool(const StateImageDevicePool& origin,
                                              const StateImageDevicePool& mirror) {
    if (&origin == &mirror) {
        throw std::invalid_argument("StateImage mirror is the origin pool");
    }
    if (origin.slot_count() != mirror.slot_count()) {
        throw std::invalid_argument("StateImage mirror slot count differs");
    }
    if (origin.linear().layer_count() != mirror.linear().layer_count()) {
        throw std::invalid_argument("StateImage mirror linear layer count differs");
    }
    // An operation on a component the mirror does not have cannot be replayed at all.
    if ((origin.mtp_hidden_store() == nullptr) != (mirror.mtp_hidden_store() == nullptr)) {
        throw std::invalid_argument("StateImage mirror MTP-hidden inventory differs");
    }
    if ((origin.ple_store() == nullptr) != (mirror.ple_store() == nullptr)) {
        throw std::invalid_argument("StateImage mirror PLE inventory differs");
    }
    const CyclicKVCache* source_dflash  = origin.dflash_local();
    const CyclicKVCache* replica_dflash = mirror.dflash_local();
    if ((source_dflash == nullptr) != (replica_dflash == nullptr)) {
        throw std::invalid_argument("StateImage mirror DFlash-local inventory differs");
    }
    if (source_dflash != nullptr) {
        // Same per-layer replay and slot geometry. The head count and head dim are the rank's shard
        // widths and may differ; the slot count and layer count may not.
        if (source_dflash->layer_count() != replica_dflash->layer_count()) {
            throw std::invalid_argument("StateImage mirror DFlash layer count differs");
        }
        if (source_dflash->lane_capacity() != replica_dflash->lane_capacity()) {
            throw std::invalid_argument("StateImage mirror DFlash slot geometry differs");
        }
    }
}

MirroredStateImageDevicePool::MirroredStateImageDevicePool(StateImageDevicePool& origin,
                                                           StateImageDevicePool& mirror,
                                                           tp2::MirrorTarget where)
    : origin_(origin), mirror_(mirror), where_(where) {
    require_target(where_);
    require_matching_state_image_device_pool(origin_, mirror_);
}

void MirroredStateImageDevicePool::zero_slot(std::int32_t slot, cudaStream_t stream) {
    origin_.zero_slot(slot, stream);
    on_mirror_device(where_, [&] { mirror_.zero_slot(slot, where_.stream); });
}

void MirroredStateImageDevicePool::zero_all(cudaStream_t stream) {
    origin_.zero_all(stream);
    on_mirror_device(where_, [&] { mirror_.zero_all(where_.stream); });
}

void MirroredStateImageDevicePool::copy_slot(std::int32_t source, std::int32_t destination,
                                             cudaStream_t stream) {
    origin_.copy_slot(source, destination, stream);
    on_mirror_device(where_, [&] { mirror_.copy_slot(source, destination, where_.stream); });
}

void MirroredStateImageDevicePool::copy_dflash_local(std::int32_t source,
                                                     std::int32_t destination,
                                                     cudaStream_t stream) {
    origin_.copy_dflash_local(source, destination, stream);
    on_mirror_device(where_,
                     [&] { mirror_.copy_dflash_local(source, destination, where_.stream); });
}

void MirroredStateImageDevicePool::copy_to_host(std::int32_t source,
                                                HostStateImageView destination,
                                                cudaStream_t stream) const {
    (void)source;
    (void)destination;
    (void)stream;
    throw_host_transfer_unavailable("StateImage Device-to-Host copy");
}

void MirroredStateImageDevicePool::copy_from_host(HostStateImageConstView source,
                                                  std::int32_t destination,
                                                  cudaStream_t stream) const {
    (void)source;
    (void)destination;
    (void)stream;
    throw_host_transfer_unavailable("StateImage Host-to-Device restore");
}

} // namespace ninfer::models::qwen3_8_flash_next
