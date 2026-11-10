#pragma once

// tp2 rank-1 mirror of the Flash-Next StateImage device pool.
//
// The core mirror (core/tp2/kv_mirror.h) replays a core StateImageStorage. This family stores its
// continuation state as a composite — a LinearAttentionStatePool, a continuation-hidden tensor and
// optional MTP-hidden, PLE and DFlash-local components — so its operations are family-shaped
// (zero_slot / zero_all / copy_slot / copy_dflash_local, raw stream) and the family keeps its own
// mirror wrapper instead of flattening that storage into one core image. See
// review/phase3/state-image.md §2 for why this is the family's shape rather than a core view.
//
// Contract, the same as the KV mirror:
//   * attach before the origin's first mutation;
//   * index-identical, not byte-identical: rank 1 replays the same slot indices on its own shard,
//     whose component widths may differ (GDN channels, value heads, KV heads are halved at tp 2);
//   * every replay runs on rank 1's stream, with rank 1's device current;
//   * Host transfers are refused, because tp 2 has no Host state tier.

#include "core/tp2/kv_mirror.h"
#include "ninfer/models/qwen3_8_flash_next/state_image.h"

#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next {

// Validates that `mirror` can replay `origin`'s slot operations: equal slot count, the same
// component inventory, and equal per-layer iteration counts. Component widths are deliberately not
// compared — those are the per-rank shard widths. Throws std::invalid_argument.
void require_matching_state_image_device_pool(const StateImageDevicePool& origin,
                                              const StateImageDevicePool& mirror);

// Rank 0's StateImage device pool with rank 1's replica attached. Construct once, before any slot
// is zeroed or copied.
class MirroredStateImageDevicePool {
public:
    MirroredStateImageDevicePool(StateImageDevicePool& origin, StateImageDevicePool& mirror,
                                 tp2::MirrorTarget where);

    MirroredStateImageDevicePool(const MirroredStateImageDevicePool&)            = delete;
    MirroredStateImageDevicePool& operator=(const MirroredStateImageDevicePool&) = delete;

    [[nodiscard]] const tp2::MirrorTarget& target() const noexcept { return where_; }
    [[nodiscard]] StateImageDevicePool& origin_pool() noexcept { return origin_; }
    [[nodiscard]] StateImageDevicePool& mirror_pool() noexcept { return mirror_; }

    void zero_slot(std::int32_t slot, cudaStream_t stream = nullptr);
    void zero_all(cudaStream_t stream = nullptr);
    void copy_slot(std::int32_t source, std::int32_t destination, cudaStream_t stream = nullptr);
    void copy_dflash_local(std::int32_t source, std::int32_t destination,
                           cudaStream_t stream = nullptr);

    // tp 2 has no Host state tier: rank 1 has no Host copy, so a spill or restore would leave the
    // ranks' state divergent. These throw std::logic_error instead of half-copying.
    void copy_to_host(std::int32_t source, HostStateImageView destination,
                      cudaStream_t stream = nullptr) const;
    void copy_from_host(HostStateImageConstView source, std::int32_t destination,
                        cudaStream_t stream = nullptr) const;

private:
    StateImageDevicePool& origin_;
    StateImageDevicePool& mirror_;
    tp2::MirrorTarget where_;
};

} // namespace ninfer::models::qwen3_8_flash_next
