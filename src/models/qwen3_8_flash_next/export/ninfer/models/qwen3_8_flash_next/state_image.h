#pragma once

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/buffer.h"
#include "core/layout.h"
#include "core/cyclic_kv_cache_pool.h"
#include "core/linear_attention_state.h"
#include "core/linear_attention_state_pool.h"
#include "core/tensor.h"
#include "core/transfer_work.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next {

struct DFlashLocalStateSpec {
    std::uint32_t layers   = 0;
    std::uint32_t capacity = 0;
    std::int32_t kv_heads  = 0;
    std::int32_t head_dim  = 0;
};

struct PleStateSpec {
    std::int32_t channels = 0;
    std::int32_t width    = 0;
};

struct StateImageSpec {
    LinearAttentionStatePoolSpec linear;
    std::int32_t hidden = 0;
    // Optional target-private predictor continuation. Flash-Next MTP carries the
    // pre-final-mixer four-stream activation independently from the collapsed
    // continuation hidden used by the target LM head.
    std::optional<std::int32_t> mtp_hidden;
    std::optional<PleStateSpec> ple;
    std::optional<DFlashLocalStateSpec> dflash_local;
};

struct StateImageHostLayout {
    StateImageSpec spec;
    LayoutRegion linear_conv;
    std::size_t linear_conv_layer_bytes = 0;
    LayoutRegion linear_recurrent;
    std::size_t linear_recurrent_layer_bytes = 0;
    LayoutRegion continuation_hidden;
    std::optional<LayoutRegion> mtp_hidden;
    std::optional<LayoutRegion> ple;
    std::optional<LayoutRegion> dflash_local_k;
    std::optional<LayoutRegion> dflash_local_v;
    std::size_t dflash_local_layer_bytes = 0;
    std::size_t image_bytes              = 0;
};

struct StateImageDeviceLayout {
    LinearAttentionStatePoolLayout linear;
    TensorRegion continuation_hidden;
    std::optional<TensorRegion> mtp_hidden;
    std::optional<TensorRegion> ple;
    std::optional<CyclicKVCachePoolLayout> dflash_local;
    StateImageHostLayout host;
};

[[nodiscard]] TransferWork state_image_transfer_work(const StateImageHostLayout& layout);
[[nodiscard]] TransferWork dflash_local_transfer_work(const StateImageHostLayout& layout);

[[nodiscard]] StateImageDeviceLayout plan_state_image_device_pool(LayoutBuilder& builder,
                                                                  const StateImageSpec& spec);

// One rank's StateImage geometry at tensor-parallel `width`: the spec the rank's own StateImage
// pool is planned from, so a rank-1 pool exists to attach a mirror to. Built from the same model
// configuration as the decoder geometry (`shard_decoder_state_spec`), and the two are meant to be
// produced together when a Program is planned per rank.
//
// Three fields shard, all of them head- or channel-axis extents that the tp2 placement table
// splits. The placement table is the authority; the table below names it
// (`src/models/qwen3_8_flash_next_125b_a6b/impl/shard_rules.cpp:37-49`).
//
// | field | 125B | at width 2 | why |
// |---|---|---|---|
// | `linear.value_heads` | 48 | **24** | the GDN recurrent state is one matrix per value head: `{key_head_dim, value_head_dim, value_heads}`. The placement table splits `in_proj_z` rows, `in_proj_a`/`in_proj_b` rows, `A_log` and `dt_bias` by the 48 value heads, and `out_proj` columns by the 6144 value dims. |
// | `linear.conv_channels` | 10240 | **5120** | the GDN convolution state is `{conv_channels, conv_width}` and `conv_channels = 2 * key_dim + value_dim` (`config.h:29-31`). `conv1d.weight` is split by columns over those 10240 Q\|K\|V channels, so a rank's conv state covers its own channel slice. |
// | `dflash_local.kv_heads` | per drafter | **/ width** | the drafter's local cyclic KV is per KV head, exactly like the text KV cache. `DFlashConfig::kv_heads` must divide by the width for the same reason `DecoderStateSpec::kv_heads` must. |
//
// Everything else is deliberately whole on every rank:
//   * `linear.layers` and `dflash_local.layers`: both ranks carry every layer; the shard is inside
//     a layer, never across layers.
//   * `linear.slot_count` and `dflash_local.capacity`: the rank-1 mirror addresses slots by index,
//     so equal slot capacity is part of its contract, not a per-rank choice.
//   * `linear.conv_width` (the convolution window), `linear.value_head_dim`, `linear.key_head_dim`,
//     `dflash_local.head_dim`: per-head or per-window extents, not per-rank ones.
//   * `linear.conv_dtype`: both ranks use the same state codec.
//   * `hidden`, `mtp_hidden`: activation vectors of the residual stream. Every rank's next layer
//     consumes the whole width, so these stay whole; the output head is split by vocabulary rows,
//     not by hidden columns.
//   * `ple`: the per-layer embedding state is an additive embedding of `channels = 4 * hidden`,
//     consumed by a projection whose input must be whole on both ranks.
//
// A **primary-only drafter** (`NINFER_TP_DRAFTER=primary`: rank 0 holds the drafter's rings, rank 1
// none) is a different decision: it omits `dflash_local` from rank 1's spec entirely, which no
// per-field shard can express. This function does not make that choice; it shards the component
// when the spec has one, and a mirror's component-inventory check refuses a pair where one side
// omits it.
//
// Width 1 returns `spec` unchanged, field for field, so the single-device lane is untouched by
// construction. Throws std::invalid_argument for a width below 1, a rank outside `[0, width)`, a
// non-positive `value_heads`/`conv_channels`, a `conv_channels` that is not `2 * key_dim +
// value_dim` for the spec's own value dims, or an extent that does not divide by the width.
[[nodiscard]] StateImageSpec shard_state_image_spec(const StateImageSpec& spec, int width, int rank);

struct HostStateImageView {
    std::byte* data                    = nullptr;
    const StateImageHostLayout* layout = nullptr;
};

struct HostStateImageConstView {
    const std::byte* data              = nullptr;
    const StateImageHostLayout* layout = nullptr;
};

struct HostStateSlotHandle {
    std::uint32_t index      = 0;
    std::uint32_t generation = 0;
};

/** Fixed-capacity pinned storage for complete physical StateImage payloads; owns no cache policy.
 */
class HostStatePool {
public:
    HostStatePool(StateImageHostLayout layout, std::uint32_t capacity);

    HostStatePool(const HostStatePool&)            = delete;
    HostStatePool& operator=(const HostStatePool&) = delete;
    HostStatePool(HostStatePool&&)                 = delete;
    HostStatePool& operator=(HostStatePool&&)      = delete;

    [[nodiscard]] std::optional<HostStateSlotHandle> allocate() noexcept;
    [[nodiscard]] bool release(HostStateSlotHandle handle) noexcept;

    [[nodiscard]] HostStateImageView writable_view(HostStateSlotHandle handle);
    [[nodiscard]] HostStateImageConstView view(HostStateSlotHandle handle) const;

    [[nodiscard]] std::uint32_t capacity() const noexcept;

    [[nodiscard]] std::uint32_t occupied() const noexcept { return occupied_; }

    [[nodiscard]] const StateImageHostLayout& layout() const noexcept { return layout_; }

private:
    struct Slot {
        std::uint32_t generation = 1;
        bool occupied            = false;
    };

    [[nodiscard]] bool valid(HostStateSlotHandle handle) const noexcept;
    [[nodiscard]] std::byte* slot_data(std::uint32_t index) const noexcept;

    StateImageHostLayout layout_;
    std::optional<PinnedHostBuffer> backing_;
    std::vector<Slot> slots_;
    std::vector<std::uint32_t> free_slots_;
    std::uint32_t free_count_ = 0;
    std::uint32_t occupied_   = 0;
};

struct StateImageDeviceSlotView {
    LinearAttentionStateSlotView linear;
    Tensor continuation_hidden;
    std::optional<Tensor> mtp_hidden;
    std::optional<Tensor> ple;
    std::optional<CyclicKVCacheSlotView> dflash_local;
};

/**
 * Caller-backed fixed storage for Qwen3.8 continuation state.
 *
 * Every absolute slot contains common GDN/hidden state and, for a DFlash Program, its local cyclic
 * K/V state. The pool owns neither slot roles nor logical checkpoint identity.
 */
class StateImageDevicePool {
public:
    StateImageDevicePool(MemorySpan backing, const StateImageDeviceLayout& layout);

    StateImageDevicePool(const StateImageDevicePool&)            = delete;
    StateImageDevicePool& operator=(const StateImageDevicePool&) = delete;
    StateImageDevicePool(StateImageDevicePool&&)                 = delete;
    StateImageDevicePool& operator=(StateImageDevicePool&&)      = delete;

    [[nodiscard]] std::int32_t slot_count() const noexcept { return linear_.slot_count(); }

    [[nodiscard]] StateImageDeviceSlotView slot_view(std::int32_t slot) const;
    [[nodiscard]] Tensor continuation_hidden_slot(std::int32_t slot) const;

    [[nodiscard]] LinearAttentionStatePool& linear() noexcept { return linear_; }

    [[nodiscard]] const LinearAttentionStatePool& linear() const noexcept { return linear_; }

    [[nodiscard]] Tensor& continuation_hidden_store() noexcept { return continuation_hidden_; }

    [[nodiscard]] const Tensor& continuation_hidden_store() const noexcept {
        return continuation_hidden_;
    }

    [[nodiscard]] Tensor* mtp_hidden_store() noexcept {
        return mtp_hidden_ ? &*mtp_hidden_ : nullptr;
    }

    [[nodiscard]] const Tensor* mtp_hidden_store() const noexcept {
        return mtp_hidden_ ? &*mtp_hidden_ : nullptr;
    }

    [[nodiscard]] Tensor mtp_hidden_slot(std::int32_t slot) const;

    [[nodiscard]] Tensor* ple_store() noexcept { return ple_ ? &*ple_ : nullptr; }

    [[nodiscard]] const Tensor* ple_store() const noexcept { return ple_ ? &*ple_ : nullptr; }

    [[nodiscard]] Tensor ple_slot(std::int32_t slot) const;

    [[nodiscard]] CyclicKVCache* dflash_local() noexcept;
    [[nodiscard]] const CyclicKVCache* dflash_local() const noexcept;

    [[nodiscard]] const StateImageHostLayout& host_layout() const noexcept { return host_layout_; }

    void zero_slot(std::int32_t slot, cudaStream_t stream = nullptr);
    void zero_all(cudaStream_t stream = nullptr);
    void copy_slot(std::int32_t source, std::int32_t destination, cudaStream_t stream = nullptr);
    void copy_dflash_local(std::int32_t source, std::int32_t destination,
                           cudaStream_t stream = nullptr);
    void copy_to_host(std::int32_t source, HostStateImageView destination,
                      cudaStream_t stream = nullptr) const;
    void copy_from_host(HostStateImageConstView source, std::int32_t destination,
                        cudaStream_t stream = nullptr);

private:
    void validate_host_layout(const StateImageHostLayout* layout, const std::byte* data) const;

    LinearAttentionStatePool linear_;
    Tensor continuation_hidden_;
    std::optional<Tensor> mtp_hidden_;
    std::optional<Tensor> ple_;
    std::optional<CyclicKVCache> dflash_local_;
    StateImageHostLayout host_layout_;
};

} // namespace ninfer::models::qwen3_8_flash_next
