#pragma once

#include "core/layout.h"
#include "core/paged_kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next {

inline constexpr std::int32_t kKvInt8QuantGroup = 64;
inline constexpr std::int32_t kKvFp8QuantGroup  = 256;

struct DecoderStateSpec {
    std::uint32_t full_attention_layers     = 0;
    std::uint32_t mtp_layers                = 0;
    std::uint32_t capacity                  = 0;
    std::int32_t kv_heads                   = 0;
    std::int32_t attention_head_dim         = 0;
    KvCacheStorage kv_storage               = KvCacheStorage::BFloat16;
    bool enable_mtp                         = false;
    std::int32_t kv_table_rows              = 1;
    std::uint32_t text_physical_page_groups = 0;
    std::uint32_t mtp_physical_page_groups  = 0;
    std::vector<KVPlaneGeometry> auxiliary_planes;
};

struct PagedKVCacheLayout {
    DeviceKVPagePoolLayout pages;
    KVExecutionTableLayout execution_tables;
    std::uint32_t layers      = 0;
    std::uint32_t max_context = 0;
    std::int32_t kv_heads     = 0;
    PagedKVStorageLayout layer_storage;
    std::vector<KVPlaneGeometry> auxiliary_planes;

    [[nodiscard]] std::size_t payload_bytes() const noexcept { return pages.payload_bytes(); }
};

class PagedKVCache;

class PagedKVCacheView {
public:
    PagedKVCacheView() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return cache_ != nullptr; }

    [[nodiscard]] std::uint32_t max_context() const noexcept;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer) const;

private:
    friend class PagedKVCache;
    PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept;

    const PagedKVCache* cache_ = nullptr;
    Tensor block_table_;
};

class PagedKVCache {
public:
    PagedKVCache(MemorySpan backing, const PagedKVCacheLayout& layout);

    PagedKVCache(const PagedKVCache&)            = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;
    PagedKVCache(PagedKVCache&&)                 = delete;
    PagedKVCache& operator=(PagedKVCache&&)      = delete;

    [[nodiscard]] std::uint32_t max_context() const noexcept { return max_context_; }

    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }

    [[nodiscard]] DeviceKVPagePool& page_pool() noexcept { return pages_; }

    [[nodiscard]] const DeviceKVPagePool& page_pool() const noexcept { return pages_; }

    [[nodiscard]] KVExecutionTablePool& execution_tables() noexcept { return execution_tables_; }

    [[nodiscard]] const KVExecutionTablePool& execution_tables() const noexcept {
        return execution_tables_;
    }

    [[nodiscard]] PagedKVCacheView execution_view(const KVExecutionRowLease& row) const;

    [[nodiscard]] PagedKVBatchLayerView batch_layer_view(std::uint32_t layer) const;

private:
    friend class PagedKVCacheView;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer, Tensor block_table) const;

    DeviceKVPagePool pages_;
    KVExecutionTablePool execution_tables_;
    std::uint32_t layers_      = 0;
    std::uint32_t max_context_ = 0;
    std::int32_t kv_heads_     = 0;
    PagedKVStorageLayout layer_storage_;
    std::vector<KVPlaneGeometry> auxiliary_planes_;
};

struct DecoderStateLayout {
    PagedKVCacheLayout text_kv;
    std::optional<PagedKVCacheLayout> mtp_kv;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept;
};

[[nodiscard]] DecoderStateLayout plan_decoder_state(LayoutBuilder& builder,
                                                    const DecoderStateSpec& spec);

// One rank's decoder geometry at tensor-parallel `width`: the spec the rank's own page pool and
// execution tables are planned from, so a rank-1 pool exists to attach a mirror to.
//
// Exactly one field shards. `kv_heads` is the number of KV heads in every text and MTP layer, and
// at tp 2 a rank owns half of them (2 -> 1): the placement table splits the K/V projections by KV
// heads, so a rank's pool holds its own heads and nothing else. `kv_heads % width` must be 0; a
// remainder would give the ranks different per-page plane extents, which no kernel or mirror can
// reconcile, so it throws instead of rounding. Width 3 against the 125B's 2 KV heads is therefore
// refused, which is the review's head-count finding reaching the decoder geometry.
//
// Everything else is deliberately whole on every rank:
//   * `attention_head_dim` is per head, not per rank.
//   * `capacity`, `text_physical_page_groups`, `mtp_physical_page_groups`, `kv_table_rows` and
//     `kv_storage` must be equal on both ranks: the rank-1 mirror addresses the origin's physical
//     pages by index, so equal page capacity is part of its contract, not a per-rank choice.
//   * `full_attention_layers`, `mtp_layers`, `enable_mtp`: both ranks carry every layer; the shard
//     is inside a layer, never across layers.
//   * `auxiliary_planes` are shard-invariant. They are the QSA index planes, not GDN state: a
//     BF16 `[128, 64, pages, 1]` index-key plane and an I32 `[3, 64, pages, 1]` control plane,
//     appended to every layer's plane set (impl/state/decoder_state.cpp:46-48) and required at
//     exactly that geometry by the QSA kernel (ops/softmax_attention/qsa/flash_next_qsa.cu:
//     1687-1691). Neither carries a head axis — `head_extent` is 1 — so there is nothing to
//     divide, and the indexer is replicated on every rank by the tp2 placement table because the
//     shared TopK decision has to be identical on both.
//
// The GDN recurrent state is *not* here. It lives in the StateImage (`StateImageSpec::linear`,
// which does carry `value_heads = 48`), so its shard belongs to a `shard_state_image_spec` that
// does not exist yet — a different structure, and not a gap in this one.
//
// Width 1 returns `spec` unchanged, field for field, so the single-device lane is untouched.
// Throws std::invalid_argument for a width below 1, a rank outside `[0, width)`, a non-positive
// `kv_heads`, or a `kv_heads` that does not divide by `width`.
[[nodiscard]] DecoderStateSpec shard_decoder_state_spec(const DecoderStateSpec& spec, int width,
                                                       int rank);

struct DecoderState {
    PagedKVCache text_kv;
    std::optional<PagedKVCache> mtp_kv;

    DecoderState(MemorySpan backing, const DecoderStateLayout& layout);

    [[nodiscard]] PagedKVCache* mtp_cache() noexcept;
    [[nodiscard]] const PagedKVCache* mtp_cache() const noexcept;
};

} // namespace ninfer::models::qwen3_8_flash_next
