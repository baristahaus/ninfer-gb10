#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cstddef>
#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer {

struct GdnReplayRecordSpec {
    std::int32_t layers          = 0;
    std::int32_t record_capacity = 0;
    std::int32_t width           = 0;
    std::int32_t conv_channels   = 0;
    std::int32_t qk_heads        = 0;
    std::int32_t value_heads     = 0;
    std::int32_t key_dim         = 0;
    std::int32_t value_dim       = 0;
};

struct GdnReplayRecordLayout {
    GdnReplayRecordSpec spec;
    TensorRegion conv;
    TensorRegion key;
    TensorRegion value;
    TensorRegion gate;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

[[nodiscard]] GdnReplayRecordLayout plan_gdn_replay_records(LayoutBuilder& builder,
                                                            const GdnReplayRecordSpec& spec);

// `rank`'s replay-record spec at `width`: the per-head extents halve and everything else stays
// whole. The records are the GDN state a speculative round rewinds, so a rank must hold exactly the
// heads its own GDN layers compute — the same three fields `shard_state_image_spec` halves for the
// recurrent state (`value_heads`, `conv_channels`, and here `qk_heads` too, because the replay
// record carries the key side explicitly).
//
// Deliberately whole on every rank:
//   * `layers`: both ranks carry every layer; the shard is inside a layer, never across layers.
//   * `record_capacity`: the rank-1 mirror addresses record rows by index, so equal capacity is
//     part of its contract.
//   * `width`: the draft window of the speculative round, a schedule extent, not a per-rank one.
//   * `key_dim`, `value_dim`: per-head extents. A rank owns whole heads, so a head is never cut.
//
// The head counts must divide by the width, not merely the channel sum: 16 key heads do not divide
// by 3 even though some of the derived extents do, and a rank with a partial head would compute a
// different head than its KV/state mirror holds. `conv_channels` must also equal
// `2 * qk_heads * key_dim + value_heads * value_dim`, which is the invariant the GDN state plan
// relies on; a spec that breaks it is rejected rather than halved into an inconsistent geometry.
//
// Width 1 returns `spec` unchanged, field for field, so the single-device lane is untouched by
// construction. Throws std::invalid_argument for a width below 1, a rank outside `[0, width)`, a
// non-positive extent, a broken channel invariant, or a head count that does not divide.
[[nodiscard]] GdnReplayRecordSpec shard_gdn_replay_spec(const GdnReplayRecordSpec& spec, int width,
                                                        int rank);

struct GdnReplayRecordLayer {
    Tensor conv;  // BF16 [conv_channels, width, rows]
    Tensor key;   // BF16 [key_dim, qk_heads, width, rows]
    Tensor value; // BF16 [value_dim, value_heads, width, rows]
    Tensor gate;  // FP32 [2, value_heads, width, rows], ordered {g, beta}
};

/**
 * Bound non-owning all-layer ReplaySSM transition records.
 *
 * Layer and physical record row share the outer index `layer * record_capacity + row` in every
 * plane. The object owns no allocation and stores no active-row or valid-prefix metadata.
 */
struct GdnReplayRecords {
    Tensor conv;
    Tensor key;
    Tensor value;
    Tensor gate;
    GdnReplayRecordSpec spec;

    GdnReplayRecords() = default;
    GdnReplayRecords(MemorySpan backing, const GdnReplayRecordLayout& layout);

    [[nodiscard]] GdnReplayRecordLayer layer(std::int32_t layer, std::int32_t rows) const;
};

/**
 * Copies record rows [0, rows) of every layer and plane from source to destination. One strided
 * device-to-device copy per layer and plane, stream-ordered and graph-capturable. Both sets share
 * one valid spec and the same plane geometry; rows must be in [1, record_capacity].
 */
void copy_gdn_replay_record_rows(const GdnReplayRecords& source,
                                 const GdnReplayRecords& destination, std::int32_t rows,
                                 cudaStream_t stream);
} // namespace ninfer
