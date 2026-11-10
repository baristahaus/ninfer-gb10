#pragma once

// The tensor-parallel placement of each logical parameter of the Flash-Next 125B-A6B.
//
// One row per parameter, at the axis that parameter's own packing requires. `extent` is the
// sharded extent: rows for `Rows`, columns for `Columns`, and the whole parameter size for the
// placements that do not split. The arithmetic is `artifact::split_evenly`; this table only says
// which axis each parameter takes, because that follows the model's head and channel counts.
//
// Width 2 divides every row. That is the review's head-count finding in table form: 24 query heads
// and 2 KV heads halve, 16 GDN key heads and 48 value heads halve, the 248320 vocabulary and the
// 640 shared-expert rows halve — and at width 3 the KV, GDN-key, vocabulary and shared-expert rows
// do not.
//
// Two rows needed the kernel's slot layout checked before they could be stated, and the check
// changed one of them:
//
//   self_attn.q_proj.weight is [12288, 2560]. `split_query_gate_kernel` reads the packed row at
//   head * 2 * head_dim for the query and + head_dim for the gate, so the layout is per-head
//   interleaved [q0 g0 q1 g1 ...]. A contiguous half therefore gives each rank 12 complete
//   head-pairs, and the single-segment Rows split is correct.
//
//   linear_attn.in_proj_qkv.weight is [10240, 2560], but `causal_conv1d_silu` splits that width as
//   three consecutive segments: Q 2048, K 2048, V 6144. A contiguous half would give rank 0 all of
//   Q and K plus a sixth of V, and rank 1 the rest of V with no Q or K at all. It is therefore
//   stated as three segments that shard independently.

#include "artifact/binder.h"
#include "artifact/shard.h"

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::models::flash_next_125b_a6b {

// One logical parameter's placement. `segments` are its consecutive blocks, each of which shards
// independently: most parameters have one segment, but GDN's QKV projection has three.
struct ParameterShard {
    std::string_view suffix;
    artifact::ShardAxis axis;
    // The parameter's consecutive blocks, each sharding independently. Entries past
    // `segment_count` are zero. A fixed array rather than a span: the table is constexpr and a span
    // initialised from a braced list would point at a temporary.
    std::array<std::int64_t, 3> segments{};
    std::uint8_t segment_count = 1;
};

// The parameter's total logical extent.
[[nodiscard]] std::int64_t total_extent(const ParameterShard& shard) noexcept;

// One rank's logical ranges for `shard` at `width`, one block per segment. A segmented parameter
// gives several disjoint ranges, which is why the resolver takes a range list.
[[nodiscard]] std::vector<artifact::ShardRange> ranges_for(const ParameterShard& shard, int width,
                                                          int rank);

[[nodiscard]] std::span<const ParameterShard> parameter_shards() noexcept;

// The parameter as `rank` sees it at `width`: the same reference with its binding sliced to the
// rank's ranges and its sharded extent reduced to the rank's share. `shape[0]` shrinks for Rows and
// `shape[1]` for Columns — a segmented parameter shrinks by the sum of its per-rank segments, so
// the QKV projection's 10240 rows become 5120 at width 2. A parameter the table does not name is
// Replicated and comes back unchanged. At width 1 the reference is returned unchanged.
//
// Throws std::invalid_argument when the table names the parameter but its binding does not cover
// the extents the table declares: that is a mismatch between the artifact and the model, and
// clipping it silently would give a rank fewer elements than its kernels expect.
[[nodiscard]] artifact::ParameterReference shard_parameter(
    const artifact::ParameterReference& reference, int width, int rank);

// `rank`'s placement policy as the binder's transform, so a caller that binds one plan per rank
// asks the model for the policy and never names a shard axis itself.
//
// Width 1 returns an empty function rather than an identity one: the binder's contract is that an
// unset transform is the single-device lane, and `has_shard_transform()` must report that lane
// honestly. An identity lambda would make a width-1 load look sharded to anything that asks.
[[nodiscard]] artifact::Binder::ShardTransform shard_transform(int width, int rank);

} // namespace ninfer::models::flash_next_125b_a6b
