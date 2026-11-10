#pragma once

// The tensor-parallel placement of each logical parameter of the Qwen3.5 family, for the
// Qwen3.8-27B (dense, 64-layer) geometry — the member that fits a two-GPU box.
//
// One row per parameter, at the axis that parameter's own packing requires. `extent` is the
// sharded extent: rows for `Rows`, columns for `Columns`, and the whole parameter size for the
// placements that do not split. The arithmetic is `artifact::split_evenly`; this table only says
// which axis each parameter takes, because that follows the model's head and channel counts.
//
// The 27B geometry (tools/convert/qwen3_5.py, the 27B recipe, and review/95-NEXT-STEP.md §2):
// hidden 5120, 64 layers of which every fourth is full attention (16) and the rest GDN (48),
// 24 query heads and 4 KV heads of width 256, GDN 16 key heads and 48 value heads of width 128,
// dense intermediate 17408, vocabulary 248320.
//
// Width 2 divides every row: 24 query heads and 4 KV heads halve, 16 GDN key heads and 48 value
// heads halve, the 248320 vocabulary and the 17408 intermediate halve. At width 3 the KV heads,
// the GDN key heads, the dense intermediate and the vocabulary do not — see the test.
//
// Three packings were checked against the consuming code before the rows could be stated. One of
// them changes the row:
//
//   `attention/query`, `attention/key`, `attention/gate` and `attention/value` are FOUR separate
//   logical parameters (`load/text.cpp:14-17`), not one packed projection. The execution layer
//   packs them per rank at construction (`execution/parameters.cpp:104-125`, role order query,
//   key, gate, value) from that rank's own bindings, so the pack is rebuilt per rank and a
//   contiguous half of each parameter gives a rank 12 complete query heads. Rows is correct for
//   all four; no segment is needed. The same holds for the GDN projection, which is four separate
//   parameters (`gdn/query`, `gdn/key`, `gdn/value`, `gdn/z`) packed per rank into Q|K|V + Z.
//
//   `gdn/convolution` is [4, 10240] in the artifact: the taps are rows and the 10240 depthwise
//   channels are columns, in Q|K|V order (the converter transposes HF's `conv1d.weight`
//   (channels,1,taps) to (taps,channels); `tools/convert/qwen3_5.py:562-569`). Its channel axis is
//   therefore three consecutive segments Q 2048 | K 2048 | V 6144, and a contiguous half would
//   give rank 0 all of Q and K plus a quarter of V and rank 1 the rest of V with no Q or K. It is
//   stated as three segments that shard independently, exactly like the Flash-Next QKV projection.
//   The Flash-Next table segments its own equivalent `linear_attn.conv1d.weight` row the same way
//   (2048 | 2048 | 6144) and its test asserts the segmentation, so both tables state the channel
//   order rather than a contiguous block.
//
//   `text/token_embedding` is replicated and `text/output_head` shards by rows. Every rank must
//   gather any token id, so the embedding cannot be vocabulary-split; the head's rows are pure
//   output features, so they can be.
//
// Scope rule: only a parameter whose name starts with `text/` or `mtp/` is matched. The draft
// components (`dflash/*`, `dflash2/*`) bind block parameters with the SAME suffixes
// (`load/dflash.cpp:27-39`) but their own geometry (32 query / 8 KV heads of width 128), so a
// suffix match on them would apply this table's 27B extents and throw. The drafter's split is a
// separate decision (capy splits it across ranks; ours is not sharded yet) and stays whole here.
// Vision, proposal and the DFlash2 selector are outside the table and therefore replicated.
//
// This table is the 27B's geometry. A MoE family member (Qwen3.6-35B-A3B) has different extents,
// so loading one through this transform fails at the extent check instead of silently mis-sharding
// it; that member needs its own table.

#include "artifact/binder.h"
#include "artifact/shard.h"

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5 {

// One logical parameter's placement. `segments` are its consecutive blocks, each of which shards
// independently: most parameters have one segment, but GDN's convolution has three.
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

// True when this table owns `name`: the parameter is under `text/` or `mtp/`. The draft components
// reuse the same suffixes with a different geometry, so the scope rule is part of the contract.
[[nodiscard]] bool in_scope(std::string_view name) noexcept;

// The parameter as `rank` sees it at `width`: the same reference with its binding sliced to the
// rank's ranges and its sharded extent reduced to the rank's share. `shape[0]` shrinks for Rows and
// `shape[1]` for Columns — a segmented parameter shrinks by the sum of its per-rank segments, so
// the convolution's 10240 columns become 5120 at width 2. A parameter the table does not name, or
// one outside the scope rule, is Replicated and comes back unchanged. At width 1 the reference is
// returned unchanged.
//
// Throws std::invalid_argument when the table names the parameter but its binding does not cover
// the extents the table declares: that is a mismatch between the artifact and the model, and
// clipping it silently would give a rank fewer elements than its kernels expect.
[[nodiscard]] artifact::ParameterReference
shard_parameter(const artifact::ParameterReference& reference, int width, int rank);

// `rank`'s placement policy as the binder's transform, so a caller that binds one plan per rank
// asks the model for the policy and never names a shard axis itself.
//
// Width 1 returns an empty function rather than an identity one: the binder's contract is that an
// unset transform is the single-device lane, and `has_shard_transform()` must report that lane
// honestly. An identity lambda would make a width-1 load look sharded to anything that asks.
[[nodiscard]] artifact::Binder::ShardTransform shard_transform(int width, int rank);

} // namespace ninfer::models::qwen3_5
