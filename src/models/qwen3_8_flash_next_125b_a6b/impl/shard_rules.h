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
// Two rows are packed per head and need the kernel's layout confirmed before a shard is applied:
// `self_attn.q_proj.weight` is [12288, 2560] = 48 head-slots (24 gated query heads, each a
// query+gate pair) and `linear_attn.in_proj_qkv.weight` is [10240, 2560] = 80 head-slots (16 Q,
// 16 K, 48 V). A contiguous half is only correct if those slots are laid out so that each rank's
// half keeps every query head with its gate and every value head with its key; the rule is stated
// here and the packing is asserted where the shard is applied, not in this table.

#include "artifact/shard.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace ninfer::models::flash_next_125b_a6b {

struct ParameterShard {
    std::string_view suffix;
    artifact::ShardAxis axis;
    std::int64_t extent;
};

[[nodiscard]] std::span<const ParameterShard> parameter_shards() noexcept;

} // namespace ninfer::models::flash_next_125b_a6b
