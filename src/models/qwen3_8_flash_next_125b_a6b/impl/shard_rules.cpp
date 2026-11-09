#include "models/qwen3_8_flash_next_125b_a6b/impl/shard_rules.h"

#include <array>
#include <vector>

namespace ninfer::models::flash_next_125b_a6b {
namespace {

using artifact::ShardAxis;



// Suffixes are relative to one layer's prefix, so the same rows serve the 48 text layers and the
// MTP layer. The extents are the fixed 125B-A6B geometry from the model reference.
constexpr std::array<ParameterShard, 23> kParameterShards{{
    // ---- QSA (12 full-attention layers) -------------------------------------------------------
    // Rows by the 24 gated query heads: 12288 = 48 head-slots of 256. See the header's packing note.
    {"self_attn.q_proj.weight", ShardAxis::Rows, {12288}, 1},
    // Rows by the 2 KV heads: 512 = 2 x 256.
    {"self_attn.k_proj.weight", ShardAxis::Rows, {512}, 1},
    {"self_attn.v_proj.weight", ShardAxis::Rows, {512}, 1},
    // Columns by the 24 query heads: 6144 = 24 x 256.
    {"self_attn.o_proj.weight", ShardAxis::Columns, {6144}, 1},
    // Per-head norms are whole on every rank: each rank owns its own heads' norms.
    {"self_attn.q_norm.weight", ShardAxis::Replicated, {256}, 1},
    {"self_attn.k_norm.weight", ShardAxis::Replicated, {256}, 1},
    // The QSA indexer's query/key projection is [640, 2560] = 5 head-slots of 128 (4 query, 1 KV).
    // Its KV slot cannot halve, so the indexer is replicated until its packing is settled.
    {"self_attn.indexer.index_qk_proj.weight", ShardAxis::Replicated, {640}, 1},
    {"self_attn.indexer.q_layernorm.weight", ShardAxis::Replicated, {128}, 1},
    {"self_attn.indexer.k_layernorm.weight", ShardAxis::Replicated, {128}, 1},

    // ---- GDN (36 linear-attention layers) -----------------------------------------------------
    // Rows by the 48 value heads.
    {"linear_attn.A_log", ShardAxis::Rows, {48}, 1},
    {"linear_attn.dt_bias", ShardAxis::Rows, {48}, 1},
    {"linear_attn.in_proj_a.weight", ShardAxis::Rows, {48}, 1},
    {"linear_attn.in_proj_b.weight", ShardAxis::Rows, {48}, 1},
    // Columns by the 10240 Q|K|V convolution channels: 10240 = 80 head-slots of 128.
    {"linear_attn.conv1d.weight", ShardAxis::Columns, {10240}, 1},
    // Rows by the 80 Q|K|V head-slots: 16 Q, 16 K, 48 V. See the header's packing note.
    {"linear_attn.in_proj_qkv.weight", ShardAxis::Rows, {2048, 2048, 6144}, 3},
    // Rows by the 48 value heads: 6144 = 48 x 128.
    {"linear_attn.in_proj_z.weight", ShardAxis::Rows, {6144}, 1},
    {"linear_attn.norm.weight", ShardAxis::Replicated, {128}, 1},
    // Columns by the 48 value heads.
    {"linear_attn.out_proj.weight", ShardAxis::Columns, {6144}, 1},

    // ---- HyperConnection (per layer, and the final mixer) -------------------------------------
    // The four residual streams are whole on every rank.
    {"attn_hyper_connection.block_inject_weight.weight", ShardAxis::Replicated, {40960}, 1},
    {"attn_hyper_connection.hc_norm.weight", ShardAxis::Replicated, {10240}, 1},
    // The mix contracts over the 10240 residual width, so its columns shard and its rows do not.
    {"attn_hyper_connection.input_mix_weight_down.weight", ShardAxis::Columns, {10240}, 1},
    {"attn_hyper_connection.input_mix_weight_up.weight", ShardAxis::Rows, {10240}, 1},

    // ---- MoE ----------------------------------------------------------------------------------
    // The router is whole on every rank: each rank routes its own tokens and must see all 512
    // experts plus the shared-gate row, whatever experts it actually holds.
    {"mlp.gate.weight", ShardAxis::Replicated, {513}, 1},
}};

// The comment block above the QKV row is where the segmentation came from; keep it adjacent.
} // namespace

std::int64_t total_extent(const ParameterShard& shard) noexcept {
    std::int64_t total = 0;
    for (std::uint8_t index = 0; index < shard.segment_count; ++index) {
        total += shard.segments[index];
    }
    return total;
}

std::vector<artifact::ShardRange> ranges_for(const ParameterShard& shard, int width, int rank) {
    std::vector<artifact::ShardRange> ranges;
    ranges.reserve(shard.segment_count);
    std::int64_t offset = 0;
    for (std::uint8_t index = 0; index < shard.segment_count; ++index) {
        const std::int64_t segment = shard.segments[index];
        const auto placement = artifact::split_evenly(shard.axis, segment, width);
        const auto& range    = placement.ranges[static_cast<std::size_t>(rank)];
        ranges.push_back(artifact::ShardRange{offset + range.begin, range.extent});
        offset += segment;
    }
    return ranges;
}

std::span<const ParameterShard> parameter_shards() noexcept {
    return kParameterShards;
}

} // namespace ninfer::models::flash_next_125b_a6b
