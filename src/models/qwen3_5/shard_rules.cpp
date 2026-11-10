#include "models/qwen3_5/shard_rules.h"

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5 {
namespace {

using artifact::ShardAxis;

// Suffixes are relative to one layer's prefix, so the same rows serve the 16 full-attention layers,
// the 48 GDN layers and the MTP layer. The extents are the fixed Qwen3.8-27B geometry.
constexpr std::array<ParameterShard, 30> kParameterShards{{
    // ---- Full attention (16 layers, layer_types[i] == full_attention) --------------------------
    // Four separate logical parameters, packed per rank at execution time, so a contiguous half of
    // each gives a rank 12 complete query heads. See the header's packing note.
    // Rows by the 24 query heads: 6144 = 24 x 256.
    {"attention/query", ShardAxis::Rows, {6144}, 1},
    // Rows by the 4 KV heads: 1024 = 4 x 256.
    {"attention/key", ShardAxis::Rows, {1024}, 1},
    {"attention/value", ShardAxis::Rows, {1024}, 1},
    // The gate is a second query-width projection, so it halves the same way.
    {"attention/gate", ShardAxis::Rows, {6144}, 1},
    // Columns by the 24 query heads: 6144 = 24 x 256.
    {"attention/output", ShardAxis::Columns, {6144}, 1},
    // Per-head norms are whole on every rank: each rank owns its own heads' norms.
    {"attention/query_norm", ShardAxis::Replicated, {256}, 1},
    {"attention/key_norm", ShardAxis::Replicated, {256}, 1},

    // ---- GDN (48 linear-attention layers) ------------------------------------------------------
    // Rows by the 16 key heads of width 128.
    {"gdn/query", ShardAxis::Rows, {2048}, 1},
    {"gdn/key", ShardAxis::Rows, {2048}, 1},
    // Rows by the 48 value heads of width 128.
    {"gdn/value", ShardAxis::Rows, {6144}, 1},
    {"gdn/z", ShardAxis::Rows, {6144}, 1},
    // Rows by the 48 value heads: the per-head decay and step tensors.
    {"gdn/a_projection", ShardAxis::Rows, {48}, 1},
    {"gdn/b_projection", ShardAxis::Rows, {48}, 1},
    {"gdn/a_log", ShardAxis::Rows, {48}, 1},
    {"gdn/dt_bias", ShardAxis::Rows, {48}, 1},
    // Columns by the 10240 depthwise channels, in Q|K|V order: three segments that shard
    // independently. A contiguous half would mix the roles. See the header's packing note.
    {"gdn/convolution", ShardAxis::Columns, {2048, 2048, 6144}, 3},
    {"gdn/norm", ShardAxis::Replicated, {128}, 1},
    // Columns by the 48 value heads: 6144 = 48 x 128.
    {"gdn/output", ShardAxis::Columns, {6144}, 1},

    // ---- Per-layer norms ----------------------------------------------------------------------
    // The residual width is whole on every rank.
    {"input_norm", ShardAxis::Replicated, {5120}, 1},
    {"post_attention_norm", ShardAxis::Replicated, {5120}, 1},

    // ---- Dense MLP (every layer) --------------------------------------------------------------
    // Gate and up by their 17408 intermediate rows, down by its 17408 intermediate columns, and
    // each rank's partial output is summed by the all-reduce.
    {"mlp/gate", ShardAxis::Rows, {17408}, 1},
    {"mlp/up", ShardAxis::Rows, {17408}, 1},
    {"mlp/down", ShardAxis::Columns, {17408}, 1},

    // ---- Embedding and output head ------------------------------------------------------------
    // The embedding is replicated: every rank gathers any token id, so a vocabulary split would
    // leave a lookup outside the local slice with no row.
    {"text/token_embedding", ShardAxis::Replicated, {248320LL * 5120}, 1},
    // Rows by the 248320-entry vocabulary: 124160 each at width 2.
    {"text/output_head", ShardAxis::Rows, {248320}, 1},
    {"text/final_norm", ShardAxis::Replicated, {5120}, 1},

    // ---- MTP layer ----------------------------------------------------------------------------
    // The MTP block is bound with the target's geometry (load/mtp.cpp:14), so its attention and MLP
    // rows above already match it; these are the parts that only the MTP layer has.
    // The stem maps [embedding | hidden] to the residual width, which must stay whole per rank.
    {"mtp/input_projection", ShardAxis::Replicated, {5120LL * 10240}, 1},
    {"mtp/embedding_norm", ShardAxis::Replicated, {5120}, 1},
    {"mtp/hidden_norm", ShardAxis::Replicated, {5120}, 1},
    {"mtp/final_norm", ShardAxis::Replicated, {5120}, 1},
}};

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
        const auto placement       = artifact::split_evenly(shard.axis, segment, width);
        const auto& range          = placement.ranges[static_cast<std::size_t>(rank)];
        ranges.push_back(artifact::ShardRange{offset + range.begin, range.extent});
        offset += segment;
    }
    return ranges;
}

bool in_scope(std::string_view name) noexcept {
    // The draft components reuse these suffixes with their own head counts, so the table owns only
    // the target's own text and MTP parameters.
    return name.starts_with("text/") || name.starts_with("mtp/");
}

namespace {

// The table row whose suffix is the tail of `name`. Parameters are bound with a layer prefix, so
// the match is on the suffix; `in_scope` keeps a draft component from matching a text row.
const ParameterShard* shard_for(std::string_view name) noexcept {
    if (!in_scope(name)) { return nullptr; }
    for (const ParameterShard& shard : kParameterShards) {
        if (name.size() < shard.suffix.size()) { continue; }
        if (name.substr(name.size() - shard.suffix.size()) == shard.suffix) { return &shard; }
    }
    return nullptr;
}

} // namespace

artifact::ParameterReference shard_parameter(const artifact::ParameterReference& reference,
                                             int width, int rank) {
    if (width <= 1) { return reference; }
    const ParameterShard* shard = shard_for(reference.name);
    if (shard == nullptr) { return reference; } // Replicated: not in the table, or out of scope.

    const auto ranges = ranges_for(*shard, width, rank);
    // The table declares the extent along the sharded axis, which is a row or column count for a
    // split and the whole element count for a placement that does not split. `binding.elements` is
    // always the TOTAL element count, so comparing it against a row count is a unit mismatch: for
    // text/output_head the binding is 248320 * 5120 elements against a declared 248320 rows. The
    // synthetic tests never caught it because they set `elements` to the extent by hand.
    {
        const std::uint64_t declared = static_cast<std::uint64_t>(total_extent(*shard));
        std::uint64_t bound          = reference.binding.elements;
        if (shard->axis == artifact::ShardAxis::Rows ||
            shard->axis == artifact::ShardAxis::Columns) {
            const std::size_t axis_index = shard->axis == artifact::ShardAxis::Rows ? 0 : 1;
            if (reference.shape.size() <= axis_index) {
                throw std::invalid_argument("sharding: " + std::string(reference.name) +
                                            " has no axis to shard");
            }
            bound = reference.shape[axis_index];
        }
        if (declared != bound) {
            throw std::invalid_argument("sharding: " + std::string(reference.name) + " binds " +
                                        std::to_string(bound) +
                                        " along its sharded axis but the table declares " +
                                        std::to_string(declared));
        }
    }

    artifact::ParameterReference sharded = reference;

    // `ranges_for` returns ranges along the SHARDED AXIS, but a Binding's parts are in the
    // parameter's flat logical element order, which is row-major: a block of rows is one contiguous
    // element range, while a block of columns is one short range per row. Passing an axis-space
    // range straight to `shard_binding_ranges` therefore works for Rows and silently takes a
    // contiguous slab of rows for Columns — the first real width-2 load failed its coverage check
    // exactly there.
    std::vector<artifact::ShardRange> logical;
    if (shard->axis == artifact::ShardAxis::Rows) {
        const std::uint64_t columns = reference.shape.size() > 1 ? reference.shape[1] : 1;
        for (const auto& range : ranges) {
            logical.push_back({static_cast<std::uint64_t>(range.begin) * columns,
                               static_cast<std::uint64_t>(range.extent) * columns});
        }
    } else if (shard->axis == artifact::ShardAxis::Columns) {
        const std::uint64_t rows    = reference.shape[0];
        const std::uint64_t columns = reference.shape[1];
        for (const auto& range : ranges) {
            for (std::uint64_t row = 0; row < rows; ++row) {
                logical.push_back({row * columns + static_cast<std::uint64_t>(range.begin),
                                   static_cast<std::uint64_t>(range.extent)});
            }
        }
    } else {
        for (const auto& range : ranges) {
            logical.push_back({static_cast<std::uint64_t>(range.begin),
                               static_cast<std::uint64_t>(range.extent)});
        }
    }
    sharded.binding = artifact::shard_binding_ranges(reference.binding, logical);

    std::uint64_t share = 0;
    for (const auto& range : ranges) { share += static_cast<std::uint64_t>(range.extent); }

    // Only a split changes the shape. Replicated keeps every rank's shape whole.
    if (shard->axis == artifact::ShardAxis::Rows || shard->axis == artifact::ShardAxis::Columns) {
        const std::size_t axis_index = shard->axis == artifact::ShardAxis::Rows ? 0 : 1;
        if (sharded.shape.size() <= axis_index) {
            throw std::invalid_argument("sharding: " + std::string(reference.name) +
                                        " has no axis to shard");
        }
        sharded.shape[axis_index] = share;
    }
    return sharded;
}

std::span<const ParameterShard> parameter_shards() noexcept { return kParameterShards; }

artifact::Binder::ShardTransform shard_transform(int width, int rank) {
    if (width <= 1) { return {}; }
    if (rank < 0 || rank >= width) {
        throw std::invalid_argument("sharding: rank " + std::to_string(rank) + " outside width " +
                                    std::to_string(width));
    }
    return [width, rank](const artifact::ParameterReference& reference) {
        return shard_parameter(reference, width, rank);
    };
}

} // namespace ninfer::models::qwen3_5
