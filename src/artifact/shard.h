#pragma once

// Tensor-parallel sharding of one logical parameter, in the parameter's own coordinates.
//
// The artifact's stored objects stay opaque: a shard rule says which rows or columns of a logical
// parameter each rank computes with, and the binder maps those ranges through the parameter's
// Binding parts. This header owns the vocabulary and the arithmetic only — which axis a given
// parameter takes is a model decision and lives with the model, because it follows that model's
// head and channel counts.
//
// The split is always contiguous and equal. Equal because unequal per-rank geometry would give the
// ranks different kernel shapes, which nothing downstream can reconcile; contiguous because the
// models' head groupings require it (every query head must stay with its KV head, every GDN value
// head with its key head), and a contiguous block split preserves those groupings exactly when the
// extent divides.

#include <cstdint>
#include <vector>

namespace ninfer::artifact {

// How one logical parameter is placed across the ranks of an execution group.
enum class ShardAxis : std::uint8_t {
    Rows,        // each rank owns a contiguous block of rows (output features)
    Columns,     // each rank owns a contiguous block of columns (input features)
    Replicated,  // every rank owns the whole parameter
    PrimaryOnly, // only rank 0 owns it; every other rank holds none
};

// One rank's block of a sharded parameter: `extent` units starting at `begin`. An empty block has
// extent 0, which is how a rank that holds none of a PrimaryOnly parameter is described.
struct ShardRange {
    std::int64_t begin  = 0;
    std::int64_t extent = 0;

    [[nodiscard]] bool empty() const noexcept { return extent == 0; }
    [[nodiscard]] std::int64_t end() const noexcept { return begin + extent; }
    constexpr bool operator==(const ShardRange&) const noexcept = default;
};

// The per-rank ranges of one logical parameter.
struct ShardPlacement {
    ShardAxis axis = ShardAxis::Replicated;
    std::vector<ShardRange> ranges; // one per rank, rank 0 first
};

// Splits `extent` into `width` contiguous equal ranges along `axis`.
//
// Rows and Columns require `extent % width == 0` and throw std::invalid_argument otherwise: a
// remainder would leave the ranks with different geometries. Replicated gives every rank the whole
// parameter; PrimaryOnly gives rank 0 the whole parameter and every other rank an empty range.
// Throws std::invalid_argument for a width below 1 or a negative extent.
[[nodiscard]] ShardPlacement split_evenly(ShardAxis axis, std::int64_t extent, int width);

} // namespace ninfer::artifact
