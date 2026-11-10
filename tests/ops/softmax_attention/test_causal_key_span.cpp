// test_causal_key_span.cpp — host-only gate on the causal decode key-span bound.
//
// WHY THIS EXISTS: the visible-keys ceiling is 1,048,576 and the split count is clamped, so a
// split's key span is what grows when the ceiling rises. Nothing in the kernels bounds that span
// directly; the bound is a chain of integer facts that hold only if every link holds:
//
//   * the plan's `capacity` must be at least the split count `live(visible)` derives at runtime,
//     or the partial workspace is short;
//   * the key tile must divide the KV page, or a tile straddles a logical page and the kernel
//     reads the next page's keys from the previous page's physical address;
//   * every index derived from the span must fit the `int` the kernels use for it.
//
// This test pins all three for every registered geometry at both ceilings. It calls the real plan
// selectors rather than re-deriving their arithmetic, so a change to the planning rule moves these
// numbers instead of silently diverging from them.

#include "ops/softmax_attention/dense/causal_cache/bf16/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {

using ninfer::ops::CausalAttentionExecutionEnvelope;
using ninfer::ops::detail::Bf16KvPartition;
using ninfer::ops::detail::CausalKvPartition;

constexpr std::int32_t kInt32Max = std::numeric_limits<std::int32_t>::max();
// The two production devices' physical SM counts: RTX 5090 (sm_120a) and GB10 (sm_121a).
constexpr int kSmCounts[] = {170, 72};
constexpr std::uint32_t kOldCeiling = 262144;
constexpr std::uint32_t kNewCeiling = 1048576;

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("  [FAIL] %s\n", what);
        ++failures;
    }
}

// A split's tiles are read with one page-table lookup per tile, so a tile that spans two logical
// pages would read the wrong physical page for its tail keys. That cannot happen when the key tile
// divides the page, and it is the reason `key_rows` is a multiple-of-32 constant rather than an
// arbitrary tile height.
void check_tile_divides_page(int key_rows, const char* label) {
    if (key_rows <= 0 || ninfer::kPagedKVPageSize % key_rows != 0) {
        std::printf("  [FAIL] %s: key tile %d does not divide the %d-key page\n", label, key_rows,
                    ninfer::kPagedKVPageSize);
        ++failures;
    }
}

void check_bf16(const CausalAttentionExecutionEnvelope& envelope, int heads, int width, int batch,
                int sms) {
    const auto plan =
        ninfer::ops::detail::select_bf16_kv_causal_plan(heads, width, batch, envelope, sms);
    const auto& partition = plan.partition;
    const auto description =
        ninfer::ops::detail::bf16_kv_instance_description(plan.instance);
    check_tile_divides_page(description.key_rows, "bf16");

    // The widest row the envelope allows is the one the workspace was sized for.
    const auto widest = partition.live(static_cast<int>(envelope.max_visible_keys));
    check(partition.capacity >= widest.splits,
          "bf16 capacity covers the split count at the widest row");
    check(partition.capacity <= 256, "bf16 capacity fits the merge reduction block");
    check(widest.keys_per_split % description.key_rows == 0,
          "bf16 keys per split is a whole number of key tiles");
    check(static_cast<std::int64_t>(widest.keys_per_split) * widest.splits >=
              envelope.max_visible_keys,
          "bf16 splits cover the widest row");
    // The kernels hold the split's key start in an `int`.
    check(static_cast<std::int64_t>(widest.splits - 1) * widest.keys_per_split <= kInt32Max,
          "bf16 split start fits int32");
    std::printf("  bf16 heads=%2d w=%3d b=%d sms=%3d ceiling=%7u: capacity=%3d splits=%3d "
                "keys/split=%6d tiles/split=%4d\n",
                heads, width, batch, sms, envelope.max_visible_keys, partition.capacity,
                widest.splits, widest.keys_per_split,
                widest.keys_per_split / description.key_rows);
}

void check_int8(const CausalAttentionExecutionEnvelope& envelope, int heads, int width, int batch,
                int sms) {
    const auto plan =
        ninfer::ops::detail::select_int8_kv_causal_plan(heads, width, batch, envelope, sms);
    const auto& partition = plan.partition;
    // The grouped kernels split the visible range into equal key tiles; the tiled family reads the
    // table once per tile. Both use a tile that divides the page.
    check_tile_divides_page(ninfer::ops::detail::Int8KvCausalPlan::kTokenTile, "int8 token tile");
    check(partition.capacity >= 1 && partition.capacity <= CausalKvPartition::kMaxSplits,
          "int8 capacity fits the merge reduction block");
    check(partition.capacity == partition.active(static_cast<int>(envelope.max_visible_keys)),
          "int8 capacity is the active split count at the widest row");
    check(partition.key_shift >= 6 && partition.key_shift <= 12,
          "int8 key_shift keeps the minimum keys per split in [64,4096]");
    // The grouped kernel forms `split * logical_tiles` and `(split+1) * logical_tiles` in int,
    // where logical_tiles = div_up(window, Bc) and Bc is the schedule's key tile (32 or 64). 32 is
    // the worst case: it maximises the tile count and therefore the product.
    constexpr std::int64_t kWorstKeyTile = 32;
    const std::int64_t tiles = static_cast<std::int64_t>(envelope.max_visible_keys) / kWorstKeyTile +
                               (envelope.max_visible_keys % kWorstKeyTile != 0 ? 1 : 0);
    check(tiles * CausalKvPartition::kMaxSplits <= kInt32Max,
          "int8 split * logical_tiles fits int32");
    const std::int64_t active = partition.active(static_cast<int>(envelope.max_visible_keys));
    std::printf("  int8 heads=%2d w=%3d b=%d sms=%3d ceiling=%7u: capacity=%3d target=%3d "
                "shift=%d keys/split>=%d\n",
                heads, width, batch, sms, envelope.max_visible_keys, partition.capacity,
                partition.target, partition.key_shift, 1 << partition.key_shift);
    check(active >= 1, "int8 active split count is positive");
}

// The envelope only bounds the WIDEST row; other batch rows are shorter, and a shorter row is not
// automatically a smaller split count because `live()` re-rounds keys per split against the row
// length. The workspace is sized once from `capacity`, so the real invariant is that no row length
// in the envelope produces more splits than that. Scanning the whole range is the only way to know
// that without re-deriving the rounding rule.
void check_bf16_covers_every_row(const CausalAttentionExecutionEnvelope& envelope, int heads,
                                 int width, int batch, int sms) {
    const auto plan =
        ninfer::ops::detail::select_bf16_kv_causal_plan(heads, width, batch, envelope, sms);
    const auto& partition = plan.partition;
    int worst            = 0;
    int worst_visible    = 0;
    for (std::uint32_t visible = 1; visible <= envelope.max_visible_keys; ++visible) {
        const int splits = partition.live(static_cast<int>(visible)).splits;
        if (splits > worst) {
            worst         = splits;
            worst_visible = static_cast<int>(visible);
        }
    }
    std::printf("  bf16 heads=%2d w=%3d b=%d sms=%3d ceiling=%7u: capacity=%3d "
                "worst splits over every row=%3d at visible=%d\n",
                heads, width, batch, sms, envelope.max_visible_keys, partition.capacity, worst,
                worst_visible);
    check(worst <= partition.capacity,
          "bf16 capacity covers every row length in the envelope, not only the widest");
}

void check_int8_covers_every_row(const CausalAttentionExecutionEnvelope& envelope, int heads,
                                 int width, int batch, int sms) {
    const auto plan =
        ninfer::ops::detail::select_int8_kv_causal_plan(heads, width, batch, envelope, sms);
    const auto& partition = plan.partition;
    int worst            = 0;
    for (std::uint32_t visible = 1; visible <= envelope.max_visible_keys; ++visible) {
        const int splits = partition.active(static_cast<int>(visible));
        if (splits > worst) { worst = splits; }
    }
    check(worst <= partition.capacity,
          "int8 capacity covers every row length in the envelope, not only the widest");
}

} // namespace

int main() {
    std::printf("causal decode key-span bound (host)\n");
    for (const std::uint32_t ceiling : {kOldCeiling, kNewCeiling}) {
        // A decode call's envelope starts at one visible key; the ceiling is the widest row the
        // captured plan must serve.
        const CausalAttentionExecutionEnvelope envelope{1, ceiling};
        for (const int sms : kSmCounts) {
            for (const int heads : {24, 16}) {
                // Decode widths are the speculative window sizes; a wide prefill width exercises
                // the tiled family.
                for (const int width : {1, 4, 16}) {
                    for (const int batch : {1, 8}) {
                        check_bf16(envelope, heads, width, batch, sms);
                        check_int8(envelope, heads, width, batch, sms);
                    }
                }
            }
        }
    }
    // The full-range scan is what the workspace claim actually rests on, so run it for the decode
    // geometry of both model classes at both ceilings.
    for (const std::uint32_t ceiling : {kOldCeiling, kNewCeiling}) {
        const CausalAttentionExecutionEnvelope envelope{1, ceiling};
        for (const int heads : {24, 16}) {
            check_bf16_covers_every_row(envelope, heads, 1, 1, 170);
            check_int8_covers_every_row(envelope, heads, 1, 1, 170);
        }
    }
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures,
                failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
