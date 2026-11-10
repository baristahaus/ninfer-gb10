// Host-only contract test for the Qwen3.5 family's 27B shard table: every parameter must split
// cleanly at width 2, the hostile extents must refuse width 3, the GDN convolution must shard as
// three role segments rather than one contiguous block, and the draft components must not match the
// text rows. This is the review's head-count finding made executable.

#include "artifact/binder.h"
#include "artifact/shard.h"
#include "models/qwen3_5/shard_rules.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::artifact::ShardAxis;
using ninfer::artifact::ShardRange;
using ninfer::artifact::split_evenly;
using ninfer::models::qwen3_5::in_scope;
using ninfer::models::qwen3_5::parameter_shards;
using ninfer::models::qwen3_5::ranges_for;
using ninfer::models::qwen3_5::shard_parameter;
using ninfer::models::qwen3_5::shard_transform;
using ninfer::models::qwen3_5::total_extent;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

bool splits_at(const ShardAxis axis, const std::int64_t extent, const int width) {
    try {
        const auto placement = split_evenly(axis, extent, width);
        return placement.ranges.size() == static_cast<std::size_t>(width);
    } catch (const std::invalid_argument&) { return false; }
}

const ninfer::models::qwen3_5::ParameterShard* row(std::string_view suffix) {
    for (const auto& shard : parameter_shards()) {
        if (shard.suffix == suffix) { return &shard; }
    }
    return nullptr;
}

int test_width_two_is_complete() {
    int failures = 0;
    for (const auto& shard : parameter_shards()) {
        // Per segment, not just the total: a segmented parameter whose segments divide while the
        // total does not (or the reverse) is the case a total-only check misses.
        for (std::uint8_t index = 0; index < shard.segment_count; ++index) {
            failures += check(splits_at(shard.axis, shard.segments[index], 2),
                              std::string(shard.suffix) + " segment " + std::to_string(index) +
                                  " must split at width 2");
        }
    }
    failures += check(!parameter_shards().empty(), "the shard table is not empty");
    return failures;
}

int test_width_three_is_partial() {
    // The rows the review measured as hostile at width 3, and the ones that still divide.
    struct Expected {
        const char* suffix;
        bool splits;
    };

    const Expected expected[] = {
        {"attention/query", true},   // 6144 = 24 x 256 divides by 3
        {"attention/key", false},    // 1024 = 4 KV heads
        {"attention/value", false},  // 1024 = 4 KV heads
        {"attention/gate", true},    // 6144 divides by 3
        {"attention/output", true},  // 6144 divides by 3
        {"gdn/query", false},        // 2048 = 16 key heads
        {"gdn/key", false},          // 2048 = 16 key heads
        {"gdn/value", true},         // 6144 = 48 x 128
        {"gdn/z", true},             // 6144
        {"gdn/a_projection", true},  // 48 divides by 3
        {"gdn/a_log", true},         // 48
        {"gdn/dt_bias", true},       // 48
        {"gdn/convolution", false},  // 2048 | 2048 | 6144: the Q and K segments do not
        {"gdn/output", true},        // 6144
        {"mlp/gate", false},         // 17408 = 2^10 x 17
        {"mlp/up", false},           // 17408
        {"mlp/down", false},         // 17408
        {"text/output_head", false}, // 248320 = 2^9 x 5 x 97
    };
    int failures = 0;
    for (const auto& item : expected) {
        const auto* shard = row(item.suffix);
        failures += check(shard != nullptr, std::string(item.suffix) + " is in the shard table");
        if (shard == nullptr) { continue; }
        bool splits = true;
        for (std::uint8_t index = 0; index < shard->segment_count; ++index) {
            splits = splits && splits_at(shard->axis, shard->segments[index], 3);
        }
        failures += check(splits == item.splits, std::string(item.suffix) + " at width 3 must " +
                                                     (item.splits ? "split" : "refuse"));
    }
    return failures;
}

int test_convolution_segmentation() {
    int failures = 0;
    // The convolution's 10240 depthwise channels are Q 2048 | K 2048 | V 6144. A contiguous half
    // would give rank 0 all of Q and K plus a quarter of V, and rank 1 the rest of V with no Q or
    // K, so the row is three segments that shard independently.
    const auto* conv = row("gdn/convolution");
    failures += check(conv != nullptr, "the GDN convolution is in the table");
    if (conv == nullptr) { return failures; }
    failures += check(conv->axis == ShardAxis::Columns, "the convolution shards its channel axis");
    failures += check(conv->segment_count == 3, "the convolution has three segments");
    failures +=
        check(total_extent(*conv) == 10240, "the convolution segments cover 10240 channels");
    failures +=
        check(conv->segments[0] == 2048 && conv->segments[1] == 2048 && conv->segments[2] == 6144,
              "the segments are Q 2048, K 2048, V 6144");

    // At width 2 each rank owns half of each segment: 1024 + 1024 + 3072.
    const auto rank0 = ranges_for(*conv, 2, 0);
    const auto rank1 = ranges_for(*conv, 2, 1);
    failures += check(rank0.size() == 3 && rank1.size() == 3, "each rank owns three blocks");
    failures += check(rank0[0] == ShardRange{0, 1024} && rank0[1] == ShardRange{2048, 1024} &&
                          rank0[2] == ShardRange{4096, 3072},
                      "rank 0 takes the first half of Q, K and V");
    failures += check(rank1[0] == ShardRange{1024, 1024} && rank1[1] == ShardRange{3072, 1024} &&
                          rank1[2] == ShardRange{7168, 3072},
                      "rank 1 takes the second half of Q, K and V");

    // A single-segment parameter still yields one contiguous block.
    const auto* key = row("attention/key");
    failures += check(key != nullptr, "the KV projection is in the table");
    if (key != nullptr) {
        const auto r0 = ranges_for(*key, 2, 0);
        failures += check(r0.size() == 1 && r0[0] == ShardRange{0, 512},
                          "the KV projection is one block of 2 heads per rank");
    }
    return failures;
}

// A reference whose binding covers `extent` elements — the parameter's sharded extent, which is the
// row count for a Rows parameter and the column count for a Columns one.
ninfer::artifact::ParameterReference synthetic(std::string name, std::vector<std::uint64_t> shape,
                                               std::uint64_t /*extent*/) {
    ninfer::artifact::ParameterReference reference;
    reference.name  = std::move(name);
    reference.shape = std::move(shape);
    // A Binding's element count is the product of the shape, and its parts tile the whole
    // parameter. Setting it to the sharded extent made this fixture agree with the extent check's
    // old wrong assumption — which is why a real artifact found that bug and the test did not.
    std::uint64_t elements = 1;
    for (const auto dim : reference.shape) { elements *= dim; }
    reference.binding.elements = elements;
    reference.binding.parts.push_back(
        ninfer::artifact::Part{ninfer::artifact::ObjectHandle{0}, 0, elements});
    return reference;
}

int test_shard_parameter() {
    int failures = 0;

    // A Rows parameter halves its first axis and its binding: 24 query heads become 12.
    const auto query =
        shard_parameter(synthetic("text/layers/3/attention/query", {6144, 5120}, 6144), 2, 0);
    failures += check(query.shape[0] == 3072 && query.shape[1] == 5120,
                      "the query projection halves its rows and keeps its columns");
    failures += check(query.binding.elements == 3072 * 5120, "the query binding halves");
    failures += check(query.binding.parts.size() == 1 && query.binding.parts[0].begin == 0 &&
                          query.binding.parts[0].end == 3072 * 5120,
                      "rank 0 owns the first 12 query heads");

    // The KV projection halves by KV heads: 4 become 2.
    const auto key =
        shard_parameter(synthetic("text/layers/3/attention/key", {1024, 5120}, 1024), 2, 1);
    failures += check(key.shape[0] == 512, "the KV projection halves to 2 heads");
    failures += check(key.binding.parts.size() == 1 && key.binding.parts[0].begin == 512 * 5120 &&
                          key.binding.parts[0].end == 1024 * 5120,
                      "rank 1 owns the second pair of KV heads");

    // A Columns parameter halves its second axis.
    const auto out =
        shard_parameter(synthetic("text/layers/3/attention/output", {5120, 6144}, 6144), 2, 1);
    failures += check(out.shape[0] == 5120 && out.shape[1] == 3072,
                      "the output projection halves its columns and keeps its rows");

    // A segmented parameter halves each segment: 10240 columns become 5120, in three slices.
    const auto conv =
        shard_parameter(synthetic("text/layers/4/gdn/convolution", {4, 10240}, 10240), 2, 0);
    failures += check(conv.shape[0] == 4 && conv.shape[1] == 5120,
                      "the convolution keeps its taps and halves its channels");
    // A column shard is one short range PER ROW, so three segments over four taps give twelve
    // slices rather than three. The row-major logical order is why: a block of columns is not
    // contiguous, which is the defect the first real width-2 load found.
    failures += check(conv.binding.parts.size() == 4 * 3,
                      "the convolution binding is one slice per tap per segment");
    failures += check(conv.binding.elements == 5120 * 4, "the convolution binding totals the share");

    // A Replicated parameter is untouched.
    const auto norm = shard_parameter(synthetic("text/layers/3/input_norm", {5120}, 5120), 2, 1);
    failures += check(norm.shape[0] == 5120 && norm.binding.elements == 5120,
                      "a replicated parameter is unchanged");
    failures += check(norm.binding.parts.size() == 1 && norm.binding.parts[0].end == 5120,
                      "a replicated parameter keeps its whole binding");

    // The embedding is replicated even though it is large.
    const auto embedding = shard_parameter(
        synthetic("text/token_embedding", {248320, 5120}, 248320ULL * 5120ULL), 2, 0);
    failures +=
        check(embedding.shape[0] == 248320 && embedding.binding.elements == 248320ULL * 5120ULL,
              "the embedding is replicated: every rank gathers any token id");

    // The output head halves by vocabulary.
    const auto head = shard_parameter(synthetic("text/output_head", {248320, 5120}, 248320), 2, 0);
    failures += check(head.shape[0] == 124160, "the output head halves the vocabulary");

    // Width 1 returns the reference unchanged.
    const auto single =
        shard_parameter(synthetic("text/layers/3/attention/key", {1024, 5120}, 1024), 1, 0);
    failures += check(single.shape[0] == 1024 && single.binding.elements == 1024 * 5120,
                      "width 1 leaves the parameter whole");

    // A binding that disagrees with the table is rejected, not clipped.
    bool threw = false;
    try {
        (void)shard_parameter(synthetic("text/layers/3/attention/key", {512, 5120}, 512), 2, 0);
    } catch (const std::invalid_argument&) { threw = true; }
    failures += check(threw, "a binding that disagrees with the table is rejected");
    return failures;
}

// The draft components bind block parameters with the same suffixes but their own geometry
// (32 query / 8 KV heads of width 128). The scope rule keeps them out of the table, so they pass
// through whole instead of throwing on a 27B extent.
int test_scope_rule() {
    int failures = 0;
    failures += check(in_scope("text/layers/3/attention/query"), "text parameters are in scope");
    failures += check(in_scope("mtp/layers/0/attention/query"), "MTP parameters are in scope");
    failures +=
        check(!in_scope("dflash/layers/0/attention/query"), "the DFlash drafter is out of scope");
    failures +=
        check(!in_scope("dflash2/layers/0/attention/query"), "the DFlash2 drafter is out of scope");
    failures += check(!in_scope("vision/blocks/0/attention/query"), "Vision is out of scope");

    const auto draft =
        shard_parameter(synthetic("dflash/layers/0/attention/query", {4096, 5120}, 4096), 2, 0);
    failures += check(draft.shape[0] == 4096 && draft.binding.elements == 4096 * 5120,
                      "a draft parameter with a colliding suffix passes through unchanged");
    return failures;
}

// The binder transform the loader installs per rank: width 1 gives an empty function, width 2 gives
// a callable that slices as `shard_parameter` does, and a rank outside the width is refused.
int test_shard_transform_factory() {
    int failures = 0;
    failures += check(!shard_transform(1, 0), "width 1 returns an empty transform");
    failures +=
        check(static_cast<bool>(shard_transform(2, 0)), "width 2 rank 0 returns a transform");
    failures +=
        check(static_cast<bool>(shard_transform(2, 1)), "width 2 rank 1 returns a transform");

    const auto reference = synthetic("text/layers/3/attention/key", {1024, 5120}, 1024);
    const auto rank0     = shard_transform(2, 0)(reference);
    const auto rank1     = shard_transform(2, 1)(reference);
    failures += check(rank0.shape[0] == 512 && rank0.binding.parts[0].begin == 0 &&
                          rank0.binding.parts[0].end == 512 * 5120,
                      "rank 0's transform owns the first half");
    failures += check(rank1.shape[0] == 512 && rank1.binding.parts[0].begin == 512 * 5120 &&
                          rank1.binding.parts[0].end == 1024 * 5120,
                      "rank 1's transform owns the second half");
    failures += check(rank0.shape[0] + rank1.shape[0] == reference.shape[0],
                      "the two ranks' transforms partition the parameter");

    // An unlisted parameter is Replicated and the transform must not touch it.
    const auto whole = synthetic("text/layers/3/moe/router", {256, 5120}, 256);
    const auto kept  = shard_transform(2, 1)(whole);
    failures += check(kept.shape[0] == 256 && kept.binding.elements == 256 * 5120,
                      "an unlisted parameter passes through the transform unchanged");

    bool threw = false;
    try {
        (void)shard_transform(2, 2);
    } catch (const std::invalid_argument&) { threw = true; }
    failures += check(threw, "a rank outside the width is refused");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_width_two_is_complete();
    failures += test_width_three_is_partial();
    failures += test_convolution_segmentation();
    failures += test_shard_parameter();
    failures += test_scope_rule();
    failures += test_shard_transform_factory();
    if (failures == 0) { std::cout << "ok (" << parameter_shards().size() << " parameters)\n"; }
    return failures == 0 ? 0 : 1;
}
