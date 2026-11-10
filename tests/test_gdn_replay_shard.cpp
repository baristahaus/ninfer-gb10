// Host-only contract test for shard_gdn_replay_spec: width 1 is the identity, width 2 halves the
// per-head extents and the record payload, and a width that does not divide the head count is
// refused rather than rounded. This is B2c from review/phase3/rank1-arena.md made executable.

#include "core/gdn_replay_records.h"
#include "core/layout.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cerr << "FAIL: " << label << '\n';
    ++failures;
}

template <class Fn>
void expect_throws(Fn&& fn, const std::string& label) {
    try {
        fn();
    } catch (const std::invalid_argument&) { return; }
    std::cerr << "FAIL: " << label << " was accepted\n";
    ++failures;
}

// The 27B's GDN: 16 key heads and 48 value heads of width 128, convolution 2 * 2048 + 6144.
ninfer::GdnReplayRecordSpec spec_27b() {
    ninfer::GdnReplayRecordSpec spec;
    spec.layers          = 48;
    spec.record_capacity = 4;
    spec.width           = 4; // draft window 3 + 1
    spec.conv_channels   = 10240;
    spec.qk_heads        = 16;
    spec.value_heads     = 48;
    spec.key_dim         = 128;
    spec.value_dim       = 128;
    return spec;
}

std::size_t record_bytes(const ninfer::GdnReplayRecordSpec& spec) {
    ninfer::LayoutBuilder builder;
    (void)ninfer::plan_gdn_replay_records(builder, spec);
    return builder.finish(256);
}

bool same_spec(const ninfer::GdnReplayRecordSpec& a, const ninfer::GdnReplayRecordSpec& b) {
    return a.layers == b.layers && a.record_capacity == b.record_capacity && a.width == b.width &&
           a.conv_channels == b.conv_channels && a.qk_heads == b.qk_heads &&
           a.value_heads == b.value_heads && a.key_dim == b.key_dim && a.value_dim == b.value_dim;
}

void test_width_one_is_the_identity() {
    const auto spec  = spec_27b();
    const auto whole = ninfer::shard_gdn_replay_spec(spec, 1, 0);
    expect(same_spec(whole, spec), "width 1 returns every field unchanged");
    expect(record_bytes(whole) == record_bytes(spec),
           "width 1 plans the same record payload as the unsharded spec");
}

void test_width_two_halves() {
    const auto spec  = spec_27b();
    const auto rank0 = ninfer::shard_gdn_replay_spec(spec, 2, 0);
    const auto rank1 = ninfer::shard_gdn_replay_spec(spec, 2, 1);
    expect(same_spec(rank0, rank1), "both ranks see the same per-rank geometry");

    // The three per-head fields halve; the per-head dims and the schedule extents do not.
    expect(rank0.qk_heads == 8, "16 key heads become 8");
    expect(rank0.value_heads == 24, "48 value heads become 24");
    expect(rank0.conv_channels == 5120, "10240 convolution channels become 5120");
    expect(rank0.key_dim == 128 && rank0.value_dim == 128, "the per-head dims stay whole");
    expect(rank0.layers == spec.layers && rank0.record_capacity == spec.record_capacity &&
               rank0.width == spec.width,
           "layers, record capacity and the draft window stay whole");

    // The ranks' heads partition the whole, and the per-rank channels are the per-rank heads'.
    expect(rank0.qk_heads * 2 == spec.qk_heads, "the two ranks' key heads partition the whole");
    expect(rank0.value_heads * 2 == spec.value_heads,
           "the two ranks' value heads partition the whole");
    expect(rank0.conv_channels ==
               2 * rank0.qk_heads * rank0.key_dim + rank0.value_heads * rank0.value_dim,
           "the per-rank convolution channels match the per-rank heads");

    // The payload follows: half the bytes.
    expect(record_bytes(rank0) * 2 == record_bytes(spec), "the record payload halves at width 2");
}

void test_wide_heads_divide_and_narrow_refuse() {
    // 16 key heads: width 3 and width 5 are refused, not rounded.
    expect_throws([] { (void)ninfer::shard_gdn_replay_spec(spec_27b(), 3, 0); },
                  "16 key heads at width 3");
    expect_throws([] { (void)ninfer::shard_gdn_replay_spec(spec_27b(), 5, 0); },
                  "16 key heads at width 5");

    // Whole heads, not just a dividing width: with a 3-wide head, the derived key width (16 x 3 =
    // 48) divides by 3 while the 16 heads do not, so only the head check refuses it. Without that
    // check the arithmetic would truncate to 5 heads and the consistency check would fail later.
    ninfer::GdnReplayRecordSpec narrow_heads = spec_27b();
    narrow_heads.key_dim                     = 3;
    narrow_heads.value_dim                   = 3;
    narrow_heads.conv_channels               = 2 * 16 * 3 + 48 * 3;
    expect_throws([&] { (void)ninfer::shard_gdn_replay_spec(narrow_heads, 3, 0); },
                  "16 heads of width 3 at width 3");

    // A 6-key / 48-value geometry does divide by 3, so the refusal above is about divisibility and
    // not about the width itself.
    ninfer::GdnReplayRecordSpec wide = spec_27b();
    wide.qk_heads                    = 6;
    wide.value_heads                 = 48;
    wide.conv_channels               = 2 * 6 * 128 + 48 * 128; // 7680
    const auto third                 = ninfer::shard_gdn_replay_spec(wide, 3, 2);
    expect(third.qk_heads == 2 && third.value_heads == 16 && third.conv_channels == 2560,
           "a 6-key/48-value geometry splits at width 3");
    expect(third.conv_channels ==
               2 * third.qk_heads * third.key_dim + third.value_heads * third.value_dim,
           "the width-3 per-rank channels match the per-rank heads");
    expect(record_bytes(third) * 3 == record_bytes(wide),
           "the record payload is a third at width 3");
}

void test_invalid_arguments() {
    expect_throws([] { (void)ninfer::shard_gdn_replay_spec(spec_27b(), 0, 0); }, "width 0");
    expect_throws([] { (void)ninfer::shard_gdn_replay_spec(spec_27b(), -2, 0); }, "negative width");
    expect_throws([] { (void)ninfer::shard_gdn_replay_spec(spec_27b(), 2, 2); }, "rank == width");
    expect_throws([] { (void)ninfer::shard_gdn_replay_spec(spec_27b(), 2, -1); }, "negative rank");

    auto no_heads     = spec_27b();
    no_heads.qk_heads = 0;
    expect_throws([&] { (void)ninfer::shard_gdn_replay_spec(no_heads, 2, 0); }, "zero key heads");

    auto no_channels          = spec_27b();
    no_channels.conv_channels = 0;
    expect_throws([&] { (void)ninfer::shard_gdn_replay_spec(no_channels, 2, 0); },
                  "zero convolution channels");

    // A spec whose channels do not match its heads is rejected, not halved into an inconsistent
    // geometry.
    auto broken          = spec_27b();
    broken.conv_channels = 10000;
    expect_throws([&] { (void)ninfer::shard_gdn_replay_spec(broken, 2, 0); },
                  "a convolution count that is not 2*key_width + value_width");
}

} // namespace

int main() {
    test_width_one_is_the_identity();
    test_width_two_halves();
    test_wide_heads_divide_and_narrow_refuse();
    test_invalid_arguments();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "gdn replay shard: ok\n";
    return 0;
}
