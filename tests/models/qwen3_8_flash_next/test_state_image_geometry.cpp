// Host-only contract test for the per-rank StateImage geometry: the spec a rank's StateImage pool
// is planned from. It touches no CUDA — the shard function is pure and the planner is a layout
// builder — so it runs anywhere.

#include <ninfer/models/qwen3_8_flash_next/state_image.h>

#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace q38 = ninfer::models::qwen3_8_flash_next;

namespace {

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

template <typename Exception, typename Fn>
int expect_throws(Fn&& fn, const std::string& label) {
    try {
        fn();
    } catch (const Exception&) { return 0; }
    std::cerr << "FAIL: " << label << " did not throw std::invalid_argument\n";
    return 1;
}

bool same_optional_int(const std::optional<std::int32_t>& left,
                       const std::optional<std::int32_t>& right) {
    return left.has_value() == right.has_value() && (!left || *left == *right);
}

bool same_ple(const std::optional<q38::PleStateSpec>& left,
              const std::optional<q38::PleStateSpec>& right) {
    return left.has_value() == right.has_value() &&
           (!left || (left->channels == right->channels && left->width == right->width));
}

bool same_dflash(const std::optional<q38::DFlashLocalStateSpec>& left,
                 const std::optional<q38::DFlashLocalStateSpec>& right) {
    return left.has_value() == right.has_value() &&
           (!left || (left->layers == right->layers && left->capacity == right->capacity &&
                      left->kv_heads == right->kv_heads && left->head_dim == right->head_dim));
}

bool same_spec(const q38::StateImageSpec& left, const q38::StateImageSpec& right) {
    return left.linear.layers == right.linear.layers &&
           left.linear.conv_channels == right.linear.conv_channels &&
           left.linear.conv_width == right.linear.conv_width &&
           left.linear.value_heads == right.linear.value_heads &&
           left.linear.value_head_dim == right.linear.value_head_dim &&
           left.linear.key_head_dim == right.linear.key_head_dim &&
           left.linear.slot_count == right.linear.slot_count &&
           left.linear.conv_dtype == right.linear.conv_dtype && left.hidden == right.hidden &&
           same_optional_int(left.mtp_hidden, right.mtp_hidden) && same_ple(left.ple, right.ple) &&
           same_dflash(left.dflash_local, right.dflash_local);
}

// The 125B's StateImage spec as `layouts_impl.h:150-176` builds it: 48 GDN value heads of 128,
// 10240 = 2 * 2048 (key) + 6144 (value) convolution channels, 2560 hidden, 4 * hidden MTP hidden,
// PLE 4 * hidden by 9, and a two-head DFlash-local drafter.
q38::StateImageSpec flash_next_spec() {
    q38::StateImageSpec spec;
    spec.linear.layers         = 36;
    spec.linear.conv_channels  = 10240;
    spec.linear.conv_width     = 3;
    spec.linear.value_heads    = 48;
    spec.linear.value_head_dim = 128;
    spec.linear.key_head_dim   = 128;
    spec.linear.slot_count     = 4;
    spec.linear.conv_dtype     = ninfer::DType::BF16;
    spec.hidden                = 2560;
    spec.mtp_hidden            = 4 * 2560;
    spec.ple                   = q38::PleStateSpec{.channels = 4 * 2560, .width = 9};
    spec.dflash_local =
        q38::DFlashLocalStateSpec{.layers = 2, .capacity = 4, .kv_heads = 2, .head_dim = 128};
    return spec;
}

// A spec whose per-head and per-channel extents all divide by three.
q38::StateImageSpec divisible_by_three_spec() {
    q38::StateImageSpec spec;
    spec.linear.layers         = 3;
    spec.linear.conv_channels  = 30; // 2 * 3 (key) + 24 (value)
    spec.linear.conv_width     = 3;
    spec.linear.value_heads    = 6;
    spec.linear.value_head_dim = 4;
    spec.linear.key_head_dim   = 4;
    spec.linear.slot_count     = 2;
    spec.linear.conv_dtype     = ninfer::DType::BF16;
    spec.hidden                = 16;
    return spec;
}

struct Planned {
    q38::StateImageDeviceLayout layout;
    std::size_t bytes = 0;
};

Planned plan(const q38::StateImageSpec& spec) {
    ninfer::LayoutBuilder builder;
    Planned planned;
    planned.layout = q38::plan_state_image_device_pool(builder, spec);
    planned.bytes  = builder.finish(256);
    return planned;
}

} // namespace

int main() {
    int failures = 0;
    const q38::StateImageSpec base = flash_next_spec();

    // Width 1 is the single-device lane: unchanged, field for field, and the planned layout is
    // byte-for-byte the same plan.
    const q38::StateImageSpec identity = q38::shard_state_image_spec(base, 1, 0);
    failures += check(same_spec(identity, base), "width 1 did not return the spec unchanged");
    const Planned whole    = plan(base);
    const Planned identity_plan = plan(identity);
    failures += check(whole.bytes == identity_plan.bytes &&
                          whole.layout.linear.conv.size() == identity_plan.layout.linear.conv.size() &&
                          whole.layout.linear.conv[0].bytes ==
                              identity_plan.layout.linear.conv[0].bytes,
                      "width 1 planned a different StateImage layout");

    // Width 2: exactly three fields shard, and the planned component bytes follow them.
    const q38::StateImageSpec rank0 = q38::shard_state_image_spec(base, 2, 0);
    const q38::StateImageSpec rank1 = q38::shard_state_image_spec(base, 2, 1);
    failures += check(same_spec(rank0, rank1), "the two ranks of an even split disagree");
    failures += check(rank0.linear.value_heads == 24 && rank0.linear.conv_channels == 5120,
                      "width 2 did not halve the GDN value heads and convolution channels");
    failures += check(rank0.dflash_local && rank0.dflash_local->kv_heads == 1,
                      "width 2 did not halve the DFlash KV heads");
    q38::StateImageSpec expected = base;
    expected.linear.value_heads       = 24;
    expected.linear.conv_channels     = 5120;
    expected.dflash_local->kv_heads   = 1;
    failures += check(same_spec(rank0, expected),
                      "width 2 changed a field that must stay whole on every rank");

    const Planned sharded = plan(rank0);
    // conv {conv_channels, conv_width, slots} BF16; recurrent {key_head_dim, value_head_dim,
    // value_heads, slots} FP32 (core/linear_attention_state_pool.cpp:95-99).
    const std::size_t slots = static_cast<std::size_t>(rank0.linear.slot_count);
    failures += check(sharded.layout.linear.conv[0].bytes == 5120ULL * 3ULL * slots * 2ULL,
                      "sharded convolution state bytes are not conv_channels/width * conv_width * "
                      "slots * BF16");
    failures += check(sharded.layout.linear.recurrent[0].bytes ==
                          128ULL * 128ULL * 24ULL * slots * 4ULL,
                      "sharded recurrent state bytes are not key_head_dim * value_head_dim * "
                      "value_heads/width * slots * FP32");
    failures += check(sharded.layout.continuation_hidden.shape[0] == 2560 &&
                          sharded.layout.continuation_hidden.shape[1] == 4,
                      "the continuation hidden must stay whole and keep the slot count");

    // Width 3 divides the 48 value heads (16 each) but NOT the convolution channels: 10240 / 3 is
    // not integral, and neither is the 2048-wide key segment. The head count alone is not enough to
    // shard the StateImage, so this throws — the finding, not a failure.
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)q38::shard_state_image_spec(base, 3, 0); },
        "width 3 against the 125B StateImage (convolution channels do not divide)");
    // A spec whose every per-head and per-channel extent divides does shard at width 3.
    const q38::StateImageSpec thirds = q38::shard_state_image_spec(divisible_by_three_spec(), 3, 2);
    failures += check(thirds.linear.value_heads == 2 && thirds.linear.conv_channels == 10,
                      "width 3 did not split a spec whose extents divide");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)q38::shard_state_image_spec(base, 5, 0); },
        "width 5 against the 125B StateImage (value dims do not divide)");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)q38::shard_state_image_spec(base, 4, 0); },
        "width 4 against the 125B StateImage (48 value heads divide, 10240 does not)");

    // Invalid arguments and inconsistent geometry.
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)q38::shard_state_image_spec(base, 0, 0); }, "zero width");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)q38::shard_state_image_spec(base, 2, -1); }, "negative rank");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)q38::shard_state_image_spec(base, 2, 2); }, "rank outside the width");
    q38::StateImageSpec negative_heads = base;
    negative_heads.linear.value_heads  = -2;
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)q38::shard_state_image_spec(negative_heads, 2, 0); },
        "negative value heads");
    q38::StateImageSpec inconsistent = base;
    inconsistent.linear.conv_channels = 10241; // 2 * key_dim + value_dim cannot be odd
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)q38::shard_state_image_spec(inconsistent, 2, 0); },
        "convolution channels that are not 2 * key_dim + value_dim");
    q38::StateImageSpec odd_dflash = base;
    odd_dflash.dflash_local->kv_heads = 3;
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)q38::shard_state_image_spec(odd_dflash, 2, 0); },
        "DFlash KV heads that do not divide");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
