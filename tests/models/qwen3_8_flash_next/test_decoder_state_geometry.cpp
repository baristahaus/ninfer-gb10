// Host-only contract test for the per-rank decoder geometry: the spec a rank-1 page pool is
// planned from. It touches no CUDA — the shard function is pure — so it runs anywhere.

#include <ninfer/models/qwen3_8_flash_next/decoder_state.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace q38 = ninfer::models::qwen3_8_flash_next;

namespace {

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

template <typename Fn>
int expect_throws(Fn&& fn, const std::string& label) {
    try {
        fn();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "FAIL: " << label << " did not throw std::invalid_argument\n";
    return 1;
}

// The 125B's decoder spec as `persistent_layout` builds it, minus the parts the shard function does
// not read. The auxiliary planes are the QSA index planes: a BF16 128-wide index-key plane and an
// I32 3-word control plane, one of each appended to every layer.
q38::DecoderStateSpec text_spec(std::int32_t kv_heads) {
    return q38::DecoderStateSpec{
        .full_attention_layers     = 12,
        .mtp_layers                = 1,
        .capacity                  = 262144,
        .kv_heads                  = kv_heads,
        .attention_head_dim        = 256,
        .kv_storage                = ninfer::KvCacheStorage::BFloat16,
        .enable_mtp                = true,
        .kv_table_rows             = 8,
        .text_physical_page_groups = 4097,
        .mtp_physical_page_groups  = 4100,
        .auxiliary_planes          = {{ninfer::DType::BF16, 128, 1, 256},
                                      {ninfer::DType::I32, 3, 1, 256}},
    };
}

bool same_auxiliary(const std::vector<ninfer::KVPlaneGeometry>& left,
                    const std::vector<ninfer::KVPlaneGeometry>& right) {
    return left == right;
}

// Plans a decoder state over a scratch layout builder; returns the layout for comparison.
q38::DecoderStateLayout plan(const q38::DecoderStateSpec& spec) {
    ninfer::LayoutBuilder builder;
    q38::DecoderStateLayout layout = q38::plan_decoder_state(builder, spec);
    (void)builder.finish(256);
    return layout;
}

// Width 1 must be the identity, field for field: the single-device lane plans exactly what it
// planned before the group existed.
int test_width_one_is_identity() {
    int failures = 0;
    const q38::DecoderStateSpec spec   = text_spec(2);
    const q38::DecoderStateSpec single = q38::shard_decoder_state_spec(spec, 1, 0);

    failures += check(single.kv_heads == spec.kv_heads, "width 1 keeps kv_heads");
    failures += check(single.attention_head_dim == spec.attention_head_dim,
                      "width 1 keeps attention_head_dim");
    failures += check(single.capacity == spec.capacity, "width 1 keeps capacity");
    failures += check(single.full_attention_layers == spec.full_attention_layers,
                      "width 1 keeps the full-attention layer count");
    failures += check(single.mtp_layers == spec.mtp_layers, "width 1 keeps the MTP layer count");
    failures += check(single.kv_storage == spec.kv_storage, "width 1 keeps the KV storage");
    failures += check(single.enable_mtp == spec.enable_mtp, "width 1 keeps enable_mtp");
    failures += check(single.kv_table_rows == spec.kv_table_rows,
                      "width 1 keeps the execution-table row count");
    failures += check(single.text_physical_page_groups == spec.text_physical_page_groups,
                      "width 1 keeps the text physical page groups");
    failures += check(single.mtp_physical_page_groups == spec.mtp_physical_page_groups,
                      "width 1 keeps the MTP physical page groups");
    failures += check(same_auxiliary(single.auxiliary_planes, spec.auxiliary_planes),
                      "width 1 keeps the auxiliary planes");

    // And the lane it plans is the same lane: same plane inventory, same bytes.
    const q38::DecoderStateLayout base   = plan(spec);
    const q38::DecoderStateLayout single_layout = plan(single);
    failures += check(base.text_kv.payload_bytes() == single_layout.text_kv.payload_bytes(),
                      "width 1 plans the same text KV payload bytes");
    failures += check(base.text_kv.pages.planes.size() == single_layout.text_kv.pages.planes.size(),
                      "width 1 plans the same text KV plane count");
    failures += check(base.mtp_kv.has_value() == single_layout.mtp_kv.has_value() &&
                          (!base.mtp_kv ||
                           base.mtp_kv->payload_bytes() == single_layout.mtp_kv->payload_bytes()),
                      "width 1 plans the same MTP KV payload bytes");
    return failures;
}

// Width 2 halves the KV heads and nothing else: the two ranks' specs must partition the head count
// and agree on every page/state fact, because the rank-1 mirror addresses rank 0's pages by index.
int test_width_two_halves_the_heads() {
    int failures = 0;
    const q38::DecoderStateSpec spec = text_spec(2);
    const q38::DecoderStateSpec rank0 = q38::shard_decoder_state_spec(spec, 2, 0);
    const q38::DecoderStateSpec rank1 = q38::shard_decoder_state_spec(spec, 2, 1);

    failures += check(rank0.kv_heads == 1, "rank 0 holds half the KV heads at width 2");
    failures += check(rank1.kv_heads == 1, "rank 1 holds half the KV heads at width 2");
    failures += check(rank0.kv_heads + rank1.kv_heads == spec.kv_heads,
                      "the two ranks partition the KV heads");
    failures += check(rank0.attention_head_dim == 256 && rank1.attention_head_dim == 256,
                      "attention_head_dim is per head, not per rank");

    // Every fact the mirror contract needs equal must be equal.
    failures += check(rank0.capacity == rank1.capacity, "capacity is equal on both ranks");
    failures += check(rank0.text_physical_page_groups == rank1.text_physical_page_groups &&
                          rank0.text_physical_page_groups == spec.text_physical_page_groups,
                      "the text physical page groups are equal and unchanged");
    failures += check(rank0.mtp_physical_page_groups == rank1.mtp_physical_page_groups &&
                          rank0.mtp_physical_page_groups == spec.mtp_physical_page_groups,
                      "the MTP physical page groups are equal and unchanged");
    failures += check(rank0.kv_table_rows == rank1.kv_table_rows &&
                          rank0.kv_table_rows == spec.kv_table_rows,
                      "the execution-table rows are equal and unchanged");
    failures += check(rank0.kv_storage == rank1.kv_storage, "the KV storage is unchanged");
    failures += check(rank0.full_attention_layers == spec.full_attention_layers &&
                          rank1.full_attention_layers == spec.full_attention_layers,
                      "both ranks carry every attention layer");
    failures += check(rank0.mtp_layers == spec.mtp_layers && rank1.mtp_layers == spec.mtp_layers,
                      "both ranks carry the MTP layer");
    failures += check(rank0.enable_mtp == spec.enable_mtp, "enable_mtp is unchanged");

    // The QSA index planes are shard-invariant: neither carries a head axis (head_extent is 1), and
    // the indexer is replicated because the shared TopK decision must be identical on both ranks.
    failures += check(same_auxiliary(rank0.auxiliary_planes, spec.auxiliary_planes),
                      "rank 0's auxiliary planes are unchanged");
    failures += check(same_auxiliary(rank1.auxiliary_planes, spec.auxiliary_planes),
                      "rank 1's auxiliary planes are unchanged");
    failures += check(rank0.auxiliary_planes.size() == 2 &&
                          rank0.auxiliary_planes[0].head_extent == 1 &&
                          rank0.auxiliary_planes[1].head_extent == 1,
                      "the auxiliary planes have no head axis to shard");

    // The shard reaches the planned layout: the same plane inventory, fewer bytes per plane. The
    // pool's page count and geometry fields are what a mirror compares, so the plane count must not
    // change with the width.
    const q38::DecoderStateLayout base   = plan(spec);
    const q38::DecoderStateLayout sharded = plan(rank0);
    failures += check(base.text_kv.pages.planes.size() == sharded.text_kv.pages.planes.size(),
                      "a sharded rank plans the same text KV plane count");
    failures += check(sharded.text_kv.payload_bytes() < base.text_kv.payload_bytes(),
                      "a sharded rank plans fewer text KV payload bytes");
    failures += check(base.text_kv.pages.spec.page_group_count ==
                          sharded.text_kv.pages.spec.page_group_count,
                      "a sharded rank plans the same physical page count");
    return failures;
}

// The 125B has 2 KV heads, so width 3 has no equal split: it must throw rather than round a rank
// down to one head and another up to one, which would give the ranks different plane extents.
int test_width_three_is_refused() {
    int failures = 0;
    const q38::DecoderStateSpec spec = text_spec(2);
    failures += expect_throws([&] { (void)q38::shard_decoder_state_spec(spec, 3, 0); },
                              "width 3 against 2 KV heads");
    failures += expect_throws([&] { (void)q38::shard_decoder_state_spec(spec, 3, 1); },
                              "width 3 rank 1 against 2 KV heads");

    // A geometry whose heads do divide still splits, so the refusal is about the count, not 3.
    const q38::DecoderStateSpec six = text_spec(6);
    failures += check(q38::shard_decoder_state_spec(six, 3, 2).kv_heads == 2,
                      "width 3 splits 6 KV heads into 2 per rank");
    return failures;
}

int test_invalid_arguments() {
    int failures = 0;
    const q38::DecoderStateSpec spec = text_spec(2);
    failures += expect_throws([&] { (void)q38::shard_decoder_state_spec(spec, 0, 0); },
                              "width 0");
    failures += expect_throws([&] { (void)q38::shard_decoder_state_spec(spec, -1, 0); },
                              "negative width");
    failures += expect_throws([&] { (void)q38::shard_decoder_state_spec(spec, 2, 2); },
                              "rank outside the width");
    failures += expect_throws([&] { (void)q38::shard_decoder_state_spec(spec, 2, -1); },
                              "negative rank");
    failures += expect_throws([&] { (void)q38::shard_decoder_state_spec(text_spec(0), 2, 0); },
                              "zero KV heads");
    return failures;
}

} // namespace

int main() {
    try {
        int failures = 0;
        failures += test_width_one_is_identity();
        failures += test_width_two_halves_the_heads();
        failures += test_width_three_is_refused();
        failures += test_invalid_arguments();
        if (failures == 0) { std::cout << "ok\n"; }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
