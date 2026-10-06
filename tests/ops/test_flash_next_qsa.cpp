#include "ninfer/ops/flash_next_qsa.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr int kHidden = 2560;
constexpr int kHeadDim = 256;
constexpr int kQueryRows = 6144;
constexpr int kKvRows = 512;
constexpr int kCachePages = 33;

Weight bf16_weight(const DeviceBuffer& storage, int rows, int columns) {
    Weight out{};
    out.payload = out.qdata = storage.p;
    out.payload_bytes = storage.bytes;
    out.qtype = QType::BF16;
    out.layout = QuantLayout::Contiguous;
    out.n = out.shape[0] = out.padded_shape[0] = rows;
    out.k = out.shape[1] = out.padded_shape[1] = columns;
    out.ndim = 2;
    return out;
}

void store_bf16(DeviceBuffer& storage, std::size_t element, float value) {
    const std::uint16_t bits = f32_to_bf16(value);
    storage.copy_from_host(&bits, sizeof(bits), element * sizeof(bits));
}

int run(int kPrefillTokens) {
    constexpr std::size_t kFullContextPrefillWorkspace = 2879492096ULL;
    if (ops::flash_next_qsa_workspace_capacity_bytes(8192, 262144) !=
        kFullContextPrefillWorkspace) {
        std::cerr << "Flash-Next QSA full-context prefill workspace regressed\n";
        return 1;
    }
    std::vector<float> input(kHidden, 0.0F);
    input[0] = 0.5F;
    input[1] = 0.25F;
    round_to_bf16(input);
    DeviceBuffer d_input = to_device_bf16(input);

    const auto matrix = [](int rows, int columns) {
        return DeviceBuffer(static_cast<std::size_t>(rows) * columns * sizeof(std::uint16_t));
    };
    DeviceBuffer d_query_gate = matrix(2 * kQueryRows, kHidden);
    DeviceBuffer d_key = matrix(kKvRows, kHidden);
    DeviceBuffer d_value = matrix(kKvRows, kHidden);
    DeviceBuffer d_output_weight = matrix(kHidden, kQueryRows);
    DeviceBuffer d_index_query = matrix(512, kHidden);
    DeviceBuffer d_index_key = matrix(128, kHidden);
    d_query_gate.fill();
    d_key.fill();
    d_value.fill();
    d_output_weight.fill();
    d_index_query.fill();
    d_index_key.fill();
    store_bf16(d_query_gate, static_cast<std::size_t>(kHeadDim) * kHidden, 2.0F);
    store_bf16(d_value, 1, 2.0F);
    store_bf16(d_output_weight, 0, 1.0F);
    store_bf16(d_index_query, 0, 2.0F);
    store_bf16(d_index_key, 0, 2.0F);

    DeviceBuffer d_query_norm(kHeadDim * sizeof(std::uint16_t));
    DeviceBuffer d_key_norm(kHeadDim * sizeof(std::uint16_t));
    DeviceBuffer d_index_query_norm(128 * sizeof(std::uint16_t));
    DeviceBuffer d_index_key_norm(128 * sizeof(std::uint16_t));
    d_query_norm.fill();
    d_key_norm.fill();
    d_index_query_norm.fill();
    d_index_key_norm.fill();

    DeviceBuffer d_k_pages(static_cast<std::size_t>(kHeadDim) * 64 * 2 * kCachePages *
                           sizeof(std::uint16_t));
    DeviceBuffer d_v_pages(static_cast<std::size_t>(kHeadDim) * 64 * 2 * kCachePages *
                           sizeof(std::uint16_t));
    DeviceBuffer d_raw_pages(static_cast<std::size_t>(128) * 64 * kCachePages *
                             sizeof(std::uint16_t));
    DeviceBuffer d_position_pages(static_cast<std::size_t>(3) * 64 * kCachePages *
                                  sizeof(std::int32_t));
    std::vector<int> identity_table(kCachePages);
    for (int page = 0; page < kCachePages; ++page) { identity_table[page] = page; }
    DeviceBuffer d_table = to_device_i32(identity_table);
    d_k_pages.fill();
    d_v_pages.fill();
    d_raw_pages.fill();
    d_position_pages.fill();
    PagedKVBatchLayerView cache{
        .k_pages = Tensor(d_k_pages.p, DType::BF16, {kHeadDim, 64, 2, kCachePages}),
        .v_pages = Tensor(d_v_pages.p, DType::BF16, {kHeadDim, 64, 2, kCachePages}),
        .block_tables = Tensor(d_table.p, DType::I32, {kCachePages, 1}),
        .auxiliary_pages = {
            Tensor(d_raw_pages.p, DType::BF16, {128, 64, kCachePages, 1}),
            Tensor(d_position_pages.p, DType::I32, {3, 64, kCachePages, 1}),
        },
        .head_dim = kHeadDim,
        .num_kv_heads = 2,
        .storage = KvCacheStorage::BFloat16KeyValue,
    };

    DeviceBuffer d_cache_positions = to_device_i32(std::vector<int>{0});
    DeviceBuffer d_rope_positions = to_device_i32(std::vector<int>{0, 0, 0});
    DeviceBuffer d_valid = to_device_i32(std::vector<int>{1});
    DeviceBuffer d_rows = to_device_i32(std::vector<int>{0});
    GuardedDeviceBuffer d_destination(kHidden * sizeof(std::uint16_t));
    Tensor input_tensor(d_input.p, DType::BF16, {kHidden, 1});
    Tensor destination(d_destination.data(), DType::BF16, {kHidden, 1});
    ops::FlashNextQsaWeights weights{
        .query_gate = bf16_weight(d_query_gate, 2 * kQueryRows, kHidden),
        .key = bf16_weight(d_key, kKvRows, kHidden),
        .value = bf16_weight(d_value, kKvRows, kHidden),
        .output = bf16_weight(d_output_weight, kHidden, kQueryRows),
        .query_norm = Tensor(d_query_norm.p, DType::BF16, {kHeadDim}),
        .key_norm = Tensor(d_key_norm.p, DType::BF16, {kHeadDim}),
        .index_query = bf16_weight(d_index_query, 512, kHidden),
        .index_key = bf16_weight(d_index_key, 128, kHidden),
        .index_query_norm = Tensor(d_index_query_norm.p, DType::BF16, {128}),
        .index_key_norm = Tensor(d_index_key_norm.p, DType::BF16, {128}),
    };
    WorkspaceArena workspace(ops::flash_next_qsa_workspace_capacity_bytes(1,
                                                                           64 * kCachePages));
    ops::flash_next_qsa(
        input_tensor, Tensor(d_cache_positions.p, DType::I32, {1, 1}),
        Tensor(d_rope_positions.p, DType::I32, {1, 1, 3}),
        Tensor(d_valid.p, DType::I32, {1}), Tensor(d_rows.p, DType::I32, {1}), weights,
        cache, {.min_visible_keys = 1, .max_visible_keys = 1}, destination, workspace,
        nullptr);
    cuda_synchronize();

    std::vector<double> expected(kHidden, 0.0);
    expected[0] = 0.5 / (1.0 + std::exp(-1.0));
    int failures = verify_pointwise("Flash-Next QSA one-token attention",
                                    from_device_bf16(d_destination.data(), kHidden), expected,
                                    {/*absolute*/ 4.0e-3, /*relative*/ 2.0e-2});
    failures += d_destination.verify_guards("Flash-Next QSA destination");
    std::uint16_t cached_value = 0;
    d_v_pages.copy_to_host(&cached_value, sizeof(cached_value));
    if (cached_value != 0x3f00U) {
        std::cerr << "Flash-Next QSA did not commit its projected value to the cache\n";
        ++failures;
    }

    // Exercise the public FP8 cache profile through the same complete leaf. The one-token
    // attention result is independent of the represented K value and the projected V=0.5 is
    // exactly representable by the row-scaled E4M3 codec.
    DeviceBuffer d_fp8_k_pages(static_cast<std::size_t>(kHeadDim) * 64 * 2 * kCachePages);
    DeviceBuffer d_fp8_v_pages(static_cast<std::size_t>(kHeadDim) * 64 * 2 * kCachePages);
    DeviceBuffer d_fp8_k_scales(static_cast<std::size_t>(64) * 2 * kCachePages *
                               sizeof(std::uint16_t));
    DeviceBuffer d_fp8_v_scales(static_cast<std::size_t>(64) * 2 * kCachePages *
                               sizeof(std::uint16_t));
    d_fp8_k_pages.fill();
    d_fp8_v_pages.fill();
    d_fp8_k_scales.fill();
    d_fp8_v_scales.fill();
    PagedKVBatchLayerView fp8_cache{
        .k_pages = Tensor(d_fp8_k_pages.p, DType::FP8_E4M3FN, {kHeadDim, 64, 2, kCachePages}),
        .v_pages = Tensor(d_fp8_v_pages.p, DType::FP8_E4M3FN, {kHeadDim, 64, 2, kCachePages}),
        .k_scale_pages = Tensor(d_fp8_k_scales.p, DType::FP16, {1, 64, 2, kCachePages}),
        .v_scale_pages = Tensor(d_fp8_v_scales.p, DType::FP16, {1, 64, 2, kCachePages}),
        .block_tables = Tensor(d_table.p, DType::I32, {kCachePages, 1}),
        .auxiliary_pages = {
            Tensor(d_raw_pages.p, DType::BF16, {128, 64, kCachePages, 1}),
            Tensor(d_position_pages.p, DType::I32, {3, 64, kCachePages, 1}),
        },
        .head_dim = kHeadDim,
        .num_kv_heads = 2,
        .storage = KvCacheStorage::Fp8E4M3Row256,
    };
    d_destination.fill();
    workspace.reset();
    ops::flash_next_qsa(
        input_tensor, Tensor(d_cache_positions.p, DType::I32, {1, 1}),
        Tensor(d_rope_positions.p, DType::I32, {1, 1, 3}),
        Tensor(d_valid.p, DType::I32, {1}), Tensor(d_rows.p, DType::I32, {1}), weights,
        fp8_cache, {.min_visible_keys = 1, .max_visible_keys = 1}, destination, workspace,
        nullptr);
    cuda_synchronize();
    failures += verify_pointwise("Flash-Next QSA FP8 one-token attention",
                                 from_device_bf16(d_destination.data(), kHidden), expected,
                                 {/*absolute*/ 4.0e-3, /*relative*/ 2.0e-2});
    failures += d_destination.verify_guards("Flash-Next QSA FP8 destination");
    std::uint8_t cached_fp8_value = 0;
    std::uint16_t cached_fp8_scale = 0;
    d_fp8_v_pages.copy_to_host(&cached_fp8_value, sizeof(cached_fp8_value));
    d_fp8_v_scales.copy_to_host(&cached_fp8_scale, sizeof(cached_fp8_scale));
    if (cached_fp8_value == 0 || cached_fp8_scale == 0) {
        std::cerr << "Flash-Next QSA did not commit its projected value to the FP8 cache\n";
        ++failures;
    }

    // The INT8 profile stores four 64-value group scales per row; V=0.5 is exact in it as well.
    DeviceBuffer d_int8_k_pages(static_cast<std::size_t>(kHeadDim) * 64 * 2 * kCachePages);
    DeviceBuffer d_int8_v_pages(static_cast<std::size_t>(kHeadDim) * 64 * 2 * kCachePages);
    DeviceBuffer d_int8_k_scales(static_cast<std::size_t>(4) * 64 * 2 * kCachePages *
                                 sizeof(std::uint16_t));
    DeviceBuffer d_int8_v_scales(static_cast<std::size_t>(4) * 64 * 2 * kCachePages *
                                 sizeof(std::uint16_t));
    d_int8_k_pages.fill();
    d_int8_v_pages.fill();
    d_int8_k_scales.fill();
    d_int8_v_scales.fill();
    PagedKVBatchLayerView int8_cache{
        .k_pages = Tensor(d_int8_k_pages.p, DType::I8, {kHeadDim, 64, 2, kCachePages}),
        .v_pages = Tensor(d_int8_v_pages.p, DType::I8, {kHeadDim, 64, 2, kCachePages}),
        .k_scale_pages = Tensor(d_int8_k_scales.p, DType::FP16, {4, 64, 2, kCachePages}),
        .v_scale_pages = Tensor(d_int8_v_scales.p, DType::FP16, {4, 64, 2, kCachePages}),
        .block_tables = Tensor(d_table.p, DType::I32, {kCachePages, 1}),
        .auxiliary_pages = {
            Tensor(d_raw_pages.p, DType::BF16, {128, 64, kCachePages, 1}),
            Tensor(d_position_pages.p, DType::I32, {3, 64, kCachePages, 1}),
        },
        .head_dim = kHeadDim,
        .num_kv_heads = 2,
        .storage = KvCacheStorage::Int8Group64,
    };
    d_destination.fill();
    workspace.reset();
    ops::flash_next_qsa(
        input_tensor, Tensor(d_cache_positions.p, DType::I32, {1, 1}),
        Tensor(d_rope_positions.p, DType::I32, {1, 1, 3}),
        Tensor(d_valid.p, DType::I32, {1}), Tensor(d_rows.p, DType::I32, {1}), weights,
        int8_cache, {.min_visible_keys = 1, .max_visible_keys = 1}, destination, workspace,
        nullptr);
    cuda_synchronize();
    failures += verify_pointwise("Flash-Next QSA INT8 one-token attention",
                                 from_device_bf16(d_destination.data(), kHidden), expected,
                                 {/*absolute*/ 4.0e-3, /*relative*/ 2.0e-2});
    failures += d_destination.verify_guards("Flash-Next QSA INT8 destination");
    std::int8_t cached_int8_value = 0;
    std::uint16_t cached_int8_scale = 0;
    d_int8_v_pages.copy_to_host(&cached_int8_value, sizeof(cached_int8_value));
    d_int8_v_scales.copy_to_host(&cached_int8_scale, sizeof(cached_int8_scale));
    if (cached_int8_value != 127 || cached_int8_scale == 0) {
        std::cerr << "Flash-Next QSA did not commit its projected value to the INT8 cache\n";
        ++failures;
    }

    // Complete the first index group and verify the persistent BF16 compression boundary.  The
    // first call already stored raw key [1,0,...] at position zero.
    for (int position = 1; position < 3; ++position) {
        store_bf16(d_raw_pages, static_cast<std::size_t>(128) * position, 1.0F);
    }
    DeviceBuffer d_group_end_position = to_device_i32(std::vector<int>{3});
    d_destination.fill();
    workspace.reset();
    ops::flash_next_qsa(
        input_tensor, Tensor(d_group_end_position.p, DType::I32, {1, 1}),
        Tensor(d_rope_positions.p, DType::I32, {1, 1, 3}),
        Tensor(d_valid.p, DType::I32, {1}), Tensor(d_rows.p, DType::I32, {1}), weights,
        cache, {.min_visible_keys = 4, .max_visible_keys = 4}, destination, workspace, nullptr);
    cuda_synchronize();
    std::uint16_t compressed_value = 0;
    d_raw_pages.copy_to_host(&compressed_value, sizeof(compressed_value),
                             static_cast<std::size_t>(128) * 3 * sizeof(std::uint16_t));
    const std::uint16_t expected_compressed =
        f32_to_bf16(1.0F / std::sqrt(1.0F / 128.0F + 1.0e-6F));
    if (compressed_value != expected_compressed) {
        std::cerr << "Flash-Next QSA compressed-key boundary mismatch\n";
        ++failures;
    }

    // Force the selected-group route over 513 complete index groups.  Group 512 is the unique
    // positive-score group, so exact top-512 selection must retain it while dropping one of the
    // zero-score groups.  The attention query is zero, making the independent oracle a uniform
    // average over 512*4 selected positions plus the one-token causal tail.
    constexpr int kLongPosition = 2052;
    for (int position = 2048; position < 2052; ++position) {
        const int page = position / 64;
        const int page_offset = position % 64;
        const std::size_t value_offset = static_cast<std::size_t>(kHeadDim) *
            (page_offset + 64 * (2 * page));
        store_bf16(d_v_pages, value_offset, 1.0F);
    }
    const int compressed_position = 2051;
    store_bf16(d_raw_pages, static_cast<std::size_t>(128) *
        (compressed_position % 64 + 64 * (compressed_position / 64)), 1.0F);
    DeviceBuffer d_long_cache_positions = to_device_i32(std::vector<int>{kLongPosition});
    DeviceBuffer d_long_rope_positions = to_device_i32(std::vector<int>{0, 0, 0});
    d_destination.fill();
    workspace.reset();
    ops::flash_next_qsa(
        input_tensor, Tensor(d_long_cache_positions.p, DType::I32, {1, 1}),
        Tensor(d_long_rope_positions.p, DType::I32, {1, 1, 3}),
        Tensor(d_valid.p, DType::I32, {1}), Tensor(d_rows.p, DType::I32, {1}), weights,
        cache,
        {.min_visible_keys = kLongPosition + 1, .max_visible_keys = kLongPosition + 1},
        destination, workspace, nullptr);
    cuda_synchronize();
    std::fill(expected.begin(), expected.end(), 0.0);
    expected[0] = (4.0 + 0.5) / 2049.0 / (1.0 + std::exp(-1.0));
    failures += verify_pointwise("Flash-Next QSA selected-group attention",
                                 from_device_bf16(d_destination.data(), kHidden), expected,
                                 {/*absolute*/ 4.0e-4, /*relative*/ 4.0e-2});
    failures += d_destination.verify_guards("Flash-Next QSA selected destination");

    // Exercise the tensor-core prefill scorer over the same selected-group boundary. New cache
    // values are zero, while the uniquely positive old group retains four unit values. Main Q is
    // zero, so each result is the exact uniform average over 2048 selected tokens plus its open
    // causal tail.
    d_value.fill();
    d_index_key.fill();
    std::vector<float> prefill_input(static_cast<std::size_t>(kHidden) * kPrefillTokens, 0.0F);
    for (int token = 0; token < kPrefillTokens; ++token) {
        prefill_input[static_cast<std::size_t>(token) * kHidden] = 0.5F;
        prefill_input[static_cast<std::size_t>(token) * kHidden + 1] = 0.25F;
    }
    round_to_bf16(prefill_input);
    DeviceBuffer d_prefill_input = to_device_bf16(prefill_input);
    std::vector<int> prefill_positions(kPrefillTokens);
    for (int token = 0; token < kPrefillTokens; ++token) {
        prefill_positions[token] = kLongPosition + token;
    }
    DeviceBuffer d_prefill_positions = to_device_i32(prefill_positions);
    DeviceBuffer d_prefill_rope = to_device_i32(std::vector<int>(3 * kPrefillTokens, 0));
    DeviceBuffer d_prefill_valid = to_device_i32(std::vector<int>{kPrefillTokens});
    GuardedDeviceBuffer d_prefill_destination(
        static_cast<std::size_t>(kHidden) * kPrefillTokens * sizeof(std::uint16_t));
    Tensor prefill_destination(d_prefill_destination.data(), DType::BF16,
                               {kHidden, kPrefillTokens});
    WorkspaceArena prefill_workspace(
        ops::flash_next_qsa_workspace_capacity_bytes(kPrefillTokens, 64 * kCachePages));
    ops::flash_next_qsa(
        Tensor(d_prefill_input.p, DType::BF16, {kHidden, kPrefillTokens}),
        Tensor(d_prefill_positions.p, DType::I32, {kPrefillTokens, 1}),
        Tensor(d_prefill_rope.p, DType::I32, {kPrefillTokens, 1, 3}),
        Tensor(d_prefill_valid.p, DType::I32, {1}), Tensor(d_rows.p, DType::I32, {1}),
        weights, cache,
        {.min_visible_keys = kLongPosition + 1,
         .max_visible_keys = kLongPosition + kPrefillTokens},
        prefill_destination, prefill_workspace, nullptr);
    cuda_synchronize();
    std::vector<double> prefill_expected(
        static_cast<std::size_t>(kHidden) * kPrefillTokens, 0.0);
    for (int token = 0; token < kPrefillTokens; ++token) {
        const int sequence_length = kLongPosition + token + 1;
        const int tail = sequence_length % 4;
        prefill_expected[static_cast<std::size_t>(token) * kHidden] =
            4.0 / (2048.0 + tail) / (1.0 + std::exp(-1.0));
    }
    failures += verify_pointwise(
        "Flash-Next QSA tensor-core prefill selection",
        from_device_bf16(d_prefill_destination.data(), prefill_expected.size()),
        prefill_expected, {/*absolute*/ 4.0e-4, /*relative*/ 4.0e-2});
    failures += d_prefill_destination.verify_guards("Flash-Next QSA prefill destination");
    // Neighboring queries reuse overlapping, partly overlapping, and disjoint rows.
    // Keep nonuniform represented K/V and an independent FP64 softmax oracle. Holes
    // occupy complete tiles and the final valid entry sits after those holes.
    d_k_pages.fill();
    d_v_pages.fill();
    store_bf16(d_query_gate, static_cast<std::size_t>(255) * kHidden, 2.0F);
    store_bf16(d_query_norm, 255, 0.0F); // QSA RMSNorm uses (1 + weight).
    for (int position = 0; position < 256; ++position) {
        const auto base = static_cast<std::size_t>(kHeadDim) *
            (position % 64 + 64 * 2 * (position / 64));
        store_bf16(d_k_pages, base + 255, 0.125F * (position % 7 - 3));
        store_bf16(d_v_pages, base, 0.125F * (position % 11 - 5));
    }
    // Different signed query/gate inputs catch a query tile reading another row's data.
    const auto query_input = [](int token) {
        return (token % 2 == 0 ? 1.0F : -1.0F) * (0.375F + 0.125F * (token % 3));
    };
    for (int token = 0; token < kPrefillTokens; ++token) {
        store_bf16(d_prefill_input, static_cast<std::size_t>(token) * kHidden,
                   query_input(token));
    }
    const int active = kPrefillTokens - 1;
    d_prefill_valid.copy_from_host(&active, sizeof(active));
    for (int pattern = 0; pattern < 3; ++pattern) {
        std::vector<int> selected(2051 * kPrefillTokens, -1);
        std::vector<double> oracle(static_cast<std::size_t>(kHidden) * kPrefillTokens, 0.0);
        for (int token = 0; token < active; ++token) {
            double numerator = 0.0, denominator = 0.0;
            const int shift = token % 2 * (pattern == 1 ? 4 : pattern == 2 ? 1 : 0);
            for (int item = 0; item < 17; ++item) {
                const int position = 4 * item + shift;
                const int slot = item == 16 ? 2050 : item;
                selected[2051 * token + slot] = position;
                const double key = 0.125 * (position % 7 - 3);
                const double value = 0.125 * (position % 11 - 5);
                const double projected_query = 2.0 * query_input(token);
                const double q = projected_query /
                    std::sqrt(projected_query * projected_query / 256.0 + 1.0e-6);
                const double probability = std::exp(q * key / 16.0);
                numerator += probability * value;
                denominator += probability;
            }
            oracle[static_cast<std::size_t>(token) * kHidden] =
                numerator / denominator / (1.0 + std::exp(-2.0 * query_input(token)));
        }
        DeviceBuffer d_selected = to_device_i32(selected);
        Tensor selected_tensor(d_selected.p, DType::I32, {2051, kPrefillTokens});
        d_prefill_destination.fill();
        prefill_workspace.reset();
        ops::flash_next_qsa(
            Tensor(d_prefill_input.p, DType::BF16, {kHidden, kPrefillTokens}),
            Tensor(d_prefill_positions.p, DType::I32, {kPrefillTokens, 1}),
            Tensor(d_prefill_rope.p, DType::I32, {kPrefillTokens, 1, 3}),
            Tensor(d_prefill_valid.p, DType::I32, {1}), Tensor(d_rows.p, DType::I32, {1}),
            weights, cache, {.min_visible_keys = kLongPosition + 1,
                             .max_visible_keys = kLongPosition + kPrefillTokens},
            prefill_destination, prefill_workspace, nullptr, nullptr,
            {.reused_indices = &selected_tensor});
        cuda_synchronize();
        auto actual = from_device_bf16(d_prefill_destination.data(), oracle.size());
        actual.resize(static_cast<std::size_t>(active) * kHidden);
        oracle.resize(actual.size());
        failures += verify_pointwise("QSA neighboring selections", actual, oracle,
                                      {/*absolute*/ 4.0e-3, /*relative*/ 2.0e-2});
        failures += d_prefill_destination.verify_guards("QSA neighboring output");
        if (from_device<int>(d_selected, selected.size()) != selected) {
            std::cerr << "QSA modified reused selections\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) { return 77; }
    try {
        const int failures = run(17) + run(18);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Flash-Next QSA\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Flash-Next QSA: " << error.what() << '\n';
        return 1;
    }
}
