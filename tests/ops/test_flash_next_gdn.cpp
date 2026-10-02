#include "ninfer/ops/flash_next_gdn.h"
#include "ninfer/ops/gdn_replay.h"

#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

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

int run() {
    constexpr int hidden = 2560;
    constexpr int qk = 2048;
    constexpr int value = 6144;
    constexpr int convolution = 10240;
    const auto matrix = [](int rows, int columns) {
        DeviceBuffer result(static_cast<std::size_t>(rows) * columns * sizeof(std::uint16_t));
        result.fill();
        return result;
    };

    std::vector<float> input(hidden, 0.0F);
    input[0] = 1.0F;
    DeviceBuffer d_input = to_device_bf16(input);
    DeviceBuffer d_a = matrix(48, hidden);
    DeviceBuffer d_b = matrix(48, hidden);
    DeviceBuffer d_qkv = matrix(convolution, hidden);
    DeviceBuffer d_z = matrix(value, hidden);
    DeviceBuffer d_output = matrix(hidden, value);
    store_bf16(d_qkv, 0, 1.0F);
    store_bf16(d_qkv, static_cast<std::size_t>(qk) * hidden, 1.0F);
    store_bf16(d_qkv, static_cast<std::size_t>(2 * qk) * hidden, 1.0F);
    store_bf16(d_z, 0, 1.0F);
    store_bf16(d_output, 0, 1.0F);

    DeviceBuffer d_a_log(48 * sizeof(float));
    DeviceBuffer d_dt_bias(48 * sizeof(float));
    d_a_log.fill();
    d_dt_bias.fill();
    DeviceBuffer d_conv(static_cast<std::size_t>(convolution) * 4 * sizeof(std::uint16_t));
    d_conv.fill();
    store_bf16(d_conv, static_cast<std::size_t>(3) * convolution, 1.0F);
    store_bf16(d_conv, static_cast<std::size_t>(3) * convolution + qk, 1.0F);
    store_bf16(d_conv, static_cast<std::size_t>(3) * convolution + 2 * qk, 1.0F);
    DeviceBuffer d_norm(128 * sizeof(std::uint16_t));
    d_norm.fill();
    store_bf16(d_norm, 0, 1.0F);

    DeviceBuffer d_conv_state(static_cast<std::size_t>(convolution) * 3 * sizeof(std::uint16_t));
    DeviceBuffer d_recurrent_state(static_cast<std::size_t>(128) * 128 * 48 * sizeof(float));
    d_conv_state.fill();
    d_recurrent_state.fill();
    GuardedDeviceBuffer d_destination(hidden * sizeof(std::uint16_t));

    ops::FlashNextGdnWeights weights{
        .a_log = Tensor(d_a_log.p, DType::FP32, {48}),
        .dt_bias = Tensor(d_dt_bias.p, DType::FP32, {48}),
        .convolution = Tensor(d_conv.p, DType::BF16, {convolution, 4}),
        .a_projection = bf16_weight(d_a, 48, hidden),
        .b_projection = bf16_weight(d_b, 48, hidden),
        .query_key_value = bf16_weight(d_qkv, convolution, hidden),
        .output_gate = bf16_weight(d_z, value, hidden),
        .norm = Tensor(d_norm.p, DType::BF16, {128}),
        .output = bf16_weight(d_output, hidden, value),
    };
    Tensor input_tensor(d_input.p, DType::BF16, {hidden, 1});
    Tensor conv_state(d_conv_state.p, DType::BF16, {convolution, 3});
    Tensor recurrent_state(d_recurrent_state.p, DType::FP32, {128, 128, 48});
    Tensor destination(d_destination.data(), DType::BF16, {hidden, 1});
    WorkspaceArena workspace(ops::flash_next_gdn_workspace_capacity_bytes(1));
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int multiprocessors = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
    ops::flash_next_gdn(input_tensor, weights, conv_state, conv_state, recurrent_state,
                        recurrent_state, destination, workspace,
                        DeviceExecutionView{nullptr, multiprocessors});
    cuda_synchronize();

    const auto bf16 = [](double value) {
        return static_cast<double>(bf16_to_f32(f32_to_bf16(static_cast<float>(value))));
    };
    const double silu_one = bf16(1.0 / (1.0 + std::exp(-1.0)));
    const double normalized_qk = silu_one / std::sqrt(silu_one * silu_one + 1.0e-6);
    const double recurrent = bf16((1.0 / std::sqrt(128.0)) * 0.5 * silu_one *
                                  normalized_qk * normalized_qk);
    const double normalized = recurrent /
                              std::sqrt(recurrent * recurrent / 128.0 + 1.0e-6);
    std::vector<double> expected(hidden, 0.0);
    expected[0] = bf16(normalized * (1.0 / (1.0 + std::exp(-1.0))));
    int failures = verify_pointwise("Flash-Next GDN complete block",
                                    from_device_bf16(d_destination.data(), hidden), expected,
                                    {/*absolute*/ 2.0e-2, /*relative*/ 3.0e-3});
    failures += d_destination.verify_guards("Flash-Next GDN destination");
    return failures;
}

std::uint32_t mix(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

float pattern(std::uint32_t key, float magnitude) {
    return static_cast<float>(static_cast<std::int32_t>(mix(key) % 2001U) - 1000) *
           (magnitude / 1000.0F);
}

DeviceBuffer bf16_pattern_buffer(std::size_t elements, std::uint32_t seed, float magnitude,
                                 float offset = 0.0F) {
    std::vector<std::uint16_t> bits(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        bits[index] =
            f32_to_bf16(offset + pattern(seed + static_cast<std::uint32_t>(index), magnitude));
    }
    return to_device(bits);
}

struct PendingFoldCase {
    std::int32_t width;
    // Previous round: record row e verified slot previous_slots[e] and committed commits[e].
    std::vector<std::int32_t> previous_slots;
    std::vector<std::int32_t> commits;
    // This round: verify row b reads slot sources[b] with valid[b] columns.
    std::vector<std::int32_t> sources;
    std::vector<std::int32_t> valid;
};

// The fused pending fold must be bitwise identical to folding the previous round's records with
// gdn_replay_fold and then recording this round from the folded states.
int run_pending_fold(const PendingFoldCase& test) {
    constexpr int hidden                = 2560;
    constexpr int value                 = 6144;
    constexpr int convolution           = 10240;
    constexpr std::int32_t layers       = 36;
    constexpr std::int32_t tested_layer = 7;
    constexpr std::int32_t slot_count   = 6;
    const std::int32_t width            = test.width;
    const auto previous_rows            = static_cast<std::int32_t>(test.previous_slots.size());
    const auto batch                    = static_cast<std::int32_t>(test.sources.size());
    const std::int32_t capacity         = std::max(previous_rows, batch);
    const std::string label             = "Flash-Next GDN pending fold W=" + std::to_string(width) +
                              " R=" + std::to_string(previous_rows) + " B=" + std::to_string(batch);

    DeviceBuffer d_a = bf16_pattern_buffer(static_cast<std::size_t>(48) * hidden, 11U, 0.02F);
    DeviceBuffer d_b = bf16_pattern_buffer(static_cast<std::size_t>(48) * hidden, 13U, 0.02F);
    DeviceBuffer d_qkv =
        bf16_pattern_buffer(static_cast<std::size_t>(convolution) * hidden, 17U, 0.02F);
    DeviceBuffer d_z = bf16_pattern_buffer(static_cast<std::size_t>(value) * hidden, 19U, 0.02F);
    DeviceBuffer d_output =
        bf16_pattern_buffer(static_cast<std::size_t>(hidden) * value, 23U, 0.02F);
    DeviceBuffer d_conv = bf16_pattern_buffer(static_cast<std::size_t>(convolution) * 4, 29U, 0.3F);
    DeviceBuffer d_norm = bf16_pattern_buffer(128, 31U, 0.2F, 1.0F);
    std::vector<float> a_log(48), dt_bias(48);
    for (int head = 0; head < 48; ++head) {
        a_log[head]   = 0.5F + pattern(37U + head, 0.5F);
        dt_bias[head] = pattern(41U + head, 0.5F);
    }
    DeviceBuffer d_a_log   = to_device(a_log);
    DeviceBuffer d_dt_bias = to_device(dt_bias);
    const ops::FlashNextGdnWeights weights{
        .a_log           = Tensor(d_a_log.p, DType::FP32, {48}),
        .dt_bias         = Tensor(d_dt_bias.p, DType::FP32, {48}),
        .convolution     = Tensor(d_conv.p, DType::BF16, {convolution, 4}),
        .a_projection    = bf16_weight(d_a, 48, hidden),
        .b_projection    = bf16_weight(d_b, 48, hidden),
        .query_key_value = bf16_weight(d_qkv, convolution, hidden),
        .output_gate     = bf16_weight(d_z, value, hidden),
        .norm            = Tensor(d_norm.p, DType::BF16, {128}),
        .output          = bf16_weight(d_output, hidden, value),
    };

    const GdnReplayRecordSpec spec{.layers          = layers,
                                   .record_capacity = capacity,
                                   .width           = width,
                                   .conv_channels   = convolution,
                                   .qk_heads        = 16,
                                   .value_heads     = 48,
                                   .key_dim         = 128,
                                   .value_dim       = 128};
    LayoutBuilder record_builder;
    const GdnReplayRecordLayout record_layout = plan_gdn_replay_records(record_builder, spec);
    const std::size_t record_bytes            = record_builder.finish(256);
    DeviceBuffer record_storage(record_bytes);
    DeviceBuffer snapshot_storage(record_bytes);
    record_storage.fill(0);
    snapshot_storage.fill(0xff);
    const GdnReplayRecords records({record_storage.p, record_bytes}, record_layout);
    const GdnReplayRecords snapshot({snapshot_storage.p, record_bytes}, record_layout);

    LayoutBuilder state_builder;
    const LinearAttentionStatePoolLayout state_layout = plan_linear_attention_state_pool(
        state_builder, {.layers         = static_cast<std::uint32_t>(layers),
                        .conv_channels  = convolution,
                        .conv_width     = 3,
                        .value_heads    = 48,
                        .value_head_dim = 128,
                        .key_head_dim   = 128,
                        .slot_count     = slot_count,
                        .conv_dtype     = DType::BF16});
    DeviceBuffer state_storage(state_builder.finish(256));
    state_storage.fill(0);
    LinearAttentionStatePool state_pool({state_storage.p, state_storage.bytes}, state_layout);
    const auto layer_state  = state_pool.layer_view(static_cast<std::uint32_t>(tested_layer));
    Tensor conv_states      = layer_state.conv;
    Tensor recurrent_states = layer_state.recurrent;
    {
        DeviceBuffer conv_init = bf16_pattern_buffer(conv_states.numel(), 43U, 0.5F);
        std::vector<float> recurrent_init(recurrent_states.numel());
        for (std::size_t index = 0; index < recurrent_init.size(); ++index) {
            recurrent_init[index] = pattern(47U + static_cast<std::uint32_t>(index), 0.01F);
        }
        CUDA_CHECK(cudaMemcpy(conv_states.data, conv_init.p, conv_states.bytes(),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(recurrent_states.data, recurrent_init.data(),
                              recurrent_states.bytes(), cudaMemcpyHostToDevice));
    }

    WorkspaceArena workspace(
        ops::flash_next_gdn_workspace_capacity_bytes(width * std::max(previous_rows, batch)));
    const auto record_round = [&](std::int32_t rows, const std::vector<std::int32_t>& slots,
                                  const std::vector<std::int32_t>& valid, std::uint32_t seed,
                                  DeviceBuffer& destination,
                                  const ops::FlashNextGdnPendingFold* pending) {
        DeviceBuffer input =
            bf16_pattern_buffer(static_cast<std::size_t>(hidden) * width * rows, seed, 1.0F);
        DeviceBuffer d_valid = to_device(valid);
        DeviceBuffer d_slots = to_device(slots);
        const Tensor input_tensor(input.p, DType::BF16, {hidden, width * rows});
        const Tensor valid_tensor(d_valid.p, DType::I32, {rows});
        const Tensor slot_tensor(d_slots.p, DType::I32, {rows});
        Tensor destination_tensor(destination.p, DType::BF16, {hidden, width * rows});
        ops::flash_next_gdn_replay_record(
            input_tensor, weights, conv_states, recurrent_states, valid_tensor, slot_tensor,
            records.layer(tested_layer, rows), destination_tensor, workspace, nullptr, pending);
        cuda_synchronize();
    };

    // Previous round: every record column written from the initial states.
    DeviceBuffer previous_out(static_cast<std::size_t>(hidden) * width * previous_rows * 2);
    record_round(previous_rows, test.previous_slots,
                 std::vector<std::int32_t>(static_cast<std::size_t>(previous_rows), width), 53U,
                 previous_out, nullptr);
    copy_gdn_replay_record_rows(records, snapshot, previous_rows, nullptr);
    cuda_synchronize();
    const auto initial_states = from_device<std::uint8_t>(state_storage, state_storage.bytes);

    // Reference: fold, then record this round.
    std::vector<ops::GdnReplayFoldRow> fold_rows;
    std::vector<std::int32_t> device_rows;
    for (std::int32_t entry = 0; entry < previous_rows; ++entry) {
        const std::int32_t slot   = test.previous_slots[static_cast<std::size_t>(entry)];
        const std::int32_t commit = test.commits[static_cast<std::size_t>(entry)];
        fold_rows.push_back({slot, slot, commit});
        device_rows.insert(device_rows.end(), {slot, slot, commit, 0});
    }
    const ops::GdnReplayFoldPlan fold_plan(records, state_pool.all_layers_view());
    fold_plan.execute(fold_rows, nullptr);
    DeviceBuffer reference_out(static_cast<std::size_t>(hidden) * width * batch * 2);
    record_round(batch, test.sources, test.valid, 59U, reference_out, nullptr);
    const GdnReplayRecordLayer reference_layer = records.layer(tested_layer, batch);
    const auto reference_states = from_device<std::uint8_t>(state_storage, state_storage.bytes);
    const auto reference_conv =
        from_device<std::uint8_t>(reference_layer.conv.data, reference_layer.conv.bytes());
    const auto reference_key =
        from_device<std::uint8_t>(reference_layer.key.data, reference_layer.key.bytes());
    const auto reference_value =
        from_device<std::uint8_t>(reference_layer.value.data, reference_layer.value.bytes());
    const auto reference_gate =
        from_device<std::uint8_t>(reference_layer.gate.data, reference_layer.gate.bytes());

    // Fused: restore the unfolded states and scribble over the live records, which this round
    // overwrites while the fold reads the snapshot.
    CUDA_CHECK(cudaMemcpy(state_storage.p, initial_states.data(), initial_states.size(),
                          cudaMemcpyHostToDevice));
    record_storage.fill(0x7f);
    DeviceBuffer d_pending = to_device(device_rows);
    const ops::FlashNextGdnPendingFold pending{Tensor(d_pending.p, DType::I32, {4, previous_rows}),
                                               snapshot.layer(tested_layer, previous_rows)};
    DeviceBuffer fused_out(reference_out.bytes);
    record_round(batch, test.sources, test.valid, 59U, fused_out, &pending);

    int failures           = 0;
    const auto expect_same = [&](const std::vector<std::uint8_t>& expected,
                                 const std::vector<std::uint8_t>& actual, const char* field) {
        if (expected != actual) {
            std::cerr << label << ": " << field << " differs from fold-then-record\n";
            ++failures;
        }
    };
    // Only the tested layer's states are compared: the reference fold also folds the other
    // layers from their zero records, which the single-layer fused op never touches.
    const auto layer_bytes = [&](const std::vector<std::uint8_t>& all, const Tensor& tensor) {
        const auto offset =
            static_cast<std::size_t>(static_cast<const std::uint8_t*>(tensor.data) -
                                     static_cast<const std::uint8_t*>(state_storage.p));
        return std::vector<std::uint8_t>(all.begin() + static_cast<std::ptrdiff_t>(offset),
                                         all.begin() +
                                             static_cast<std::ptrdiff_t>(offset + tensor.bytes()));
    };
    const auto fused_states = from_device<std::uint8_t>(state_storage, state_storage.bytes);
    expect_same(layer_bytes(reference_states, conv_states), layer_bytes(fused_states, conv_states),
                "convolution states");
    expect_same(layer_bytes(reference_states, recurrent_states),
                layer_bytes(fused_states, recurrent_states), "recurrent states");
    expect_same(from_device<std::uint8_t>(reference_out, reference_out.bytes),
                from_device<std::uint8_t>(fused_out, fused_out.bytes), "block output");
    const GdnReplayRecordLayer fused_layer = records.layer(tested_layer, batch);
    expect_same(reference_conv,
                from_device<std::uint8_t>(fused_layer.conv.data, fused_layer.conv.bytes()),
                "conv records");
    expect_same(reference_key,
                from_device<std::uint8_t>(fused_layer.key.data, fused_layer.key.bytes()),
                "key records");
    expect_same(reference_value,
                from_device<std::uint8_t>(fused_layer.value.data, fused_layer.value.bytes()),
                "value records");
    expect_same(reference_gate,
                from_device<std::uint8_t>(fused_layer.gate.data, fused_layer.gate.bytes()),
                "gate records");
    if (layer_bytes(initial_states, recurrent_states) ==
        layer_bytes(fused_states, recurrent_states)) {
        std::cerr << label << ": the pending fold left the recurrent states unchanged\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) { return 77; }
    try {
        int failures = run();
        // Pending entries in a different order from the verify rows; extents 2, 4, 0 and 1; a
        // verify row (slot 3) whose entry folds nothing and one (slot 4) with no entry.
        failures += run_pending_fold({.width          = 4,
                                      .previous_slots = {5, 0, 3, 1},
                                      .commits        = {2, 4, 0, 1},
                                      .sources        = {3, 0, 4, 5, 1},
                                      .valid          = {4, 3, 4, 2, 4}});
        failures += run_pending_fold(
            {.width = 2, .previous_slots = {2}, .commits = {2}, .sources = {2}, .valid = {2}});
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Flash-Next GDN\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Flash-Next GDN: " << error.what() << '\n';
        return 1;
    }
}
