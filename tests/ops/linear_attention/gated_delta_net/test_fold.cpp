#include "core/platform.h"
#include "ops/suite.h"
#include "ninfer/ops/gated_delta_net.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gdn_replay.h"

#include "core/decode_graph.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/weight.h"
#include "ops/input_projection_test_common.h"
#include "ops/linear_attention/gated_delta_net/launch.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kStateDim       = 128;
constexpr std::int32_t kRecordCapacity = 8;
constexpr std::size_t kGuardBytes      = 256;

std::uint32_t mix(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

float signed_pattern(std::uint32_t key, float magnitude = 0.06F) {
    const std::int32_t centered = static_cast<std::int32_t>(mix(key) % 2001U) - 1000;
    return static_cast<float>(centered) * (magnitude / 1000.0F);
}

std::uint16_t bf16_pattern(std::uint32_t key, float magnitude = 0.06F) {
    return f32_to_bf16(signed_pattern(key, magnitude));
}

void* offset_pointer(void* pointer, std::size_t bytes) {
    return static_cast<void*>(static_cast<std::byte*>(pointer) + bytes);
}

const void* offset_pointer(const void* pointer, std::size_t bytes) {
    return static_cast<const void*>(static_cast<const std::byte*>(pointer) + bytes);
}

struct FoldProfile {
    std::int32_t layers;
    std::int32_t value_heads;
    std::int32_t conv_channels;
};

std::vector<float> initial_recurrent_values(std::size_t elements, std::uint32_t seed, int layer,
                                            int row) {
    std::vector<float> values(elements,
                              signed_pattern(seed + 500009U + layer * 227U + row * 43U, 0.01F));
    values[0] = std::bit_cast<float>(0x80000000U);
    return values;
}

std::vector<std::int32_t> selected_slots(std::int32_t rows) {
    if (rows == 1) { return {2}; }
    std::vector<std::int32_t> slots{10, 2, 8, 0, 6, 4, 9, 1};
    slots.resize(static_cast<std::size_t>(rows));
    return slots;
}

int run_case(const FoldProfile profile, std::int32_t width, std::int32_t rows,
             const std::vector<std::int32_t>& commits, std::uint32_t seed,
             bool distinct_destination = false, bool captured = false, bool device_rows = false,
             std::int32_t record_capacity = kRecordCapacity) {
    const int kQkHeads                           = profile.value_heads == 24 ? 8 : 16;
    const std::vector<std::int32_t> source_slots = selected_slots(rows);
    std::vector<std::int32_t> destination_slots  = source_slots;
    if (distinct_destination)
        for (int row = 0; row < rows; ++row) destination_slots[row] = rows == 1 ? 1 : 11 + row;
    const std::int32_t slot_count = rows == 1 ? 3 : distinct_destination ? 11 + rows : 11;
    const std::size_t recurrent_slot_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * profile.value_heads;
    const std::size_t recurrent_slot_bytes = recurrent_slot_elements * sizeof(float);
    const std::size_t conv_slot_elements   = static_cast<std::size_t>(profile.conv_channels) * 3;
    const std::size_t conv_slot_bytes      = conv_slot_elements * sizeof(std::uint16_t);

    const GdnReplayRecordSpec record_spec{
        .layers          = profile.layers,
        .record_capacity = record_capacity,
        .width           = width,
        .conv_channels   = profile.conv_channels,
        .qk_heads        = kQkHeads,
        .value_heads     = profile.value_heads,
        .key_dim         = kStateDim,
        .value_dim       = kStateDim,
    };
    LayoutBuilder record_builder;
    const GdnReplayRecordLayout record_layout =
        plan_gdn_replay_records(record_builder, record_spec);
    const std::size_t record_bytes = record_builder.finish(256);
    DeviceBuffer record_storage(ninfer::test::cuda_environment().context.device(),
                                record_bytes + 2 * kGuardBytes);
    record_storage.fill(0xa5);
    void* record_base = offset_pointer(record_storage.data(), kGuardBytes);
    cuda_check(cudaMemset(record_base, 0xff, record_bytes), "initialize replay records");
    const GdnReplayRecords records(record_storage.span().subspan(kGuardBytes, record_bytes),
                                   record_layout);

    std::vector<std::uint16_t> conv_records(records.conv.numel(), 0xffffU);
    std::vector<std::uint16_t> key_records(records.key.numel(), 0xffffU);
    std::vector<std::uint16_t> value_records(records.value.numel(), 0xffffU);
    std::vector<std::uint32_t> gate_records(records.gate.numel(), 0xffffffffU);
    for (std::int32_t layer = 0; layer < profile.layers; ++layer) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::int32_t commit = commits[static_cast<std::size_t>(row)];
            const std::int64_t record_outer =
                static_cast<std::int64_t>(layer) * record_capacity + row;
            for (std::int32_t token = 0; token < (commit == 0 ? 0 : width); ++token) {
                const std::int64_t column = record_outer * width + token;
                for (std::int32_t channel = 0; channel < profile.conv_channels; ++channel) {
                    conv_records[static_cast<std::size_t>(column) * profile.conv_channels +
                                 channel] =
                        bf16_pattern(seed + layer * 131U + row * 17U + token * 7U + channel);
                }
                for (std::int32_t head = 0; head < kQkHeads; ++head) {
                    const std::size_t base =
                        static_cast<std::size_t>((column * kQkHeads + head) * kStateDim);
                    for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                        key_records[base + dim] =
                            bf16_pattern(seed + 100003U + layer * 197U + row * 23U + token * 11U +
                                             head * 5U + dim,
                                         0.08F);
                    }
                }
                for (std::int32_t head = 0; head < profile.value_heads; ++head) {
                    const std::size_t vector_base =
                        static_cast<std::size_t>((column * profile.value_heads + head) * kStateDim);
                    for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                        value_records[vector_base + dim] =
                            bf16_pattern(seed + 200003U + layer * 211U + row * 29U + token * 13U +
                                             head * 7U + dim,
                                         0.08F);
                    }
                    const std::size_t gate_base =
                        static_cast<std::size_t>((column * profile.value_heads + head) * 2);
                    const float g =
                        -0.03F -
                        static_cast<float>(mix(seed + layer * 31U + row * 17U + token * 7U + head) %
                                           900U) /
                            1000.0F;
                    const float beta =
                        0.05F + static_cast<float>(mix(seed + 300007U + layer * 37U + row * 19U +
                                                       token * 11U + head) %
                                                   900U) /
                                    1000.0F;
                    gate_records[gate_base]     = std::bit_cast<std::uint32_t>(g);
                    gate_records[gate_base + 1] = std::bit_cast<std::uint32_t>(beta);
                }
            }
        }
    }
    cuda_check(cudaMemcpy(records.conv.data, conv_records.data(), records.conv.bytes(),
                          cudaMemcpyHostToDevice),
               "upload conv records");
    cuda_check(cudaMemcpy(records.key.data, key_records.data(), records.key.bytes(),
                          cudaMemcpyHostToDevice),
               "upload key records");
    cuda_check(cudaMemcpy(records.value.data, value_records.data(), records.value.bytes(),
                          cudaMemcpyHostToDevice),
               "upload value records");
    cuda_check(cudaMemcpy(records.gate.data, gate_records.data(), records.gate.bytes(),
                          cudaMemcpyHostToDevice),
               "upload gate records");

    LayoutBuilder state_backing_builder;
    StateImageBuilder state_builder;
    const LinearAttentionStateLayout state_layout = plan_linear_attention_state(
        state_builder, {.layers         = static_cast<std::uint32_t>(profile.layers),
                        .conv_channels  = profile.conv_channels,
                        .conv_width     = 3,
                        .value_heads    = profile.value_heads,
                        .value_head_dim = kStateDim,
                        .key_head_dim   = kStateDim,
                        .conv_dtype     = DType::BF16});
    const auto image_layout       = state_builder.finish(state_backing_builder, slot_count);
    const std::size_t state_bytes = state_backing_builder.finish(256);
    DeviceBuffer state_storage(ninfer::test::cuda_environment().context.device(),
                               state_bytes + 2 * kGuardBytes);
    state_storage.fill(0xa5);
    void* state_base = offset_pointer(state_storage.data(), kGuardBytes);
    cuda_check(cudaMemset(state_base, 0, state_bytes), "initialize all-layer state");
    StateImageStorage images(state_storage.span().subspan(kGuardBytes, state_bytes), image_layout);
    LinearAttentionStateView state_pool(images, state_layout);

    DeviceBuffer expected_conv(ninfer::test::cuda_environment().context.device(),
                               static_cast<std::size_t>(profile.layers) * rows * conv_slot_bytes);
    expected_conv.fill(0);
    for (std::int32_t layer = 0; layer < profile.layers; ++layer) {
        for (std::int32_t row = 0; row < rows; ++row) {
            std::vector<std::uint16_t> initial(conv_slot_elements);
            for (std::int32_t history = 0; history < 3; ++history) {
                for (std::int32_t channel = 0; channel < profile.conv_channels; ++channel) {
                    initial[static_cast<std::size_t>(history) * profile.conv_channels + channel] =
                        bf16_pattern(seed + 400009U + layer * 223U + row * 41U + history * 13U +
                                         channel,
                                     0.05F);
                }
            }
            const Tensor destination = state_pool.conv_slot(
                static_cast<std::uint32_t>(layer), source_slots[static_cast<std::size_t>(row)]);
            cuda_check(cudaMemcpy(destination.data, initial.data(), conv_slot_bytes,
                                  cudaMemcpyHostToDevice),
                       "upload initial conv state");

            std::vector<std::uint16_t> expected = initial;
            const std::int32_t commit           = commits[static_cast<std::size_t>(row)];
            if (commit == 0 && destination_slots[row] != source_slots[row])
                std::fill(expected.begin(), expected.end(), 0);
            if (commit > 0) {
                const std::int64_t record_outer =
                    static_cast<std::int64_t>(layer) * record_capacity + row;
                for (std::int32_t channel = 0; channel < profile.conv_channels; ++channel) {
                    const auto record_value = [&](std::int32_t token) {
                        return conv_records[static_cast<std::size_t>(
                                                (record_outer * width + token) *
                                                profile.conv_channels) +
                                            channel];
                    };
                    if (commit == 1) {
                        expected[channel] = initial[profile.conv_channels + channel];
                        expected[profile.conv_channels + channel] =
                            initial[2LL * profile.conv_channels + channel];
                        expected[2LL * profile.conv_channels + channel] = record_value(0);
                    } else if (commit == 2) {
                        expected[channel] = initial[2LL * profile.conv_channels + channel];
                        expected[profile.conv_channels + channel]       = record_value(0);
                        expected[2LL * profile.conv_channels + channel] = record_value(1);
                    } else {
                        expected[channel]                               = record_value(commit - 3);
                        expected[profile.conv_channels + channel]       = record_value(commit - 2);
                        expected[2LL * profile.conv_channels + channel] = record_value(commit - 1);
                    }
                }
            }
            void* expected_destination =
                offset_pointer(expected_conv.data(),
                               static_cast<std::size_t>(layer * rows + row) * conv_slot_bytes);
            cuda_check(cudaMemcpy(expected_destination, expected.data(), conv_slot_bytes,
                                  cudaMemcpyHostToDevice),
                       "upload expected conv state");
        }
    }

    const std::size_t expected_recurrent_bytes =
        static_cast<std::size_t>(profile.layers) * rows * recurrent_slot_bytes;
    DeviceBuffer expected_recurrent(ninfer::test::cuda_environment().context.device(),
                                    expected_recurrent_bytes);
    expected_recurrent.fill(0);
    DeviceBuffer local_state(ninfer::test::cuda_environment().context.device(),
                             recurrent_slot_bytes);
    local_state.fill(0);
    const std::size_t q_elements = static_cast<std::size_t>(kStateDim) * kQkHeads * width;
    const std::size_t out_elements =
        static_cast<std::size_t>(kStateDim) * profile.value_heads * width;
    DeviceBuffer q(ninfer::test::cuda_environment().context.device(),
                   q_elements * sizeof(std::uint16_t));
    DeviceBuffer out(ninfer::test::cuda_environment().context.device(),
                     out_elements * sizeof(std::uint16_t));
    q.fill(0);
    DeviceBuffer g_row(ninfer::test::cuda_environment().context.device(),
                       static_cast<std::size_t>(profile.value_heads) * width * sizeof(float));
    DeviceBuffer beta_row(ninfer::test::cuda_environment().context.device(),
                          static_cast<std::size_t>(profile.value_heads) * width * sizeof(float));
    const Tensor q_tensor(q.span(), DType::BF16, {kStateDim, kQkHeads, width, 1});
    Tensor local_state_tensor(local_state.span(), DType::FP32,
                              {kStateDim, kStateDim, profile.value_heads});
    DeviceBuffer reference_slot = to_device(std::vector<std::int32_t>{0});
    Tensor selected_reference_slot(reference_slot.span(), DType::I32, {1});
    Tensor output(out.span(), DType::BF16, {kStateDim, profile.value_heads, width, 1});
    constexpr float kScale = 1.0F / std::sqrt(128.0F);

    for (std::int32_t layer = 0; layer < profile.layers; ++layer) {
        const GdnReplayRecordLayer layer_records = records.layer(layer, rows);
        for (std::int32_t row = 0; row < rows; ++row) {
            const auto initial_recurrent =
                initial_recurrent_values(recurrent_slot_elements, seed, layer, row);
            const Tensor actual_initial = state_pool.recurrent_slot(
                static_cast<std::uint32_t>(layer), source_slots[static_cast<std::size_t>(row)]);
            cuda_check(cudaMemcpy(actual_initial.data, initial_recurrent.data(),
                                  recurrent_slot_bytes, cudaMemcpyHostToDevice),
                       "upload initial recurrent state");
            cuda_check(cudaMemcpy(local_state.data(), initial_recurrent.data(),
                                  recurrent_slot_bytes, cudaMemcpyHostToDevice),
                       "upload local reference initial state");

            const std::int32_t commit = commits[static_cast<std::size_t>(row)];
            void* expected =
                offset_pointer(expected_recurrent.data(),
                               static_cast<std::size_t>(layer * rows + row) * recurrent_slot_bytes);
            if (commit == 0) {
                const Tensor untouched = state_pool.recurrent_slot(
                    static_cast<std::uint32_t>(layer), destination_slots[row]);
                cuda_check(cudaMemcpy(expected, untouched.data, recurrent_slot_bytes,
                                      cudaMemcpyDeviceToDevice),
                           "save untouched destination recurrent state");
                continue;
            }
            std::vector<float> g_host(static_cast<std::size_t>(profile.value_heads) * width);
            std::vector<float> beta_host(static_cast<std::size_t>(profile.value_heads) * width);
            const std::int64_t record_outer =
                static_cast<std::int64_t>(layer) * record_capacity + row;
            for (std::int32_t token = 0; token < width; ++token) {
                for (std::int32_t head = 0; head < profile.value_heads; ++head) {
                    const std::size_t source = static_cast<std::size_t>(
                        ((record_outer * width + token) * profile.value_heads + head) * 2);
                    const std::size_t destination =
                        static_cast<std::size_t>(token) * profile.value_heads + head;
                    g_host[destination]    = std::bit_cast<float>(gate_records[source]);
                    beta_host[destination] = std::bit_cast<float>(gate_records[source + 1]);
                }
            }
            g_row.copy_from_host(g_host.data(), g_row.size());
            beta_row.copy_from_host(beta_host.data(), beta_row.size());
            // Snapshot each transition of the same full W record block. Saving N must not
            // change the input width or regenerate any projection at N.
            for (int token = 0; token < width; ++token) {
                Tensor query = q_tensor.slice(2, token, 1).view({kStateDim, kQkHeads, 1});
                Tensor key   = layer_records.key.slice(3, row, 1)
                                   .slice(2, token, 1)
                                   .view({kStateDim, kQkHeads, 1});
                Tensor value = layer_records.value.slice(3, row, 1)
                                   .slice(2, token, 1)
                                   .view({kStateDim, profile.value_heads, 1});
                Tensor g_tensor(g_row.span().subspan(std::size_t(token) * profile.value_heads * 4),
                                DType::FP32, {profile.value_heads, 1});
                Tensor beta_tensor(
                    beta_row.span().subspan(std::size_t(token) * profile.value_heads * 4),
                    DType::FP32, {profile.value_heads, 1});
                Tensor output_token =
                    output.slice(2, token, 1).view({kStateDim, profile.value_heads, 1});
                ops::gated_delta_net_batch_update(
                    query, key, value, g_tensor, beta_tensor, kScale, true, local_state_tensor,
                    selected_reference_slot, selected_reference_slot, output_token,
                    ninfer::test::cuda_environment().default_execution);
                if (token + 1 == commit)
                    cuda_check(cudaMemcpyAsync(expected, local_state.data(), recurrent_slot_bytes,
                                               cudaMemcpyDeviceToDevice, nullptr),
                               "save Nth snapshot");
            }
        }
    }

    const std::vector<std::uint8_t> records_before =
        from_device<std::uint8_t>(record_storage, record_storage.size());
    std::vector<ops::GdnReplayFoldRow> fold_rows(static_cast<std::size_t>(rows));
    for (std::int32_t row = 0; row < rows; ++row) {
        fold_rows[static_cast<std::size_t>(row)] = {
            source_slots[static_cast<std::size_t>(row)],
            destination_slots[static_cast<std::size_t>(row)],
            commits[static_cast<std::size_t>(row)]};
    }
    const ops::GdnReplayFoldPlan fold_plan(records, state_pool.all_layers_view());
    auto device_sources      = to_device(source_slots);
    auto device_destinations = to_device(destination_slots);
    auto device_commits      = to_device(commits);
    const Tensor source_metadata(device_sources.span(), DType::I32, {rows});
    const Tensor destination_metadata(device_destinations.span(), DType::I32, {rows});
    const Tensor commit_metadata(device_commits.span(), DType::I32, {rows});
    const auto execute = [&](CudaExecutionView cuda) {
        if (device_rows)
            fold_plan.execute(source_metadata, destination_metadata, commit_metadata, cuda);
        else
            fold_plan.execute(fold_rows, cuda);
    };
    if (captured || (width == 16 && rows == 8)) {
        cuda_synchronize();
        DeviceBuffer original(ninfer::test::cuda_environment().context.device(), state_bytes);
        DeviceContext context;
        cuda_check(cudaMemcpyAsync(original.data(), state_base, state_bytes,
                                   cudaMemcpyDeviceToDevice, context.stream()),
                   "save graph initial state");
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        if (device_rows) device_commits.fill(0);
        definition.capture(context.stream(), [&] { execute(context.execution_view()); });
        graph.instantiate(definition);
        if (device_rows)
            device_commits.copy_from_host(commits.data(), commits.size() * sizeof(int));
        // Host row descriptors are captured by value. Mutating them after capture must
        // not alter the graph's accepted prefixes. GPU state remains a live operand.
        for (auto& row : fold_rows) row.commit_columns = 0;
        // Restore GPU state between replays.
        for (int replay = 0; replay < 2; ++replay) {
            cuda_check(cudaMemcpyAsync(state_base, original.data(), state_bytes,
                                       cudaMemcpyDeviceToDevice, context.stream()),
                       "restore graph state");
            graph.launch(context.stream());
            context.synchronize();
        }
    } else {
        execute(ninfer::test::cuda_environment().default_execution);
        cuda_synchronize();
    }

    int failures             = 0;
    const std::string suffix = " L=" + std::to_string(profile.layers) +
                               " Hv=" + std::to_string(profile.value_heads) +
                               " T=" + std::to_string(width) + " B=" + std::to_string(rows);
    std::vector<float> actual_recurrent(recurrent_slot_elements);
    std::vector<float> expected_recurrent_host(recurrent_slot_elements);
    std::vector<std::uint16_t> actual_conv(conv_slot_elements);
    std::vector<std::uint16_t> expected_conv_host(conv_slot_elements);
    for (std::int32_t layer = 0; layer < profile.layers; ++layer) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const Tensor actual_state =
                state_pool.recurrent_slot(static_cast<std::uint32_t>(layer),
                                          destination_slots[static_cast<std::size_t>(row)]);
            cuda_check(cudaMemcpy(actual_recurrent.data(), actual_state.data, recurrent_slot_bytes,
                                  cudaMemcpyDeviceToHost),
                       "download folded recurrent state");
            const void* expected_state =
                offset_pointer(expected_recurrent.data(),
                               static_cast<std::size_t>(layer * rows + row) * recurrent_slot_bytes);
            cuda_check(cudaMemcpy(expected_recurrent_host.data(), expected_state,
                                  recurrent_slot_bytes, cudaMemcpyDeviceToHost),
                       "download expected recurrent state");
            if (std::memcmp(actual_recurrent.data(), expected_recurrent_host.data(),
                            recurrent_slot_bytes) != 0) {
                std::cerr << "fold recurrent state differs from Nth snapshot" << suffix
                          << " layer=" << layer << " row=" << row << "\n";
                return failures + 1;
            }

            const Tensor actual_history =
                state_pool.conv_slot(static_cast<std::uint32_t>(layer),
                                     destination_slots[static_cast<std::size_t>(row)]);
            cuda_check(cudaMemcpy(actual_conv.data(), actual_history.data, conv_slot_bytes,
                                  cudaMemcpyDeviceToHost),
                       "download folded conv state");
            const void* expected_history =
                offset_pointer(expected_conv.data(),
                               static_cast<std::size_t>(layer * rows + row) * conv_slot_bytes);
            cuda_check(cudaMemcpy(expected_conv_host.data(), expected_history, conv_slot_bytes,
                                  cudaMemcpyDeviceToHost),
                       "download expected conv state");
            if (actual_conv != expected_conv_host) {
                std::cerr << "fold conv history differs" << suffix << " layer=" << layer
                          << " row=" << row << "\n";
                return failures + 1;
            }
            if (destination_slots[row] != source_slots[row]) {
                const auto initial_values =
                    initial_recurrent_values(recurrent_slot_elements, seed, layer, row);
                const Tensor source_state =
                    state_pool.recurrent_slot(static_cast<std::uint32_t>(layer), source_slots[row]);
                const auto source_recurrent =
                    from_device<float>(source_state.data, recurrent_slot_elements);
                if (std::memcmp(source_recurrent.data(), initial_values.data(),
                                recurrent_slot_bytes) != 0) {
                    std::cerr << "fold modified recurrent source" << suffix << " layer=" << layer
                              << " row=" << row << "\n";
                    return failures + 1;
                }
                std::vector<std::uint16_t> initial_conv(conv_slot_elements);
                for (std::int32_t history = 0; history < 3; ++history) {
                    for (std::int32_t channel = 0; channel < profile.conv_channels; ++channel) {
                        initial_conv[static_cast<std::size_t>(history) * profile.conv_channels +
                                     channel] =
                            bf16_pattern(seed + 400009U + layer * 223U + row * 41U + history * 13U +
                                             channel,
                                         0.05F);
                    }
                }
                const Tensor source_history = state_pool.conv_slot(
                    static_cast<std::uint32_t>(layer), source_slots[static_cast<std::size_t>(row)]);
                if (from_device<std::uint16_t>(source_history.data, conv_slot_elements) !=
                    initial_conv) {
                    std::cerr << "fold modified convolution source" << suffix << " layer=" << layer
                              << " row=" << row << "\n";
                    return failures + 1;
                }
            }
        }
    }

    std::vector<float> inactive_recurrent(recurrent_slot_elements);
    std::vector<std::uint16_t> inactive_conv(conv_slot_elements);
    for (std::int32_t layer = 0; layer < profile.layers; ++layer) {
        for (std::int32_t slot = 0; slot < slot_count; ++slot) {
            if (std::find(source_slots.begin(), source_slots.end(), slot) != source_slots.end() ||
                std::find(destination_slots.begin(), destination_slots.end(), slot) !=
                    destination_slots.end()) {
                continue;
            }
            const Tensor recurrent =
                state_pool.recurrent_slot(static_cast<std::uint32_t>(layer), slot);
            cuda_check(cudaMemcpy(inactive_recurrent.data(), recurrent.data, recurrent_slot_bytes,
                                  cudaMemcpyDeviceToHost),
                       "download inactive recurrent state");
            if (!std::all_of(inactive_recurrent.begin(), inactive_recurrent.end(), [](float value) {
                    return std::bit_cast<std::uint32_t>(value) == 0;
                })) {
                std::cerr << "fold modified inactive recurrent slot" << suffix << " layer=" << layer
                          << " slot=" << slot << "\n";
                return failures + 1;
            }
            const Tensor conv = state_pool.conv_slot(static_cast<std::uint32_t>(layer), slot);
            cuda_check(cudaMemcpy(inactive_conv.data(), conv.data, conv_slot_bytes,
                                  cudaMemcpyDeviceToHost),
                       "download inactive conv state");
            if (!std::all_of(inactive_conv.begin(), inactive_conv.end(),
                             [](std::uint16_t value) { return value == 0; })) {
                std::cerr << "fold modified inactive conv slot" << suffix << " layer=" << layer
                          << " slot=" << slot << "\n";
                return failures + 1;
            }
        }
    }

    const std::vector<std::uint8_t> records_after =
        from_device<std::uint8_t>(record_storage, record_storage.size());
    if (records_after != records_before) {
        std::cerr << "fold modified record storage" << suffix << "\n";
        ++failures;
    }
    const auto state_guard_before = from_device<std::uint8_t>(state_storage.data(), kGuardBytes);
    const auto state_guard_after  = from_device<std::uint8_t>(
        offset_pointer(state_storage.data(), kGuardBytes + state_bytes), kGuardBytes);
    if (!std::all_of(state_guard_before.begin(), state_guard_before.end(),
                     [](auto byte) { return byte == 0xa5; }) ||
        !std::all_of(state_guard_after.begin(), state_guard_after.end(),
                     [](auto byte) { return byte == 0xa5; })) {
        std::cerr << "fold modified state outer guard" << suffix << "\n";
        ++failures;
    }
    return failures;
}

int run_record_fold_rounds(bool q8 = false, bool local = false) {
    using ninfer::test::input_projection::DevicePackedWeight;
    using ninfer::test::input_projection::make_bf16_activation;

    const FoldProfile kProfile{q8 ? 36 : 48, local ? 24 : 48, local ? 5120 : 10240};
    const int kQkHeads                   = local ? 8 : 16;
    const int kQueryRows                 = 128 * kQkHeads;
    const std::int32_t kHidden           = q8 ? 2560 : 5120;
    const std::int32_t kValueRows        = local ? 3072 : 6144;
    const std::int32_t kZRows            = local ? 3072 : 6144;
    constexpr std::int32_t kWidth        = 16;
    constexpr std::int32_t kStateSlots   = kWidth + 1;
    constexpr std::int32_t kInitialSlot  = kWidth;
    constexpr std::int32_t kSnapshotBase = 0;
    constexpr float kScale               = 1.0F / std::sqrt(128.0F);

    DevicePackedWeight qk_parent(quantized_weight::make_patterned_weight(
        q8 ? QType::Q8_G32_FP16 : QType::Q4_G64_FP16, q8 ? (local ? 8192 : 16384) : 4096, kHidden,
        1901U));
    std::optional<DevicePackedWeight> vz_parent;
    if (!q8)
        vz_parent.emplace(
            quantized_weight::make_patterned_weight(QType::Q5_G64_FP16, 12288, kHidden, 1902U));
    const std::vector<float> activation = make_bf16_activation(kHidden, kWidth, 1903U);
    DeviceBuffer device_x               = to_device_bf16(activation);
    std::vector<std::uint16_t> conv_weight_bits(static_cast<std::size_t>(kProfile.conv_channels) *
                                                4);
    for (std::size_t index = 0; index < conv_weight_bits.size(); ++index) {
        conv_weight_bits[index] = bf16_pattern(1907U + static_cast<std::uint32_t>(index), 0.02F);
    }
    DeviceBuffer device_conv_weight = to_device(conv_weight_bits);
    DeviceBuffer device_initial     = to_device(std::vector<std::int32_t>{kInitialSlot});
    DeviceBuffer device_snapshot    = to_device(std::vector<std::int32_t>{kSnapshotBase});

    LayoutBuilder record_builder;
    const GdnReplayRecordLayout record_layout =
        plan_gdn_replay_records(record_builder, {.layers          = kProfile.layers,
                                                 .record_capacity = 1,
                                                 .width           = kWidth,
                                                 .conv_channels   = kProfile.conv_channels,
                                                 .qk_heads        = kQkHeads,
                                                 .value_heads     = kProfile.value_heads,
                                                 .key_dim         = kStateDim,
                                                 .value_dim       = kStateDim});
    DeviceBuffer record_storage(ninfer::test::cuda_environment().context.device(),
                                record_builder.finish(256));
    record_storage.fill(0xff);
    GdnReplayRecords records(record_storage.span(), record_layout);

    LayoutBuilder state_backing_builder;
    StateImageBuilder state_builder;
    const LinearAttentionStateLayout state_layout = plan_linear_attention_state(
        state_builder, {.layers         = static_cast<std::uint32_t>(kProfile.layers),
                        .conv_channels  = kProfile.conv_channels,
                        .conv_width     = 3,
                        .value_heads    = kProfile.value_heads,
                        .value_head_dim = kStateDim,
                        .key_head_dim   = kStateDim,
                        .conv_dtype     = DType::BF16});
    const auto image_layout = state_builder.finish(state_backing_builder, kStateSlots);
    DeviceBuffer state_storage(ninfer::test::cuda_environment().context.device(),
                               state_backing_builder.finish(256));
    state_storage.fill(0);
    StateImageStorage images(state_storage.span(), image_layout);
    LinearAttentionStateView state_pool(images, state_layout);
    const ops::GdnReplayFoldPlan fold_plan(records, state_pool.all_layers_view());

    const std::size_t recurrent_slot_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * kProfile.value_heads;
    const std::size_t conv_slot_elements = static_cast<std::size_t>(kProfile.conv_channels) * 3;
    for (std::int32_t layer = 0; layer < kProfile.layers; ++layer) {
        const float initial_value = signed_pattern(1911U + layer * 47U, 0.01F);
        const std::vector<float> recurrent(recurrent_slot_elements, initial_value);
        const Tensor recurrent_slot =
            state_pool.recurrent_slot(static_cast<std::uint32_t>(layer), kInitialSlot);
        cuda_check(cudaMemcpy(recurrent_slot.data, recurrent.data(), recurrent_slot.bytes(),
                              cudaMemcpyHostToDevice),
                   "upload pair initial recurrent state");

        std::vector<std::uint16_t> conv(conv_slot_elements);
        for (std::int32_t history = 0; history < 3; ++history) {
            for (std::int32_t channel = 0; channel < kProfile.conv_channels; ++channel) {
                conv[static_cast<std::size_t>(history) * kProfile.conv_channels + channel] =
                    bf16_pattern(1913U + layer * 53U + history * 11U + channel, 0.04F);
            }
        }
        const Tensor conv_slot =
            state_pool.conv_slot(static_cast<std::uint32_t>(layer), kInitialSlot);
        cuda_check(
            cudaMemcpy(conv_slot.data, conv.data(), conv_slot.bytes(), cudaMemcpyHostToDevice),
            "upload pair initial conv state");
    }

    DeviceBuffer snapshot_q(ninfer::test::cuda_environment().context.device(),
                            static_cast<std::size_t>(kQueryRows) * kWidth * sizeof(std::uint16_t));
    DeviceBuffer snapshot_k(ninfer::test::cuda_environment().context.device(),
                            static_cast<std::size_t>(kQueryRows) * kWidth * sizeof(std::uint16_t));
    DeviceBuffer snapshot_v(ninfer::test::cuda_environment().context.device(),
                            static_cast<std::size_t>(kValueRows) * kWidth * sizeof(std::uint16_t));
    DeviceBuffer snapshot_z(ninfer::test::cuda_environment().context.device(),
                            static_cast<std::size_t>(kZRows) * kWidth * sizeof(std::uint16_t));
    DeviceBuffer record_q(ninfer::test::cuda_environment().context.device(), snapshot_q.size());
    DeviceBuffer record_k(ninfer::test::cuda_environment().context.device(), snapshot_k.size());
    DeviceBuffer record_v(ninfer::test::cuda_environment().context.device(), snapshot_v.size());
    DeviceBuffer record_z(ninfer::test::cuda_environment().context.device(), snapshot_z.size());
    DeviceBuffer snapshot_out(ninfer::test::cuda_environment().context.device(), snapshot_v.size());
    DeviceBuffer record_out(ninfer::test::cuda_environment().context.device(), snapshot_v.size());
    DeviceBuffer device_g(ninfer::test::cuda_environment().context.device(),
                          static_cast<std::size_t>(kProfile.value_heads) * kWidth * sizeof(float));
    DeviceBuffer device_beta(ninfer::test::cuda_environment().context.device(), device_g.size());

    Tensor x(device_x.span(), DType::BF16, {kHidden, kWidth, 1});
    Tensor conv_weight(device_conv_weight.span(), DType::BF16, {kProfile.conv_channels, 4});
    Tensor initial_selector(device_initial.span(), DType::I32, {1});
    Tensor snapshot_selector(device_snapshot.span(), DType::I32, {1});
    Tensor valid;
    Tensor snapshot_query(snapshot_q.span(), DType::BF16, {kQueryRows, kWidth, 1});
    Tensor snapshot_key(snapshot_k.span(), DType::BF16, {kQueryRows, kWidth, 1});
    Tensor snapshot_value(snapshot_v.span(), DType::BF16, {kValueRows, kWidth, 1});
    Tensor snapshot_z_tensor(snapshot_z.span(), DType::BF16, {kZRows, kWidth, 1});
    Tensor record_query(record_q.span(), DType::BF16, {kQueryRows, kWidth, 1});
    Tensor record_key(record_k.span(), DType::BF16, {kQueryRows, kWidth, 1});
    Tensor record_value(record_v.span(), DType::BF16, {kValueRows, kWidth, 1});
    Tensor record_z_tensor(record_z.span(), DType::BF16, {kZRows, kWidth, 1});
    Tensor snapshot_output(snapshot_out.span(), DType::BF16,
                           {kStateDim, kProfile.value_heads, kWidth, 1});
    Tensor record_output(record_out.span(), DType::BF16,
                         {kStateDim, kProfile.value_heads, kWidth, 1});
    Tensor g(device_g.span(), DType::FP32, {kProfile.value_heads, kWidth, 1});
    Tensor beta(device_beta.span(), DType::FP32, {kProfile.value_heads, kWidth, 1});

    const std::size_t snapshot_workspace_bytes =
        q8 ? ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                 ops::SingleProjectionWeight{qk_parent.view(), ops::LinearPolicy::A16Only}, 1,
                 kWidth, kWidth)
           : ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                 ops::PairedProjectionWeights{qk_parent.view(), vz_parent->view()}, 1, kWidth,
                 kWidth);
    const std::size_t record_workspace_bytes =
        q8 ? ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                 ops::SingleProjectionWeight{qk_parent.view(), ops::LinearPolicy::A16Only}, 1,
                 kWidth, kWidth)
           : ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                 ops::PairedProjectionWeights{qk_parent.view(), vz_parent->view()}, 1, kWidth,
                 kWidth);
    ninfer::DeviceBuffer snapshot_workspace_storage(
        ninfer::test::cuda_environment().context.device(),
        std::max<std::size_t>(256, snapshot_workspace_bytes));
    MemoryArena snapshot_workspace(snapshot_workspace_storage.span());
    ninfer::DeviceBuffer record_workspace_storage(
        ninfer::test::cuda_environment().context.device(),
        std::max<std::size_t>(256, record_workspace_bytes));
    MemoryArena record_workspace(record_workspace_storage.span());

    std::vector<std::int32_t> source_steps(kWidth), destination_steps(kWidth);
    for (int token = 0; token < kWidth; ++token) {
        source_steps[token]      = token == 0 ? kInitialSlot : token - 1;
        destination_steps[token] = token;
    }
    DeviceBuffer device_sources      = to_device(source_steps);
    DeviceBuffer device_destinations = to_device(destination_steps);

    int failures = 0;
    for (std::int32_t round = 0; round < 3; ++round) {
        const std::int32_t commit = round == 0 ? 1 : round == 1 ? kWidth - 1 : kWidth;
        std::vector<float> g_host(static_cast<std::size_t>(kProfile.value_heads) * kWidth);
        std::vector<float> beta_host(g_host.size());
        for (std::size_t index = 0; index < g_host.size(); ++index) {
            g_host[index]    = -0.04F - static_cast<float>((index + round * 17) % 80) / 100.0F;
            beta_host[index] = 0.08F + static_cast<float>((index * 7 + round * 13) % 80) / 100.0F;
        }
        device_g.copy_from_host(g_host.data(), device_g.size());
        device_beta.copy_from_host(beta_host.data(), device_beta.size());

        for (std::int32_t layer = 0; layer < kProfile.layers; ++layer) {
            GdnReplayRecordLayer layer_records = records.layer(layer, 1);
            Tensor conv_states = state_pool.layer_view(static_cast<std::uint32_t>(layer)).conv;
            if (q8) {
                ops::gdn_input_proj_conv_snapshot(
                    x, ops::SingleProjectionWeight{qk_parent.view(), ops::LinearPolicy::A16Only},
                    conv_weight, {conv_states, initial_selector, snapshot_selector}, valid,
                    {snapshot_query, snapshot_key, snapshot_value, snapshot_z_tensor},
                    snapshot_workspace, ninfer::test::cuda_environment().default_execution);
                ops::gdn_input_proj_conv_record(
                    x, ops::SingleProjectionWeight{qk_parent.view(), ops::LinearPolicy::A16Only},
                    conv_weight, {conv_states, initial_selector}, valid,
                    {record_query, record_key, record_value, record_z_tensor}, layer_records.conv,
                    record_workspace, ninfer::test::cuda_environment().default_execution);
            } else {
                ops::gdn_input_proj_conv_snapshot(
                    x, ops::PairedProjectionWeights{qk_parent.view(), vz_parent->view()},
                    conv_weight, {conv_states, initial_selector, snapshot_selector}, valid,
                    {snapshot_query, snapshot_key, snapshot_value, snapshot_z_tensor},
                    snapshot_workspace, ninfer::test::cuda_environment().default_execution);
                ops::gdn_input_proj_conv_record(
                    x, ops::PairedProjectionWeights{qk_parent.view(), vz_parent->view()},
                    conv_weight, {conv_states, initial_selector}, valid,
                    {record_query, record_key, record_value, record_z_tensor}, layer_records.conv,
                    record_workspace, ninfer::test::cuda_environment().default_execution);
            }

            Tensor snapshot_q_view = snapshot_query.view({kStateDim, kQkHeads, kWidth, 1});
            Tensor snapshot_k_view = snapshot_key.view({kStateDim, kQkHeads, kWidth, 1});
            Tensor snapshot_v_view =
                snapshot_value.view({kStateDim, kProfile.value_heads, kWidth, 1});
            Tensor record_q_view = record_query.view({kStateDim, kQkHeads, kWidth, 1});
            Tensor record_k_view = record_key.view({kStateDim, kQkHeads, kWidth, 1});
            Tensor record_v_view = record_value.view({kStateDim, kProfile.value_heads, kWidth, 1});
            Tensor recurrent_states =
                state_pool.layer_view(static_cast<std::uint32_t>(layer)).recurrent;
            // The projection is computed at W once above. Materialize all W recurrent
            // snapshots from those exact q/k/v columns, then select N-1 after folding.
            for (int token = 0; token < kWidth; ++token) {
                Tensor q_step = snapshot_q_view.slice(2, token, 1);
                Tensor k_step = snapshot_k_view.slice(2, token, 1);
                Tensor v_step = snapshot_v_view.slice(2, token, 1);
                Tensor g_step = g.slice(1, token, 1), beta_step = beta.slice(1, token, 1);
                Tensor out_step = snapshot_output.slice(2, token, 1);
                Tensor source(device_sources.span().subspan(std::size_t(token) * 4), DType::I32,
                              {1});
                Tensor destination(device_destinations.span().subspan(std::size_t(token) * 4),
                                   DType::I32, {1});
                ops::gated_delta_net_batch_update(
                    q_step, k_step, v_step, g_step, beta_step, kScale, true, recurrent_states,
                    source, destination, out_step,
                    ninfer::test::cuda_environment().default_execution);
            }
            ops::gated_delta_net_replay_record(
                record_q_view, record_k_view, record_v_view, g, beta, kScale, recurrent_states,
                valid, initial_selector,
                {layer_records.key, layer_records.value, layer_records.gate}, record_output,
                ninfer::test::cuda_environment().default_execution);
            cuda_synchronize();

            const std::string label = "record-fold pair round=" + std::to_string(round) +
                                      " layer=" + std::to_string(layer);
            const auto compare_bf16 = [&](const DeviceBuffer& lhs, const DeviceBuffer& rhs,
                                          std::size_t elements, const char* field) {
                if (from_device<std::uint16_t>(lhs, elements) !=
                    from_device<std::uint16_t>(rhs, elements)) {
                    std::cerr << label << " " << field << " differs\n";
                    return 1;
                }
                return 0;
            };
            failures += compare_bf16(snapshot_q, record_q,
                                     static_cast<std::size_t>(kQueryRows) * kWidth, "query");
            failures += compare_bf16(snapshot_k, record_k,
                                     static_cast<std::size_t>(kQueryRows) * kWidth, "key");
            failures += compare_bf16(snapshot_v, record_v,
                                     static_cast<std::size_t>(kValueRows) * kWidth, "value");
            failures +=
                compare_bf16(snapshot_z, record_z, static_cast<std::size_t>(kZRows) * kWidth, "z");
            failures +=
                compare_bf16(snapshot_out, record_out,
                             static_cast<std::size_t>(kValueRows) * kWidth, "recurrent output");
            if (failures != 0) { return failures; }
        }

        const std::array fold_rows{ops::GdnReplayFoldRow{kInitialSlot, kInitialSlot, commit}};
        fold_plan.execute(fold_rows, ninfer::test::cuda_environment().default_execution);
        cuda_synchronize();
        for (std::int32_t layer = 0; layer < kProfile.layers; ++layer) {
            const Tensor folded_recurrent =
                state_pool.recurrent_slot(static_cast<std::uint32_t>(layer), kInitialSlot);
            const Tensor snapshot_recurrent =
                state_pool.recurrent_slot(static_cast<std::uint32_t>(layer), commit - 1);
            if (from_device<std::uint32_t>(folded_recurrent.data, recurrent_slot_elements) !=
                from_device<std::uint32_t>(snapshot_recurrent.data, recurrent_slot_elements)) {
                std::cerr << "record-fold recurrent mismatch round=" << round << " layer=" << layer
                          << "\n";
                return failures + 1;
            }
            const Tensor folded_conv =
                state_pool.conv_slot(static_cast<std::uint32_t>(layer), kInitialSlot);
            const Tensor snapshot_conv =
                state_pool.conv_slot(static_cast<std::uint32_t>(layer), commit - 1);
            if (from_device<std::uint16_t>(folded_conv.data, conv_slot_elements) !=
                from_device<std::uint16_t>(snapshot_conv.data, conv_slot_elements)) {
                std::cerr << "record-fold conv mismatch round=" << round << " layer=" << layer
                          << "\n";
                return failures + 1;
            }
        }
    }
    failures += qk_parent.verify_preserved(q8 ? "record-fold Q8 parent" : "record-fold Q4 parent");
    if (vz_parent) failures += vz_parent->verify_preserved("record-fold Q5 parent");
    return failures;
}

// Desynced two-row record/fold regression (Df2Positions): row0 sits one sequence position ahead
// of row1, so one 16-column record op runs with row0 carrying a one-column pending fold (in
// place) and row1 carrying none, and the round's commit folds a different accepted prefix per
// row. Every row's committed state, record copy, and token output must equal its own
// single-stream per-token state-pool chain from the same initial state, built only from that
// row's record columns.
int run_desynced_rounds() {
    const FoldProfile kProfile{36, 24, 5120};
    const int kQkHeads                     = 8;
    constexpr std::int32_t kWidth          = 16;
    constexpr std::int32_t kRows           = 2;
    constexpr std::int32_t kRecordCapacity = 2;
    constexpr std::uint32_t kSeed          = 6100U;
    constexpr float kScale                 = 1.0F / std::sqrt(128.0F);
    // Slots: 0 spare, 1 row0 (in place), 2 row1 source, 3 row1 destination.
    constexpr std::int32_t kRow0Slot        = 1;
    constexpr std::int32_t kRow1Slot        = 2;
    constexpr std::int32_t kRow1Destination = 3;
    constexpr std::int32_t kSlotCount       = 4;
    // Row0 accepted one extra token in the previous round (it is at position p+1 while row1 is
    // at p); the round then commits desynced accepted prefixes per row.
    constexpr std::int32_t kRow0Pending = 1;
    constexpr std::int32_t kRow0Commit  = 5;
    constexpr std::int32_t kRow1Commit  = 4;

    const std::size_t recurrent_slot_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * kProfile.value_heads;
    const std::size_t recurrent_slot_bytes = recurrent_slot_elements * sizeof(float);
    const std::size_t conv_slot_elements =
        static_cast<std::size_t>(kProfile.conv_channels) * 3;
    const std::size_t conv_slot_bytes = conv_slot_elements * sizeof(std::uint16_t);
    const std::size_t columns = static_cast<std::size_t>(kWidth) * kRows;
    const std::size_t token_values = static_cast<std::size_t>(kStateDim) * kProfile.value_heads;
    const std::size_t heads = static_cast<std::size_t>(kProfile.value_heads);

    const GdnReplayRecordSpec record_spec{.layers          = kProfile.layers,
                                          .record_capacity = kRecordCapacity,
                                          .width           = kWidth,
                                          .conv_channels   = kProfile.conv_channels,
                                          .qk_heads        = kQkHeads,
                                          .value_heads     = kProfile.value_heads,
                                          .key_dim         = kStateDim,
                                          .value_dim       = kStateDim};
    LayoutBuilder record_builder;
    const GdnReplayRecordLayout record_layout =
        plan_gdn_replay_records(record_builder, record_spec);
    DeviceBuffer record_storage(ninfer::test::cuda_environment().context.device(),
                                record_builder.finish(256));
    record_storage.fill(0);
    GdnReplayRecords records(record_storage.span(), record_layout);

    LayoutBuilder state_backing_builder;
    StateImageBuilder state_builder;
    const LinearAttentionStateLayout state_layout = plan_linear_attention_state(
        state_builder, {.layers         = static_cast<std::uint32_t>(kProfile.layers),
                        .conv_channels  = kProfile.conv_channels,
                        .conv_width     = 3,
                        .value_heads    = kProfile.value_heads,
                        .value_head_dim = kStateDim,
                        .key_head_dim   = kStateDim,
                        .conv_dtype     = DType::BF16});
    const auto image_layout = state_builder.finish(state_backing_builder, kSlotCount);
    DeviceBuffer state_storage(ninfer::test::cuda_environment().context.device(),
                               state_backing_builder.finish(256));
    state_storage.fill(0);
    StateImageStorage images(state_storage.span(), image_layout);
    LinearAttentionStateView state_pool(images, state_layout);

    // Distinct represented patterns per (layer, row, token). Gates are token-major: element
    // (token, head) at token * heads + head, exactly as the record kernels consume them.
    const std::size_t qk_elements = static_cast<std::size_t>(kStateDim) * kQkHeads * columns;
    const std::size_t value_elements = token_values * columns;
    const std::size_t gate_elements = heads * columns;
    std::vector<std::uint16_t> q_bits(qk_elements), k_bits(qk_elements), v_bits(value_elements);
    std::vector<float> g_bits(gate_elements), beta_bits(gate_elements);
    std::vector<std::uint16_t> conv_bits(static_cast<std::size_t>(kProfile.layers) *
                                         kRecordCapacity * kWidth * kProfile.conv_channels);
    for (std::int32_t layer = 0; layer < kProfile.layers; ++layer) {
        for (std::int32_t row = 0; row < kRows; ++row) {
            const std::int64_t record_outer =
                static_cast<std::int64_t>(layer) * kRecordCapacity + row;
            for (std::int32_t token = 0; token < kWidth; ++token) {
                const std::size_t column = static_cast<std::size_t>(record_outer * kWidth + token);
                for (std::int32_t head = 0; head < kQkHeads; ++head) {
                    const std::size_t base = ((column * kQkHeads + head) * kStateDim);
                    for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                        q_bits[base + dim] = bf16_pattern(kSeed + 100003U + layer * 197U +
                                                              row * 23U + token * 11U + head * 5U +
                                                              dim,
                                                          0.08F);
                        k_bits[base + dim] = bf16_pattern(kSeed + 200003U + layer * 211U +
                                                              row * 29U + token * 13U + head * 7U +
                                                              dim,
                                                          0.08F);
                    }
                }
                for (std::int32_t head = 0; head < kProfile.value_heads; ++head) {
                    const std::size_t vector_base =
                        (column * kProfile.value_heads + head) * kStateDim;
                    for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                        v_bits[vector_base + dim] = bf16_pattern(kSeed + 300003U + layer * 227U +
                                                                     row * 31U + token * 17U +
                                                                     head * 11U + dim,
                                                                 0.08F);
                    }
                    const std::size_t gate_index = column * heads + head;
                    g_bits[gate_index] =
                        -0.04F -
                        static_cast<float>(mix(kSeed + layer * 41U + row * 19U + token * 7U +
                                               head) %
                                           900U) /
                            1000.0F;
                    beta_bits[gate_index] =
                        0.06F +
                        static_cast<float>(mix(kSeed + 400003U + layer * 43U + row * 23U +
                                               token * 9U + head) %
                                           900U) /
                            1000.0F;
                }
                for (std::int32_t channel = 0; channel < kProfile.conv_channels; ++channel) {
                    conv_bits[column * kProfile.conv_channels + channel] =
                        bf16_pattern(kSeed + 500003U + layer * 131U + row * 17U + token * 7U +
                                         channel,
                                     0.05F);
                }
            }
        }
    }
    cuda_check(cudaMemcpy(records.conv.data, conv_bits.data(), records.conv.bytes(),
                          cudaMemcpyHostToDevice),
               "upload conv records");

    // The previous round's records (the pending fold's source): row0 carries its extra accepted
    // token at record row 0; the gate record interleaves {g, beta} per (column, head).
    std::vector<std::uint16_t> pending_q(qk_elements), pending_k(qk_elements),
        pending_v(value_elements);
    std::vector<float> pending_gate(gate_elements * 2);
    for (std::int32_t token = 0; token < kWidth; ++token) {
        const std::size_t column = static_cast<std::size_t>(token);
        for (std::int32_t head = 0; head < kQkHeads; ++head) {
            for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                const std::size_t base = ((column * kQkHeads + head) * kStateDim) + dim;
                pending_q[base] =
                    bf16_pattern(kSeed + 600003U + token * 11U + head * 5U + dim, 0.08F);
                pending_k[base] =
                    bf16_pattern(kSeed + 700003U + token * 13U + head * 7U + dim, 0.08F);
            }
        }
        for (std::int32_t head = 0; head < kProfile.value_heads; ++head) {
            for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                pending_v[((column * kProfile.value_heads + head) * kStateDim) + dim] =
                    bf16_pattern(kSeed + 800003U + token * 17U + head * 11U + dim, 0.08F);
            }
            const std::size_t pair = (column * heads + head) * 2;
            pending_gate[pair] =
                -0.03F -
                static_cast<float>(mix(kSeed + 900003U + token * 7U + head) % 900U) / 1000.0F;
            pending_gate[pair + 1] =
                0.07F +
                static_cast<float>(mix(kSeed + 1100003U + token * 9U + head) % 900U) / 1000.0F;
        }
    }

    DeviceBuffer device_q = to_device(q_bits);
    DeviceBuffer device_k = to_device(k_bits);
    DeviceBuffer device_v = to_device(v_bits);
    DeviceBuffer device_g = to_device(g_bits);
    DeviceBuffer device_beta = to_device(beta_bits);
    DeviceBuffer device_pending_q = to_device(pending_q);
    DeviceBuffer device_pending_k = to_device(pending_k);
    DeviceBuffer device_pending_v = to_device(pending_v);
    DeviceBuffer device_pending_gate = to_device(pending_gate);
    DeviceBuffer device_out(ninfer::test::cuda_environment().context.device(),
                            value_elements * sizeof(std::uint16_t));
    device_out.fill(0);
    DeviceBuffer device_initial = to_device(std::vector<std::int32_t>{kRow0Slot, kRow1Slot});
    DeviceBuffer device_valid = to_device(std::vector<std::int32_t>{kWidth, kWidth});
    DeviceBuffer device_pending_rows =
        to_device(std::vector<std::int32_t>{kRow0Slot, kRow0Slot, kRow0Pending, 0, kRow1Slot,
                                            kRow1Slot, 0, 0});
    DeviceBuffer device_source = to_device(std::vector<std::int32_t>{kRow0Slot, kRow1Slot});
    DeviceBuffer device_destination =
        to_device(std::vector<std::int32_t>{kRow0Slot, kRow1Destination});
    DeviceBuffer device_commits =
        to_device(std::vector<std::int32_t>{kRow0Commit, kRow1Commit});

    Tensor q_tensor(device_q.span(), DType::BF16, {kStateDim, kQkHeads, kWidth, kRows});
    Tensor k_tensor(device_k.span(), DType::BF16, {kStateDim, kQkHeads, kWidth, kRows});
    Tensor v_tensor(device_v.span(), DType::BF16,
                    {kStateDim, kProfile.value_heads, kWidth, kRows});
    Tensor g_tensor(device_g.span(), DType::FP32, {kProfile.value_heads, kWidth, kRows});
    Tensor beta_tensor(device_beta.span(), DType::FP32, {kProfile.value_heads, kWidth, kRows});
    Tensor pending_q_tensor(device_pending_q.span(), DType::BF16,
                            {kStateDim, kQkHeads, kWidth, kRows});
    Tensor pending_k_tensor(device_pending_k.span(), DType::BF16,
                            {kStateDim, kQkHeads, kWidth, kRows});
    Tensor pending_v_tensor(device_pending_v.span(), DType::BF16,
                            {kStateDim, kProfile.value_heads, kWidth, kRows});
    Tensor pending_gate_tensor(device_pending_gate.span(), DType::FP32,
                               {2, kProfile.value_heads, kWidth, kRows});
    Tensor out_tensor(device_out.span(), DType::BF16,
                      {kStateDim, kProfile.value_heads, kWidth, kRows});
    Tensor initial_slots(device_initial.span(), DType::I32, {kRows});
    Tensor valid_columns(device_valid.span(), DType::I32, {kRows});
    Tensor pending_rows(device_pending_rows.span(), DType::I32, {4, kRows});
    Tensor source_slots(device_source.span(), DType::I32, {kRows});
    Tensor destination_slots(device_destination.span(), DType::I32, {kRows});
    Tensor commit_columns(device_commits.span(), DType::I32, {kRows});

    // Distinct initial states per row.
    const std::vector<float> row0_initial =
        initial_recurrent_values(recurrent_slot_elements, kSeed, 0, 0);
    const std::vector<float> row1_initial =
        initial_recurrent_values(recurrent_slot_elements, kSeed, 0, 1);
    const std::vector<std::uint16_t> initial_conv(conv_slot_elements, 0);
    for (std::int32_t layer = 0; layer < kProfile.layers; ++layer) {
        const Tensor slot0 =
            state_pool.recurrent_slot(static_cast<std::uint32_t>(layer), kRow0Slot);
        cuda_check(cudaMemcpy(slot0.data, row0_initial.data(), recurrent_slot_bytes,
                              cudaMemcpyHostToDevice),
                   "upload row0 initial state");
        const Tensor slot1 =
            state_pool.recurrent_slot(static_cast<std::uint32_t>(layer), kRow1Slot);
        cuda_check(cudaMemcpy(slot1.data, row1_initial.data(), recurrent_slot_bytes,
                              cudaMemcpyHostToDevice),
                   "upload row1 initial state");
        for (const std::int32_t slot : {kRow0Slot, kRow1Slot, kRow1Destination}) {
            const Tensor conv = state_pool.conv_slot(static_cast<std::uint32_t>(layer), slot);
            cuda_check(cudaMemcpy(conv.data, initial_conv.data(), conv_slot_bytes,
                                  cudaMemcpyHostToDevice),
                       "upload initial conv");
        }
    }

    // Single-stream reference: a per-token state-pool chain in a scratch layer buffer that
    // consumes only that row's own record columns, one token per batch update.
    DeviceBuffer local_state(ninfer::test::cuda_environment().context.device(),
                             recurrent_slot_bytes);
    DeviceBuffer reference_slot = to_device(std::vector<std::int32_t>{0});
    Tensor selected_reference_slot(reference_slot.span(), DType::I32, {1});
    Tensor local_state_tensor(local_state.span(), DType::FP32,
                              {kStateDim, kStateDim, kProfile.value_heads});
    DeviceBuffer token_out(ninfer::test::cuda_environment().context.device(),
                           token_values * sizeof(std::uint16_t));
    Tensor token_out_tensor(token_out.span(), DType::BF16, {kStateDim, kProfile.value_heads, 1});

    int failures = 0;
    const auto fail = [&](const std::string& what) {
        std::cerr << "desynced record/fold: " << what << "\n";
        ++failures;
    };
    const auto load_initial = [&](const std::vector<float>& initial) {
        cuda_check(cudaMemcpy(local_state.data(), initial.data(), recurrent_slot_bytes,
                              cudaMemcpyHostToDevice),
                   "reset reference chain state");
    };
    const auto local_bits = [&] {
        return from_device<std::uint32_t>(local_state.data(), recurrent_slot_elements);
    };
    const auto slot_bits = [&](std::int32_t layer, std::int32_t slot) {
        return from_device<std::uint32_t>(
            state_pool.recurrent_slot(static_cast<std::uint32_t>(layer), slot).data,
            recurrent_slot_elements);
    };
    const auto initial_bits = [&](const std::vector<float>& initial) {
        std::vector<std::uint32_t> bits(recurrent_slot_elements);
        for (std::size_t i = 0; i < recurrent_slot_elements; ++i)
            bits[i] = std::bit_cast<std::uint32_t>(initial[i]);
        return bits;
    };
    // One chain step: apply token `token` of `row` from the given planes with that token's
    // per-head gate words, in place in the scratch state; returns the chain's output bits.
    const auto chain_step = [&](std::int32_t row, std::int32_t token, const Tensor& q_plane,
                                const Tensor& key_plane, const Tensor& value_plane,
                                const float* g_words, const float* beta_words) {
        const std::size_t offset =
            (static_cast<std::size_t>(row) * kWidth + token) * heads;
        Tensor query = q_plane.slice(3, row, 1).slice(2, token, 1);
        Tensor key = key_plane.slice(3, row, 1).slice(2, token, 1);
        Tensor value = value_plane.slice(3, row, 1).slice(2, token, 1);
        DeviceBuffer step_g = to_device(std::vector<float>(g_words + offset,
                                                           g_words + offset + heads));
        DeviceBuffer step_beta = to_device(std::vector<float>(beta_words + offset,
                                                              beta_words + offset + heads));
        Tensor g_token(step_g.span(), DType::FP32, {kProfile.value_heads, 1});
        Tensor beta_token(step_beta.span(), DType::FP32, {kProfile.value_heads, 1});
        ops::gated_delta_net_batch_update(
            query, key, value, g_token, beta_token, kScale, true, local_state_tensor,
            selected_reference_slot, selected_reference_slot, token_out_tensor,
            ninfer::test::cuda_environment().default_execution);
        return from_device<std::uint16_t>(token_out.data(), token_values);
    };
    const auto out_at = [&](std::int32_t layer, std::int32_t row, std::int32_t token) {
        const auto* base = static_cast<const std::uint16_t*>(device_out.data());
        const std::size_t begin =
            (static_cast<std::size_t>(layer) * kRecordCapacity * kWidth +
             static_cast<std::size_t>(row) * kWidth + token) * token_values;
        return std::vector<std::uint16_t>(base + begin, base + begin + token_values);
    };
    const auto conv_word = [&](std::int32_t layer, std::int32_t row, std::int32_t token,
                               std::int32_t channel) {
        const std::size_t column =
            (static_cast<std::size_t>(layer) * kRecordCapacity + row) * kWidth + token;
        return conv_bits[column * kProfile.conv_channels + channel];
    };

    // The pending gate record interleaves {g, beta}; the chain consumes separate per-head words.
    std::vector<float> pending_g_words(gate_elements), pending_beta_words(gate_elements);
    for (std::size_t i = 0; i < gate_elements; ++i) {
        pending_g_words[i] = pending_gate[2 * i];
        pending_beta_words[i] = pending_gate[2 * i + 1];
    }

    // Round head: row0 folds its extra accepted token in place; row1 folds nothing; the record
    // then runs from the post-fold states. Per layer, checked against the reference chains.
    for (std::int32_t layer = 0; layer < kProfile.layers; ++layer) {
        Tensor recurrent_states =
            state_pool.layer_view(static_cast<std::uint32_t>(layer)).recurrent;
        GdnReplayRecordLayer layer_records = records.layer(layer, kRows);
        ops::gated_delta_net_fold_replay_record(
            q_tensor, k_tensor, v_tensor, g_tensor, beta_tensor, kScale, recurrent_states,
            valid_columns, initial_slots,
            {layer_records.key, layer_records.value, layer_records.gate}, out_tensor,
            {pending_rows, pending_k_tensor, pending_v_tensor, pending_gate_tensor},
            ninfer::test::cuda_environment().default_execution.stream());
        cuda_synchronize();

        // Row0: its pending token folds in place, exactly as the single-stream chain does.
        load_initial(row0_initial);
        chain_step(0, 0, pending_q_tensor, pending_k_tensor, pending_v_tensor,
                   pending_g_words.data(), pending_beta_words.data());
        if (slot_bits(layer, kRow0Slot) != local_bits())
            fail("row0 pending fold state != single-stream chain at layer " +
                 std::to_string(layer));

        // Row1: the round head leaves its state untouched.
        if (slot_bits(layer, kRow1Slot) != initial_bits(row1_initial))
            fail("row1 state was modified by the round head at layer " + std::to_string(layer));

        // The round's records are bit-preserving copies of the inputs.
        const auto key_after = from_device<std::uint16_t>(
            layer_records.key.data, static_cast<std::size_t>(kStateDim) * kQkHeads * columns);
        const auto value_after = from_device<std::uint16_t>(
            layer_records.value.data, static_cast<std::size_t>(kStateDim) * heads * columns);
        const auto gate_after =
            from_device<std::uint32_t>(layer_records.gate.data, heads * columns * 2);
        const auto* k_base = static_cast<const std::uint16_t*>(device_k.data());
        const auto* v_base = static_cast<const std::uint16_t*>(device_v.data());
        for (std::size_t i = 0; i < qk_elements; ++i)
            if (key_after[i] != k_base[i]) { fail("key record is not bit-preserving"); break; }
        for (std::size_t i = 0; i < value_elements; ++i)
            if (value_after[i] != v_base[i]) { fail("value record is not bit-preserving"); break; }
        for (std::size_t i = 0; i < gate_elements; ++i)
            if (gate_after[2 * i] != std::bit_cast<std::uint32_t>(g_bits[i]) ||
                gate_after[2 * i + 1] != std::bit_cast<std::uint32_t>(beta_bits[i])) {
                fail("gate record is not bit-preserving");
                break;
            }

        // Row0's token outputs continue its own chain from the folded state; row1's start at
        // its untouched state.
        load_initial(row0_initial);
        chain_step(0, 0, pending_q_tensor, pending_k_tensor, pending_v_tensor,
                   pending_g_words.data(), pending_beta_words.data());
        for (std::int32_t token = 0; token < kWidth; ++token) {
            const auto reference = chain_step(0, token, q_tensor, k_tensor, v_tensor,
                                              g_bits.data(), beta_bits.data());
            if (reference != out_at(layer, 0, token))
                fail("row0 token output != single-stream chain at layer " +
                     std::to_string(layer) + " token " + std::to_string(token));
        }
        load_initial(row1_initial);
        for (std::int32_t token = 0; token < kWidth; ++token) {
            const auto reference = chain_step(1, token, q_tensor, k_tensor, v_tensor,
                                              g_bits.data(), beta_bits.data());
            if (reference != out_at(layer, 1, token))
                fail("row1 token output != single-stream chain at layer " +
                     std::to_string(layer) + " token " + std::to_string(token));
        }
    }

    // Round commit: desynced accepted prefixes folded per row (row0 in place, row1 to a fresh
    // destination), then compared against each row's own continued chain.
    const ops::GdnReplayFoldPlan fold_plan(records, state_pool.all_layers_view());
    const std::array fold_rows{ops::GdnReplayFoldRow{kRow0Slot, kRow0Slot, kRow0Commit},
                               ops::GdnReplayFoldRow{kRow1Slot, kRow1Destination, kRow1Commit}};
    fold_plan.execute(fold_rows, ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    for (std::int32_t layer = 0; layer < kProfile.layers; ++layer) {
        load_initial(row0_initial);
        chain_step(0, 0, pending_q_tensor, pending_k_tensor, pending_v_tensor,
                   pending_g_words.data(), pending_beta_words.data());
        for (std::int32_t token = 0; token < kRow0Commit; ++token)
            chain_step(0, token, q_tensor, k_tensor, v_tensor, g_bits.data(), beta_bits.data());
        if (slot_bits(layer, kRow0Slot) != local_bits())
            fail("row0 committed state != single-stream chain at layer " +
                 std::to_string(layer));

        load_initial(row1_initial);
        for (std::int32_t token = 0; token < kRow1Commit; ++token)
            chain_step(1, token, q_tensor, k_tensor, v_tensor, g_bits.data(), beta_bits.data());
        if (slot_bits(layer, kRow1Destination) != local_bits())
            fail("row1 committed state != single-stream chain at layer " +
                 std::to_string(layer));

        // The rows carried different sequences; their committed states must differ.
        if (slot_bits(layer, kRow0Slot) == slot_bits(layer, kRow1Destination))
            fail("desynced rows collapsed to one state at layer " + std::to_string(layer));

        // Convolution history: tail_3(zeros || the row's committed record columns).
        std::vector<std::uint16_t> expected_row0(conv_slot_elements);
        std::vector<std::uint16_t> expected_row1(conv_slot_elements);
        for (std::int32_t channel = 0; channel < kProfile.conv_channels; ++channel) {
            expected_row0[channel] = conv_word(layer, 0, kRow0Commit - 3, channel);
            expected_row0[static_cast<std::size_t>(kProfile.conv_channels) + channel] =
                conv_word(layer, 0, kRow0Commit - 2, channel);
            expected_row0[2LL * kProfile.conv_channels + channel] =
                conv_word(layer, 0, kRow0Commit - 1, channel);
            expected_row1[channel] = conv_word(layer, 1, kRow1Commit - 3, channel);
            expected_row1[static_cast<std::size_t>(kProfile.conv_channels) + channel] =
                conv_word(layer, 1, kRow1Commit - 2, channel);
            expected_row1[2LL * kProfile.conv_channels + channel] =
                conv_word(layer, 1, kRow1Commit - 1, channel);
        }
        const auto conv0 = from_device<std::uint16_t>(
            state_pool.conv_slot(static_cast<std::uint32_t>(layer), kRow0Slot).data,
            conv_slot_elements);
        const auto conv3 = from_device<std::uint16_t>(
            state_pool.conv_slot(static_cast<std::uint32_t>(layer), kRow1Destination).data,
            conv_slot_elements);
        if (conv0 != expected_row0) fail("row0 conv history != tail_3 at layer " +
                                         std::to_string(layer));
        if (conv3 != expected_row1) fail("row1 conv history != tail_3 at layer " +
                                         std::to_string(layer));
    }
    return failures;
}

// The desynced round's extra state contribution: a row that steps one token through the
// Flash-Next direct path while its siblings commit through the record/fold path. Both paths
// must land on the same fp32 state bits; a reduction-order difference between them is the
// Df2Positions one-ulp row-dependent drift.
int run_desynced_direct_path() {
    constexpr std::int32_t kStateSlots = 4;
    constexpr std::int32_t kRow0Slot   = 1;
    constexpr std::uint32_t kSeed      = 6200U;
    constexpr float kScale             = 1.0F / std::sqrt(128.0F);
    constexpr std::int32_t kQkHeads    = 16;
    constexpr std::int32_t kHeads      = 48;
    const std::size_t slot_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * kHeads;
    const std::size_t slot_bytes = slot_elements * sizeof(float);
    const std::size_t packed_column = (2 * kQkHeads + kHeads) * kStateDim;

    const auto& environment = ninfer::test::cuda_environment();
    DeviceBuffer packed(environment.context.device(),
                        packed_column * sizeof(std::uint16_t));
    DeviceBuffer states(environment.context.device(), slot_bytes * kStateSlots);
    states.fill(0);
    DeviceBuffer scratch(environment.context.device(), slot_bytes);
    DeviceBuffer g_buffer(environment.context.device(), kHeads * sizeof(float));
    DeviceBuffer beta_buffer(environment.context.device(), kHeads * sizeof(float));
    DeviceBuffer out_buffer(environment.context.device(),
                            static_cast<std::size_t>(kStateDim) * kHeads *
                                sizeof(std::uint16_t));
    DeviceBuffer source_slot = to_device(std::vector<std::int32_t>{kRow0Slot});
    DeviceBuffer destination_slot = to_device(std::vector<std::int32_t>{kRow0Slot});
    DeviceBuffer reference_slot = to_device(std::vector<std::int32_t>{0});

    // One token of represented inputs, shared by both paths.
    std::vector<std::uint16_t> q_bits(static_cast<std::size_t>(kStateDim) * kQkHeads);
    std::vector<std::uint16_t> k_bits(q_bits.size());
    std::vector<std::uint16_t> v_bits(static_cast<std::size_t>(kStateDim) * kHeads);
    std::vector<float> g_bits(kHeads), beta_bits(kHeads);
    for (std::int32_t head = 0; head < kQkHeads; ++head) {
        for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
            q_bits[static_cast<std::size_t>(head) * kStateDim + dim] =
                bf16_pattern(kSeed + head * 5U + dim, 0.08F);
            k_bits[static_cast<std::size_t>(head) * kStateDim + dim] =
                bf16_pattern(kSeed + 100003U + head * 7U + dim, 0.08F);
        }
    }
    for (std::int32_t head = 0; head < kHeads; ++head) {
        for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
            v_bits[static_cast<std::size_t>(head) * kStateDim + dim] =
                bf16_pattern(kSeed + 200003U + head * 11U + dim, 0.08F);
        }
        g_bits[head] =
            -0.04F - static_cast<float>(mix(kSeed + head) % 900U) / 1000.0F;
        beta_bits[head] = 0.06F + static_cast<float>(mix(kSeed + 300003U + head) % 900U) / 1000.0F;
    }
    // The packed column interleaves q, k, and v head-major.
    std::vector<std::uint16_t> packed_bits(packed_column);
    for (std::int32_t head = 0; head < kQkHeads; ++head)
        for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
            packed_bits[static_cast<std::size_t>(head) * kStateDim + dim] =
                q_bits[static_cast<std::size_t>(head) * kStateDim + dim];
            packed_bits[static_cast<std::size_t>(kQkHeads + head) * kStateDim + dim] =
                k_bits[static_cast<std::size_t>(head) * kStateDim + dim];
        }
    for (std::int32_t head = 0; head < kHeads; ++head)
        for (std::int32_t dim = 0; dim < kStateDim; ++dim)
            packed_bits[static_cast<std::size_t>(2 * kQkHeads + head) * kStateDim + dim] =
                v_bits[static_cast<std::size_t>(head) * kStateDim + dim];
    cuda_check(cudaMemcpy(packed.data(), packed_bits.data(), packed_bits.size() * 2,
                          cudaMemcpyHostToDevice),
               "upload packed column");
    cuda_check(cudaMemcpy(g_buffer.data(), g_bits.data(), kHeads * sizeof(float),
                          cudaMemcpyHostToDevice),
               "upload gates");
    cuda_check(cudaMemcpy(beta_buffer.data(), beta_bits.data(), kHeads * sizeof(float),
                          cudaMemcpyHostToDevice),
               "upload betas");

    const std::vector<float> initial = [&] {
        std::vector<float> values(slot_elements);
        for (std::size_t i = 0; i < slot_elements; ++i)
            values[i] = signed_pattern(kSeed + 400009U + static_cast<std::uint32_t>(i), 0.05F);
        return values;
    }();
    cuda_check(cudaMemcpy(states.data(), initial.data(), slot_bytes, cudaMemcpyHostToDevice),
               "upload initial state");
    cuda_check(cudaMemcpy(scratch.data(), initial.data(), slot_bytes, cudaMemcpyHostToDevice),
               "upload reference state");

    Tensor packed_tensor(packed.data(), environment.context.device(), DType::BF16,
                         {static_cast<std::int32_t>(packed_column), 1});
    Tensor state_tensor(states.data(), environment.context.device(), DType::FP32,
                        {kStateDim, kStateDim, kHeads, kStateSlots});
    state_tensor.nb[3] = static_cast<std::int64_t>(slot_bytes);
    Tensor scratch_tensor(scratch.data(), environment.context.device(), DType::FP32,
                          {kStateDim, kStateDim, kHeads});
    Tensor g_tensor(g_buffer.data(), environment.context.device(), DType::FP32, {kHeads, 1});
    Tensor beta_tensor(beta_buffer.data(), environment.context.device(), DType::FP32,
                       {kHeads, 1});
    Tensor out_tensor(out_buffer.data(), environment.context.device(), DType::BF16,
                      {kStateDim, kHeads, 1});
    Tensor source_tensor(source_slot.span(), DType::I32, {1});
    Tensor destination_tensor(destination_slot.span(), DType::I32, {1});
    Tensor reference_tensor(reference_slot.span(), DType::I32, {1});

    // The direct packed path, in place in slot 1.
    ninfer::ops::detail::gated_delta_net::launch_recurrent_batch_update_packed_qkv(
        packed_tensor, kQkHeads, kHeads, g_tensor, beta_tensor, kScale, state_tensor,
        source_tensor, destination_tensor, out_tensor,
        environment.default_execution.stream());
    cuda_check(cudaDeviceSynchronize(), "packed direct path");

    // The single-stream reference: the same token through the state-pool batch update.
    DeviceBuffer device_q = to_device(q_bits);
    DeviceBuffer device_k = to_device(k_bits);
    DeviceBuffer device_v = to_device(v_bits);
    Tensor q_tensor(device_q.data(), environment.context.device(), DType::BF16,
                    {kStateDim, kQkHeads, 1, 1});
    Tensor k_tensor(device_k.data(), environment.context.device(), DType::BF16,
                    {kStateDim, kQkHeads, 1, 1});
    Tensor v_tensor(device_v.data(), environment.context.device(), DType::BF16,
                    {kStateDim, kHeads, 1, 1});
    Tensor g_step(g_buffer.data(), environment.context.device(), DType::FP32, {kHeads, 1});
    Tensor beta_step(beta_buffer.data(), environment.context.device(), DType::FP32, {kHeads, 1});
    Tensor token_out(out_buffer.data(), environment.context.device(), DType::BF16,
                     {kStateDim, kHeads, 1});
    ops::gated_delta_net_batch_update(
        q_tensor, k_tensor, v_tensor, g_step, beta_step, kScale, true, scratch_tensor,
        reference_tensor, reference_tensor, token_out, environment.default_execution);
    cuda_check(cudaDeviceSynchronize(), "reference batch update");

    const auto* direct_ptr =
        static_cast<const char*>(states.data()) +
        static_cast<std::ptrdiff_t>(kRow0Slot) * static_cast<std::ptrdiff_t>(slot_bytes);
    const auto direct_bits = from_device<std::uint32_t>(direct_ptr, slot_elements);
    const auto reference_bits = from_device<std::uint32_t>(scratch.data(), slot_elements);
    if (direct_bits != reference_bits) {
        std::cerr << "desynced direct path: state bits differ from the single-stream chain "
                     "(reduction-order drift)\n";
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    for (int prefix : {3, 11})
        suite.add("gdn_fold/device/l36/h24/w16/a" + std::to_string(prefix), [=] {
            return run_case({36, 24, 5120}, 16, 1, {prefix}, 5101U, true, true, true);
        });
    for (const FoldProfile profile :
         {FoldProfile{48, 48, 10240}, {30, 32, 8192}, {36, 48, 10240}, {36, 24, 5120}})
        for (bool captured : {false, true}) {
            const auto id = "gdn_fold/device/l" + std::to_string(profile.layers) + "/h" +
                            std::to_string(profile.value_heads) + (captured ? "/graph" : "/eager");
            suite.add(id, [=] {
                return run_case(profile, 6, 4, {0, 1, 3, 6}, 5100U, captured, captured, true);
            });
        }
    const auto add = [&](FoldProfile p, int width, int batch, std::vector<int> commits,
                         unsigned seed, bool distinct = false, bool captured = false,
                         int capacity = kRecordCapacity) {
        std::string id = std::string(p.value_heads == 24 ? "gdn_fold_local/l" : "gdn_fold/l") +
                         std::to_string(p.layers) + "/h" + std::to_string(p.value_heads) + "/w" +
                         std::to_string(width) + "/b" + std::to_string(batch) +
                         (distinct ? "/disjoint" : "/alias") + "/commits";
        for (int c : commits) id += "_" + std::to_string(c);
        if (captured) id += "/graph";
        if (capacity != kRecordCapacity) id += "/capacity_" + std::to_string(capacity);
        suite.add(id,
                  [=] {
                      return run_case(p, width, batch, commits, seed, distinct, captured, false,
                                      capacity);
                  },
                  {{"execution", captured || (width == 16 && batch == 8) ? "graph" : "eager"}});
    };
    add({48, 48, 10240}, 2, 1, {2}, 1801U, true);
    add({48, 48, 10240}, 3, 4, {0, 1, 2, 3}, 1811U);
    add({48, 48, 10240}, 6, 8, {0, 1, 2, 3, 6, 4, 1, 5}, 1821U);
    add({48, 48, 10240}, 7, 8, {7, 0, 1, 2, 3, 4, 5, 6}, 1823U, true);
    add({48, 48, 10240}, 16, 8, {0, 1, 2, 3, 7, 13, 15, 16}, 1825U, true);
    add({48, 48, 10240}, 16, 1, {16}, 1827U, true);
    add({48, 48, 10240}, 16, 1, {0}, 1829U, true);
    add({30, 32, 8192}, 2, 1, {2}, 1831U);
    add({30, 32, 8192}, 6, 1, {6}, 1841U);
    add({30, 32, 8192}, 6, 2, {2, 5}, 1851U);
    add({30, 32, 8192}, 16, 8, {0, 1, 2, 3, 16, 7, 12, 5}, 1861U);
    add({36, 48, 10240}, 2, 1, {2}, 1871U, true);
    add({36, 48, 10240}, 5, 4, {0, 1, 3, 5}, 1873U);
    add({36, 48, 10240}, 16, 8, {0, 1, 2, 3, 4, 5, 6, 7}, 1875U, true);
    add({36, 48, 10240}, 16, 8, {8, 9, 10, 11, 12, 13, 14, 15}, 1877U);
    add({36, 48, 10240}, 16, 1, {16}, 1879U, true);
    add({36, 48, 10240}, 16, 1, {0}, 1881U, true);
    add({36, 48, 10240}, 16, 8, {0, 0, 0, 0, 0, 0, 0, 0}, 1883U, true);
    add({36, 48, 10240}, 16, 8, {0, 0, 0, 0, 0, 0, 0, 1}, 1885U, true);
    add({36, 48, 10240}, 16, 8, {0, 0, 0, 0, 0, 0, 0, 16}, 1887U, true);
    add({30, 32, 8192}, 2, 3, {0, 1, 2}, 1889U);
    add({30, 32, 8192}, 2, 1, {1}, 1891U, false, false, 1);
    add({36, 48, 10240}, 5, 3, {0, 1, 5}, 1893U, false, false, 3);
    for (const FoldProfile profile : {FoldProfile{36, 48, 10240}, {36, 24, 5120}})
        suite.add("gdn_fold/device/l36/h" + std::to_string(profile.value_heads) + "/sparse/graph",
                  [=] { return run_case(profile, 16, 4, {0, 0, 0, 16}, 5200U, true, true, true); });
    suite.add("gdn_fold/device/l36/h48/zero/graph", [] {
        return run_case({36, 48, 10240}, 4, 4, {0, 0, 0, 0}, 5201U, true, true, true);
    });
    suite.add("gdn_fold/record_rounds/q4_q5", [] { return run_record_fold_rounds(); },
              {{"format", "q4_q5"}});
    suite.add("gdn_fold/record_rounds/q8", [] { return run_record_fold_rounds(true); },
              {{"format", "q8"}});
    suite.add("gdn_fold/desynced_two_row/l36/h24", [] { return run_desynced_rounds(); });
    suite.add("gdn_fold/desynced_direct_path/l36/h24",
              [] { return run_desynced_direct_path(); });

    for (int b = 1; b <= 8; ++b) {
        for (int t : {2, 4, 6, 8, 16}) {
            add({36, 24, 5120}, t, b, std::vector<int>(b, t), 2200U + 100 * b + t, true);
            std::vector<int> mixed(b);
            for (int row = 0; row < b; ++row) mixed[row] = (3 * row + 1) % (t + 1);
            add({36, 24, 5120}, t, b, mixed, 3200U + 100 * b + t);
        }
        add({36, 24, 5120}, 16, b, std::vector<int>(b, 0), 4200U + b, true);
        add({36, 24, 5120}, 16, b, std::vector<int>(b, 1), 4300U + b, true);
    }
    for (int first = 0; first < 16; first += 8) {
        std::vector<int> commits(8);
        for (int b = 0; b < 8; ++b) commits[b] = first + b;
        add({36, 24, 5120}, 16, 8, commits, 4400U + first, true);
    }
    for (bool distinct : {false, true}) {
        for (int commit : {8, 9, 10})
            add({36, 24, 5120}, 16, 1, {commit}, 4500U + commit, distinct, commit == 9);
        for (int width : {9, 10}) add({36, 24, 5120}, width, 1, {width}, 4600U + width, distinct);
    }
#if NINFER_TARGET_SPARK
    suite.add("gdn_fold_local/record_rounds/q8", [] { return run_record_fold_rounds(true, true); },
              {{"format", "q8"}});
#endif
    return suite.main(argc, argv);
}
