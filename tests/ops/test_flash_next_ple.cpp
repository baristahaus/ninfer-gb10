#include "ninfer/ops/flash_next_ple.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr int kStreams = 4;
constexpr int kHidden = 2560;
constexpr int kHyper = kStreams * kHidden;
constexpr int kTokens = 2;
constexpr int kState = 9;

std::vector<std::uint16_t> encode(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

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

double fp8_e4m3(std::uint8_t bits) {
    const int sign = bits >> 7;
    const int exponent = (bits >> 3) & 15;
    const int fraction = bits & 7;
    double value = 0.0;
    if (exponent == 0) {
        value = std::ldexp(static_cast<double>(fraction), -9);
    } else if (exponent == 15 && fraction == 7) {
        return std::numeric_limits<double>::quiet_NaN();
    } else {
        value = std::ldexp(1.0 + static_cast<double>(fraction) / 8.0, exponent - 7);
    }
    return sign ? -value : value;
}

double represented_bf16(double value) {
    return bf16_to_f32(f32_to_bf16(static_cast<float>(value)));
}

// fork: the history continues from a checkpoint slot into a distinct slot whose prior content
// is unrelated; the checkpoint must stay unchanged and the stale content must never be read.
int run(bool fork) {
    std::vector<float> hyper(kHyper * kTokens), key_norm(kHyper), query_norm(kHyper), conv_norm(kHyper), state(kHyper * kState);
    fill_uniform(hyper, 1701, -0.5F, 0.5F);
    fill_uniform(key_norm, 1702, -0.08F, 0.08F);
    fill_uniform(query_norm, 1703, -0.08F, 0.08F);
    fill_uniform(conv_norm, 1704, -0.08F, 0.08F);
    fill_uniform(state, 1705, -0.1F, 0.1F);
    round_to_bf16(hyper);
    round_to_bf16(key_norm);
    round_to_bf16(query_norm);
    round_to_bf16(conv_norm);
    round_to_bf16(state);

    const std::uint8_t fp8_values[] = {0x38, 0xb8, 0x30, 0xb0, 0x40, 0xc0};
    std::vector<std::uint8_t> gathered(kHidden * kTokens);
    for (std::size_t i = 0; i < gathered.size(); ++i) { gathered[i] = fp8_values[i % 6]; }
    constexpr float embedding_scale = 0.25F;
    std::vector<double> embedding(gathered.size());
    for (std::size_t i = 0; i < gathered.size(); ++i) { embedding[i] = fp8_e4m3(gathered[i]) * embedding_scale; }

    std::vector<float> key_weight(static_cast<std::size_t>(kHyper) * kHidden, 0.0F);
    std::vector<float> value_weight(static_cast<std::size_t>(kHidden) * kHidden, 0.0F);
    for (int row = 0; row < kHyper; ++row) { key_weight[static_cast<std::size_t>(row) * kHidden + (17 * row) % kHidden] = (row & 1) ? -0.125F : 0.125F; }
    for (int row = 0; row < kHidden; ++row) { value_weight[static_cast<std::size_t>(row) * kHidden + row] = 0.25F; }
    std::vector<float> conv_weight(kHyper * 4);
    for (int channel = 0; channel < kHyper; ++channel) {
        conv_weight[channel] = 0.125F;
        conv_weight[kHyper + channel] = -0.0625F;
        conv_weight[2 * kHyper + channel] = 0.03125F;
        conv_weight[3 * kHyper + channel] = 0.25F;
    }

    std::vector<double> key(kHyper * kTokens), value(kHidden * kTokens), gated(kHyper * kTokens), normalized(kHyper * kTokens), expected(kHyper * kTokens);
    std::vector<double> expected_state(state.begin(), state.end());
    for (int token = 0; token < kTokens; ++token) {
        for (int row = 0; row < kHyper; ++row) {
            double sum = 0.0;
            for (int column = 0; column < kHidden; ++column) { sum += key_weight[static_cast<std::size_t>(row) * kHidden + column] * embedding[token * kHidden + column]; }
            key[token * kHyper + row] = represented_bf16(sum);
        }
        for (int row = 0; row < kHidden; ++row) {
            double sum = 0.0;
            for (int column = 0; column < kHidden; ++column) { sum += value_weight[static_cast<std::size_t>(row) * kHidden + column] * embedding[token * kHidden + column]; }
            value[token * kHidden + row] = represented_bf16(sum);
        }
        for (int stream = 0; stream < kStreams; ++stream) {
            const int base = token * kHyper + stream * kHidden;
            double q_square = 0.0, k_square = 0.0;
            for (int d = 0; d < kHidden; ++d) {
                q_square += double(hyper[base + d]) * hyper[base + d];
                k_square += key[base + d] * key[base + d];
            }
            const double qi = 1.0 / std::sqrt(q_square / kHidden + 1.0e-6);
            const double ki = 1.0 / std::sqrt(k_square / kHidden + 1.0e-6);
            double dot = 0.0;
            for (int d = 0; d < kHidden; ++d) { dot += hyper[base + d] * qi * (1.0 + query_norm[stream * kHidden + d]) * key[base + d] * ki * (1.0 + key_norm[stream * kHidden + d]); }
            dot /= std::sqrt(static_cast<double>(kHidden));
            const double transformed = std::copysign(std::sqrt(std::max(std::abs(dot), 1.0e-6)), dot);
            const double gate = 1.0 / (1.0 + std::exp(-transformed));
            double square = 0.0;
            for (int d = 0; d < kHidden; ++d) {
                gated[base + d] =
                    represented_bf16(gate * value[token * kHidden + d]);
                square += gated[base + d] * gated[base + d];
            }
            const double inverse = 1.0 / std::sqrt(square / kHidden + 1.0e-6);
            for (int d = 0; d < kHidden; ++d) {
                normalized[base + d] = represented_bf16(
                    gated[base + d] * inverse *
                    (1.0 + conv_norm[stream * kHidden + d]));
            }
        }
        for (int channel = 0; channel < kHyper; ++channel) {
            double conv = conv_weight[channel] * expected_state[channel];
            conv += conv_weight[kHyper + channel] * expected_state[3 * kHyper + channel];
            conv += conv_weight[2 * kHyper + channel] * expected_state[6 * kHyper + channel];
            conv += conv_weight[3 * kHyper + channel] * normalized[token * kHyper + channel];
            const double represented_conv = represented_bf16(conv);
            const double silu = represented_bf16(
                represented_conv / (1.0 + std::exp(-represented_conv)));
            expected[token * kHyper + channel] = represented_bf16(
                gated[token * kHyper + channel] + silu);
        }
        for (int i = 0; i < kState - 1; ++i) {
            for (int channel = 0; channel < kHyper; ++channel) { expected_state[i * kHyper + channel] = expected_state[(i + 1) * kHyper + channel]; }
        }
        for (int channel = 0; channel < kHyper; ++channel) { expected_state[(kState - 1) * kHyper + channel] = normalized[token * kHyper + channel]; }
    }

    DeviceBuffer d_hyper = to_device(encode(hyper));
    DeviceBuffer d_gathered = to_device(gathered);
    DeviceBuffer d_key_weight = to_device(encode(key_weight));
    DeviceBuffer d_value_weight = to_device(encode(value_weight));
    DeviceBuffer d_key_norm = to_device(encode(key_norm));
    DeviceBuffer d_query_norm = to_device(encode(query_norm));
    DeviceBuffer d_conv_norm = to_device(encode(conv_norm));
    DeviceBuffer d_conv_weight = to_device(encode(conv_weight));
    DeviceBuffer d_scale = to_device(encode(std::vector<float>{embedding_scale}));
    const std::vector<std::uint16_t> state_bits = encode(state);
    std::vector<float> stale(state.size());
    fill_uniform(stale, 1723, -4.0F, 4.0F);
    DeviceBuffer d_state             = to_device(state_bits);
    DeviceBuffer d_destination_state = to_device(fork ? encode(stale) : state_bits);
    GuardedDeviceBuffer d_output(expected.size() * sizeof(std::uint16_t));
    Tensor hyper_tensor(d_hyper.p, DType::BF16, {kHyper, kTokens});
    Tensor gathered_tensor(d_gathered.p, DType::FP8_E4M3FN, {kHidden, kTokens});
    Tensor state_tensor(d_state.p, DType::BF16, {kHyper, kState});
    Tensor destination_state_tensor(fork ? d_destination_state.p : d_state.p, DType::BF16,
                                    {kHyper, kState});
    Tensor output_tensor(d_output.data(), DType::BF16, {kHyper, kTokens});
    ops::FlashNextPleWeights weights{
        .key_projection = bf16_weight(d_key_weight, kHyper, kHidden),
        .value_projection = bf16_weight(d_value_weight, kHidden, kHidden),
        .key_norm = Tensor(d_key_norm.p, DType::BF16, {kHyper}),
        .query_norm = Tensor(d_query_norm.p, DType::BF16, {kHyper}),
        .convolution_norm = Tensor(d_conv_norm.p, DType::BF16, {kHyper}),
        .convolution = Tensor(d_conv_weight.p, DType::BF16, {kHyper, 4}),
        .embedding_scale = Tensor(d_scale.p, DType::BF16, {1}),
    };
    WorkspaceArena workspace(ops::flash_next_ple_workspace_capacity_bytes(kTokens));
    ops::flash_next_ple(hyper_tensor, gathered_tensor, weights, state_tensor,
                        destination_state_tensor, output_tensor, workspace, nullptr);
    cuda_synchronize();
    const char* output_label = fork ? "Flash-Next PLE fork output" : "Flash-Next PLE output";
    const char* state_label  = fork ? "Flash-Next PLE fork state" : "Flash-Next PLE state";
    int failures =
        verify_pointwise(output_label, from_device_bf16(d_output.data(), expected.size()), expected,
                         {/*absolute*/ 1.0e-2, /*relative*/ 2.0e-2});
    failures += verify_pointwise(
        state_label, from_device_bf16(fork ? d_destination_state : d_state, expected_state.size()),
        expected_state, {/*absolute*/ 1.0e-2, /*relative*/ 2.0e-2});
    if (fork && from_device<std::uint16_t>(d_state, state_bits.size()) != state_bits) {
        std::cerr << "Flash-Next PLE fork modified its source state\n";
        ++failures;
    }
    failures += d_output.verify_guards(output_label);
    return failures;
}

// The replay fold shifts each committed record column into the row's state history: an exact
// data movement, checked bitwise for host row descriptors and for device descriptors read when a
// captured graph runs (one zero-column replay that must not write, then the committed rows).
int run_replay_fold() {
    constexpr int kWidth = 4;
    constexpr int kRows  = 2;
    constexpr int kSlots = 4;

    struct Row {
        int source, destination, columns;
    };

    constexpr Row rows[kRows]{{0, 0, 3}, {2, 3, 1}};
    std::vector<float> states(static_cast<std::size_t>(kHyper) * kState * kSlots);
    std::vector<float> records(static_cast<std::size_t>(kHyper) * kWidth * kRows);
    fill_uniform(states, 1711, -1.0F, 1.0F);
    fill_uniform(records, 1712, -1.0F, 1.0F);
    const std::vector<std::uint16_t> state_bits  = encode(states);
    const std::vector<std::uint16_t> record_bits = encode(records);

    std::vector<std::uint16_t> expected = state_bits;
    for (int row = 0; row < kRows; ++row) {
        for (int channel = 0; channel < kHyper; ++channel) {
            std::uint16_t history[kState];
            for (int i = 0; i < kState; ++i) {
                history[i] =
                    state_bits[static_cast<std::size_t>(rows[row].source) * kHyper * kState +
                               static_cast<std::size_t>(i) * kHyper + channel];
            }
            for (int column = 0; column < rows[row].columns; ++column) {
                for (int i = 0; i < kState - 1; ++i) { history[i] = history[i + 1]; }
                history[kState - 1] = record_bits[channel + static_cast<std::size_t>(kHyper) *
                                                                (row * kWidth + column)];
            }
            for (int i = 0; i < kState; ++i) {
                expected[static_cast<std::size_t>(rows[row].destination) * kHyper * kState +
                         static_cast<std::size_t>(i) * kHyper + channel] = history[i];
            }
        }
    }

    DeviceBuffer d_records = to_device(record_bits);
    Tensor records_tensor(d_records.p, DType::BF16, {kHyper, kWidth, kRows});
    int failures     = 0;
    const auto check = [&](const DeviceBuffer& d_states, const char* label) {
        const std::vector<std::uint16_t> actual =
            from_device<std::uint16_t>(d_states, state_bits.size());
        if (actual != expected) {
            std::cerr << "Flash-Next PLE replay fold (" << label << ") differs from its oracle\n";
            ++failures;
        }
    };

    DeviceBuffer d_host_states = to_device(state_bits);
    Tensor host_states(d_host_states.p, DType::BF16, {kHyper, kState, kSlots});
    const std::vector<ops::FlashNextPleFoldRow> host_rows{
        {rows[0].source, rows[0].destination, rows[0].columns},
        {rows[1].source, rows[1].destination, rows[1].columns}};
    ops::flash_next_ple_replay_fold(records_tensor, host_states, host_rows, nullptr);
    cuda_synchronize();
    check(d_host_states, "host rows");

    DeviceBuffer d_device_states = to_device(state_bits);
    Tensor device_states(d_device_states.p, DType::BF16, {kHyper, kState, kSlots});
    std::vector<std::int32_t> words(4 * kRows, 0);
    DeviceBuffer d_words = to_device(words);
    Tensor words_tensor(d_words.p, DType::I32, {4, kRows});
    DeviceContext context;
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    definition.capture(context.stream, [&] {
        ops::flash_next_ple_replay_fold_device(records_tensor, device_states, words_tensor, kRows,
                                               context.stream);
    });
    graph.instantiate(definition);
    graph.launch(context.stream);
    context.synchronize();
    if (from_device<std::uint16_t>(d_device_states, state_bits.size()) != state_bits) {
        std::cerr << "Flash-Next PLE replay fold wrote state for zero-column device rows\n";
        ++failures;
    }
    for (int row = 0; row < kRows; ++row) {
        words[4 * row]     = rows[row].source;
        words[4 * row + 1] = rows[row].destination;
        words[4 * row + 2] = rows[row].columns;
    }
    cuda_check(cudaMemcpy(d_words.p, words.data(), words.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice),
               "write device fold rows");
    graph.launch(context.stream);
    context.synchronize();
    check(d_device_states, "device rows");
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) { return 77; }
    try {
        const int failures = run(false) + run(true) + run_replay_fold();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Flash-Next PLE\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Flash-Next PLE: " << error.what() << '\n';
        return 1;
    }
}
