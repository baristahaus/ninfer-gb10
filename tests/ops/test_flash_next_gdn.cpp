#include "ninfer/ops/flash_next_gdn.h"
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

Weight bf16_weight(const DeviceBuffer& storage, int rows, int columns) {
    Weight out{};
    out.payload = out.qdata = storage.p;
    out.payload_bytes       = storage.bytes;
    out.qtype               = QType::BF16;
    out.layout              = QuantLayout::Contiguous;
    out.n = out.shape[0] = out.padded_shape[0] = rows;
    out.k = out.shape[1] = out.padded_shape[1] = columns;
    out.ndim                                   = 2;
    return out;
}

void store_bf16(DeviceBuffer& storage, std::size_t element, float value) {
    const std::uint16_t bits = f32_to_bf16(value);
    storage.copy_from_host(&bits, sizeof(bits), element * sizeof(bits));
}

// Independent FP64 oracle for the selected-slot transition with nonzero persistent state.
// The sparse represented weights above make the full projections explicit: q0=k0=v0=z0=x0,
// all other projection entries vanish. This exercises the complete 48-head update, including
// the three value heads that share key head zero. Persistent state is represented FP32.
int check_batch_state(const ops::FlashNextGdnWeights& weights) {
    constexpr int hidden = 2560, conv = 10240, qk = 2048, dim = 128, heads = 48;
    constexpr int batch = 2, slots = 4, state_size = dim * dim * heads;
    std::vector<float> expected_state(state_size * slots), expected_conv(conv * 3 * slots);
    for (std::size_t i = 0; i < expected_state.size(); ++i) {
        expected_state[i] = (static_cast<int>(i % 29) - 14) * 0.00048828125F;
    }
    for (std::size_t i = 0; i < expected_conv.size(); ++i) {
        expected_conv[i] = (static_cast<int>(i % 11) - 5) * 0.03125F;
    }
    DeviceBuffer state = to_device(expected_state), convolution = to_device_bf16(expected_conv);
    DeviceBuffer input(hidden * batch * sizeof(std::uint16_t));
    DeviceBuffer source(batch * sizeof(std::int32_t)), target(batch * sizeof(std::int32_t));
    GuardedDeviceBuffer output(hidden * batch * sizeof(std::uint16_t));
    Tensor x(input.p, DType::BF16, {hidden, batch});
    Tensor states(state.p, DType::FP32, {dim, dim, heads, slots});
    Tensor conv_states(convolution.p, DType::BF16, {conv, 3, slots});
    Tensor sources(source.p, DType::I32, {batch}), targets(target.p, DType::I32, {batch});
    Tensor destination(output.data(), DType::BF16, {hidden, batch});
    WorkspaceArena work(ops::flash_next_gdn_workspace_capacity_bytes(batch));
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create GDN stream");
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "begin GDN capture");
    ops::flash_next_gdn_batch_update(x, weights, conv_states, states, sources, targets, destination,
                                     work, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end GDN capture");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
               "instantiate GDN graph");
    std::vector<std::int32_t> src{0, 2}, dst{1, 3};
    int failures = 0;
    for (const float stimulus : {1.0F, 0.75F, -0.25F, 0.5F}) {
        std::vector<float> host_input(hidden * batch, 0.0F);
        std::vector<double> expected_output(hidden * batch, 0.0);
        for (int row = 0; row < batch; ++row) {
            const double value          = row == 0 ? stimulus : -stimulus;
            host_input[row * hidden]    = static_cast<float>(value);
            const double silu           = value / (1.0 + std::exp(-value));
            const double normalized_key = silu / std::sqrt(silu * silu + 1.0e-6);
            std::vector<double> readout(dim, 0.0);
            for (int head = 0; head < heads; ++head) {
                const double key = head < 3 ? normalized_key : 0.0;
                const auto src_base =
                    static_cast<std::size_t>(src[row]) * state_size + head * dim * dim;
                const auto dst_base =
                    static_cast<std::size_t>(dst[row]) * state_size + head * dim * dim;
                for (int v = 0; v < dim; ++v) {
                    const double projected_value = head == 0 && v == 0 ? silu : 0.0;
                    const double prediction      = 0.5 * expected_state[src_base + v * dim] * key;
                    const double delta           = 0.5 * (projected_value - prediction);
                    for (int k = 0; k < dim; ++k) {
                        const double updated = 0.5 * expected_state[src_base + v * dim + k] +
                                               (k == 0 ? key * delta : 0.0);
                        expected_state[dst_base + v * dim + k] = static_cast<float>(updated);
                    }
                    if (head == 0) {
                        readout[v] = key * expected_state[dst_base + v * dim] / std::sqrt(128.0);
                    }
                }
            }
            double square = 0.0;
            for (double v : readout) { square += v * v; }
            expected_output[row * hidden] =
                readout[0] / std::sqrt(square / dim + 1.0e-6) / (1.0 + std::exp(-value));
            for (int c = 0; c < conv; ++c) {
                const auto from              = static_cast<std::size_t>(src[row]) * conv * 3;
                const auto to                = static_cast<std::size_t>(dst[row]) * conv * 3;
                expected_conv[to + c]        = expected_conv[from + conv + c];
                expected_conv[to + conv + c] = expected_conv[from + 2 * conv + c];
                expected_conv[to + 2 * conv + c] =
                    c == 0 || c == qk || c == 2 * qk ? static_cast<float>(value) : 0.0F;
            }
        }
        std::vector<std::uint16_t> input_bits(host_input.size());
        for (std::size_t i = 0; i < host_input.size(); ++i) {
            input_bits[i] = f32_to_bf16(host_input[i]);
        }
        input.copy_from_host(input_bits.data(), input_bits.size() * sizeof(std::uint16_t));
        source.copy_from_host(src.data(), batch * sizeof(std::int32_t));
        target.copy_from_host(dst.data(), batch * sizeof(std::int32_t));
        cuda_check(cudaGraphLaunch(executable, stream), "replay GDN transition");
        cuda_synchronize(stream);
        std::vector<float> actual_state(expected_state.size());
        state.copy_to_host(actual_state.data(), state.bytes);
        failures += verify_pointwise(
            "GDN graph nonzero FP32 state oracle",
            std::vector<double>(actual_state.begin(), actual_state.end()),
            std::vector<double>(expected_state.begin(), expected_state.end()), {2.0e-3, 1.0e-2});
        failures += verify_pointwise("GDN graph complete block oracle",
                                     from_device_bf16(output.data(), hidden * batch),
                                     expected_output, {2.0e-2, 1.0e-2});
        failures += verify_exact("GDN graph convolution history",
                                 from_device_bf16(convolution.p, expected_conv.size()),
                                 std::vector<double>(expected_conv.begin(), expected_conv.end()));
        failures += output.verify_guards("GDN graph destination");
        src.swap(dst);
    }
    cuda_check(cudaGraphExecDestroy(executable), "destroy GDN executable");
    cuda_check(cudaGraphDestroy(graph), "destroy GDN graph");
    cuda_check(cudaStreamDestroy(stream), "destroy GDN stream");
    return failures;
}

int run() {
    constexpr int hidden      = 2560;
    constexpr int qk          = 2048;
    constexpr int value       = 6144;
    constexpr int convolution = 10240;
    const auto matrix         = [](int rows, int columns) {
        DeviceBuffer result(static_cast<std::size_t>(rows) * columns * sizeof(std::uint16_t));
        result.fill();
        return result;
    };

    std::vector<float> input(hidden, 0.0F);
    input[0]              = 1.0F;
    DeviceBuffer d_input  = to_device_bf16(input);
    DeviceBuffer d_a      = matrix(48, hidden);
    DeviceBuffer d_b      = matrix(48, hidden);
    DeviceBuffer d_qkv    = matrix(convolution, hidden);
    DeviceBuffer d_z      = matrix(value, hidden);
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
    Tensor input_tensor(d_input.p, DType::BF16, {hidden, 1});
    Tensor conv_state(d_conv_state.p, DType::BF16, {convolution, 3});
    Tensor recurrent_state(d_recurrent_state.p, DType::FP32, {128, 128, 48});
    Tensor destination(d_destination.data(), DType::BF16, {hidden, 1});
    WorkspaceArena workspace(ops::flash_next_gdn_workspace_capacity_bytes(1));
    ops::flash_next_gdn(input_tensor, weights, conv_state, conv_state, recurrent_state,
                        recurrent_state, destination, workspace, nullptr);
    cuda_synchronize();

    const auto bf16 = [](double value) {
        return static_cast<double>(bf16_to_f32(f32_to_bf16(static_cast<float>(value))));
    };
    const double silu_one      = bf16(1.0 / (1.0 + std::exp(-1.0)));
    const double normalized_qk = silu_one / std::sqrt(silu_one * silu_one + 1.0e-6);
    const double recurrent =
        bf16((1.0 / std::sqrt(128.0)) * 0.5 * silu_one * normalized_qk * normalized_qk);
    const double normalized = recurrent / std::sqrt(recurrent * recurrent / 128.0 + 1.0e-6);
    std::vector<double> expected(hidden, 0.0);
    expected[0]  = bf16(normalized * (1.0 / (1.0 + std::exp(-1.0))));
    int failures = verify_pointwise("Flash-Next GDN complete block",
                                    from_device_bf16(d_destination.data(), hidden), expected,
                                    {/*absolute*/ 2.0e-2, /*relative*/ 3.0e-3});
    failures += d_destination.verify_guards("Flash-Next GDN destination");
    failures += check_batch_state(weights);
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) { return 77; }
    try {
        const int failures = run();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Flash-Next GDN\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Flash-Next GDN: " << error.what() << '\n';
        return 1;
    }
}
