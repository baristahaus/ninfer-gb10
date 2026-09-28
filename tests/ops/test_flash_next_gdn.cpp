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
