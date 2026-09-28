#include "ninfer/ops/hyperconnection.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr int kStreams = 4;
constexpr int kHidden = 2560;
constexpr int kHyper = kStreams * kHidden;
constexpr int kRank = 320;

enum class DownFormat { Bf16, Fp8 };

std::vector<std::uint16_t> encode(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

Weight bf16_weight(const DeviceBuffer& storage, int rows, int columns) {
    Weight out{};
    out.payload = storage.p;
    out.payload_bytes = storage.bytes;
    out.qtype = QType::BF16;
    out.qdata = storage.p;
    out.n = rows;
    out.k = columns;
    out.ndim = 2;
    out.shape[0] = out.padded_shape[0] = rows;
    out.shape[1] = out.padded_shape[1] = columns;
    out.layout = QuantLayout::Contiguous;
    return out;
}

// The down projection is either a sparse BF16 matrix or a dense patterned row-scaled FP8 matrix;
// the FP64 reference uses the exactly decoded represented weights in both cases.
int run_case(int tokens, DownFormat format) {
    const std::string label = std::string("HyperConnection ") +
                              (format == DownFormat::Fp8 ? "FP8" : "BF16") +
                              " T=" + std::to_string(tokens);
    std::vector<float> hyper(kHyper * tokens), norm(kHyper);
    fill_uniform(hyper, 913, -0.75F, 0.75F);
    fill_uniform(norm, 914, -0.125F, 0.125F);
    round_to_bf16(hyper);
    round_to_bf16(norm);

    std::vector<float> down(static_cast<std::size_t>(kRank) * kHyper, 0.0F);
    std::vector<float> up(static_cast<std::size_t>(kHyper) * kRank, 0.0F);
    std::vector<float> inject(static_cast<std::size_t>(kStreams) * kHyper, 0.0F);
    const auto fp8_down = quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16,
                                                                  kRank, kHyper, 917U);
    if (format == DownFormat::Fp8) {
        std::vector<std::int32_t> rows(kRank);
        for (int row = 0; row < kRank; ++row) { rows[row] = row; }
        down = quantized_weight::materialize_rows_fp32(fp8_down, rows);
    } else {
        for (int row = 0; row < kRank; ++row) { down[static_cast<std::size_t>(row) * kHyper + (37 * row) % kHyper] = 0.25F; }
    }
    for (int row = 0; row < kHyper; ++row) { up[static_cast<std::size_t>(row) * kRank + row % kRank] = (row & 1) ? -0.5F : 0.5F; }
    for (int row = 0; row < kStreams; ++row) { inject[static_cast<std::size_t>(row) * kHyper + 777 * (row + 1)] = 0.75F; }

    std::vector<double> normalized(hyper.size());
    std::vector<double> block_reference(kHidden * tokens);
    std::vector<double> injection_reference(kStreams * tokens);
    for (int token = 0; token < tokens; ++token) {
        for (int stream = 0; stream < kStreams; ++stream) {
            const int base = token * kHyper + stream * kHidden;
            double square_sum = 0.0;
            for (int d = 0; d < kHidden; ++d) { square_sum += double(hyper[base + d]) * hyper[base + d]; }
            const double inverse = 1.0 / std::sqrt(square_sum / kHidden + 1.0e-6);
            for (int d = 0; d < kHidden; ++d) { normalized[base + d] = hyper[base + d] * inverse * (1.0 + norm[stream * kHidden + d]); }
        }
        std::vector<double> low(kRank);
        for (int row = 0; row < kRank; ++row) {
            double value = 0.0;
            for (int column = 0; column < kHyper; ++column) { value += down[static_cast<std::size_t>(row) * kHyper + column] * normalized[token * kHyper + column]; }
            value /= kStreams;
            low[row] = value / (1.0 + std::exp(-value));
        }
        for (int d = 0; d < kHidden; ++d) {
            double value = 0.0;
            for (int stream = 0; stream < kStreams; ++stream) {
                const int row = stream * kHidden + d;
                double logit = 0.0;
                for (int rank = 0; rank < kRank; ++rank) { logit += up[static_cast<std::size_t>(row) * kRank + rank] * low[rank]; }
                value += normalized[token * kHyper + row] / (1.0 + std::exp(-logit));
            }
            block_reference[token * kHidden + d] = value / kStreams;
        }
        for (int stream = 0; stream < kStreams; ++stream) {
            double value = 0.0;
            for (int column = 0; column < kHyper; ++column) { value += inject[static_cast<std::size_t>(stream) * kHyper + column] * normalized[token * kHyper + column]; }
            injection_reference[token * kStreams + stream] = value;
        }
    }

    const auto hyper_bits = encode(hyper);
    const auto norm_bits = encode(norm);
    const auto down_bits = encode(down);
    const auto up_bits = encode(up);
    const auto inject_bits = encode(inject);
    DeviceBuffer d_hyper = to_device(hyper_bits);
    DeviceBuffer d_norm = to_device(norm_bits);
    DeviceBuffer d_down = format == DownFormat::Fp8 ? to_device(fp8_down.payload)
                                                    : to_device(down_bits);
    DeviceBuffer d_up = to_device(up_bits);
    DeviceBuffer d_inject = to_device(inject_bits);
    GuardedDeviceBuffer d_block(block_reference.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_injection(injection_reference.size() * sizeof(std::uint16_t));
    Tensor hyper_tensor(d_hyper.p, DType::BF16, {kHyper, tokens});
    Tensor norm_tensor(d_norm.p, DType::BF16, {kHyper});
    Tensor block_tensor(d_block.data(), DType::BF16, {kHidden, tokens});
    Tensor injection_tensor(d_injection.data(), DType::BF16, {kStreams, tokens});
    ops::HyperConnectionWeights weights{
        .norm = norm_tensor,
        .down = format == DownFormat::Fp8 ? fp8_down.device_weight(d_down.p)
                                          : bf16_weight(d_down, kRank, kHyper),
        .up = bf16_weight(d_up, kHyper, kRank),
        .injection = bf16_weight(d_inject, kStreams, kHyper),
    };
    WorkspaceArena workspace(ops::hyperconnection_mix_workspace_capacity_bytes(tokens, true));
    ops::hyperconnection_mix(hyper_tensor, weights, block_tensor, &injection_tensor, workspace, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise(label + " mix", from_device_bf16(d_block.data(), block_reference.size()), block_reference, {/*absolute*/ 7.0e-3, /*relative*/ 2.0e-2});
    failures += verify_pointwise(label + " injection", from_device_bf16(d_injection.data(), injection_reference.size()), injection_reference, {/*absolute*/ 7.0e-3, /*relative*/ 2.0e-2});

    std::vector<float> block_output(kHidden * tokens);
    fill_uniform(block_output, 915, -0.5F, 0.5F);
    round_to_bf16(block_output);
    std::vector<double> combined(hyper.size());
    for (int token = 0; token < tokens; ++token) {
        for (int stream = 0; stream < kStreams; ++stream) {
            const double scale = 2.0 / (1.0 + std::exp(-injection_reference[token * kStreams + stream] / kStreams));
            for (int d = 0; d < kHidden; ++d) { combined[token * kHyper + stream * kHidden + d] = hyper[token * kHyper + stream * kHidden + d] + scale * block_output[token * kHidden + d]; }
        }
    }
    DeviceBuffer d_block_output = to_device(encode(block_output));
    Tensor block_output_tensor(d_block_output.p, DType::BF16, {kHidden, tokens});
    DeviceBuffer d_hyper_fused = to_device(hyper_bits);
    Tensor hyper_fused_tensor(d_hyper_fused.p, DType::BF16, {kHyper, tokens});
    ops::hyperconnection_combine(hyper_tensor, block_output_tensor, injection_tensor, nullptr);
    GuardedDeviceBuffer d_next_block(block_reference.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_next_injection(injection_reference.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_fused_block(block_reference.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_fused_injection(injection_reference.size() * sizeof(std::uint16_t));
    Tensor next_block(d_next_block.data(), DType::BF16, {kHidden, tokens});
    Tensor next_injection(d_next_injection.data(), DType::BF16, {kStreams, tokens});
    Tensor fused_block(d_fused_block.data(), DType::BF16, {kHidden, tokens});
    Tensor fused_injection(d_fused_injection.data(), DType::BF16, {kStreams, tokens});
    ops::hyperconnection_mix(hyper_tensor, weights, next_block, &next_injection, workspace,
                             nullptr);
    ops::hyperconnection_combine_mix(hyper_fused_tensor, block_output_tensor, injection_tensor,
                                     weights, fused_block, &fused_injection, workspace, nullptr);
    cuda_synchronize();
    failures += verify_pointwise(label + " combine", from_device_bf16(d_hyper, hyper.size()), combined, {/*absolute*/ 1.2e-2, /*relative*/ 2.0e-2});
    failures += verify_pointwise(label + " fused state",
                                 from_device_bf16(d_hyper_fused, hyper.size()), combined,
                                 {/*absolute*/ 1.2e-2, /*relative*/ 2.0e-2});
    failures += verify_pointwise(
        label + " fused block", from_device_bf16(d_fused_block.data(), block_reference.size()),
        from_device_bf16(d_next_block.data(), block_reference.size()),
        {/*absolute*/ 0.0, /*relative*/ 0.0});
    failures += verify_pointwise(
        label + " fused injection",
        from_device_bf16(d_fused_injection.data(), injection_reference.size()),
        from_device_bf16(d_next_injection.data(), injection_reference.size()),
        {/*absolute*/ 0.0, /*relative*/ 0.0});
    failures += d_block.verify_guards(label + " block input");
    failures += d_injection.verify_guards(label + " injection");
    failures += d_next_block.verify_guards(label + " next block input");
    failures += d_next_injection.verify_guards(label + " next injection");
    failures += d_fused_block.verify_guards(label + " fused block input");
    failures += d_fused_injection.verify_guards(label + " fused injection");
    return failures;
}

int run() {
    int failures = 0;
    for (const DownFormat format : {DownFormat::Bf16, DownFormat::Fp8}) {
        // T=1 GEMV, T=2/3 (3 is the MTP verify width) and T=9 sliced-K, T=65 tiled GEMM.
        for (const int tokens : {1, 2, 3, 9, 65}) { failures += run_case(tokens, format); }
    }
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) { return 77; }
    try {
        const int failures = run();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " HyperConnection\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "HyperConnection: " << error.what() << '\n';
        return 1;
    }
}
