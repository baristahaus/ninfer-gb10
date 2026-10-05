#include "ninfer/ops/hyperconnection.h"
#include "ops/op_tester.h"

#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr int kStreams = 4;
constexpr int kHidden = 2560;
constexpr int kHyper = kStreams * kHidden;
constexpr int kRank = 320;

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

// Row-scaled E4M3 weights: random finite codes and BF16 row scales; `values` receives the exact
// decoded weights for the oracle.
struct Fp8Rows {
    std::vector<std::uint8_t> codes;
    std::vector<std::uint16_t> scales;
};

Fp8Rows random_fp8_rows(int rows, int columns, float magnitude, std::uint32_t seed,
                        std::vector<float>& values) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> code(0, 255);
    std::uniform_real_distribution<float> scale(0.5F, 1.0F);
    Fp8Rows out{std::vector<std::uint8_t>(static_cast<std::size_t>(rows) * columns),
                std::vector<std::uint16_t>(rows)};
    values.resize(out.codes.size());
    for (int row = 0; row < rows; ++row) {
        out.scales[row] = f32_to_bf16(magnitude / 448.0F * scale(rng));
        const float row_scale = bf16_to_f32(out.scales[row]);
        for (int column = 0; column < columns; ++column) {
            std::uint8_t word = 0;
            do { word = static_cast<std::uint8_t>(code(rng)); } while ((word & 0x7F) == 0x7F);
            __nv_fp8_e4m3 decoded;
            decoded.__x = word;
            const std::size_t i = static_cast<std::size_t>(row) * columns + column;
            out.codes[i] = word;
            values[i] = static_cast<float>(decoded) * row_scale;
        }
    }
    return out;
}

// The registered row-scale payload: codes, then BF16 scales at the next 256-byte boundary.
std::vector<std::uint8_t> fp8_payload(const Fp8Rows& rows) {
    const std::size_t offset = (rows.codes.size() + 255) / 256 * 256;
    std::vector<std::uint8_t> payload(offset + rows.scales.size() * 2);
    std::copy(rows.codes.begin(), rows.codes.end(), payload.begin());
    std::memcpy(payload.data() + offset, rows.scales.data(), rows.scales.size() * 2);
    return payload;
}

Weight fp8_weight(const DeviceBuffer& payload, int rows, int columns) {
    Weight out = bf16_weight(payload, rows, columns);
    out.qtype = QType::FP8_E4M3FN_ROW_BF16;
    out.layout = QuantLayout::RowScale;
    out.group_size = static_cast<std::uint32_t>(columns);
    out.group = columns;
    out.scales = static_cast<const std::uint8_t*>(payload.p) +
                 (static_cast<std::size_t>(rows) * columns + 255) / 256 * 256;
    out.scale_dtype = DType::BF16;
    out.scale_ne[0] = rows;
    out.scale_nb[0] = 2;
    out.scale_nb[1] = out.scale_nb[2] = out.scale_nb[3] = static_cast<std::int64_t>(rows) * 2;
    return out;
}

// Dense weights exercise the small-token cooperative route (tokens <= 8) and the general route,
// with BF16 or row-scaled FP8 Down/Up.
int run(int kTokens, bool fp8) {
    std::vector<float> hyper(kHyper * kTokens), norm(kHyper);
    fill_uniform(hyper, 913, -0.75F, 0.75F);
    fill_uniform(norm, 914, -0.125F, 0.125F);
    round_to_bf16(hyper);
    round_to_bf16(norm);

    std::vector<float> down(static_cast<std::size_t>(kRank) * kHyper);
    std::vector<float> up(static_cast<std::size_t>(kHyper) * kRank);
    std::vector<float> inject(static_cast<std::size_t>(kStreams) * kHyper);
    fill_uniform(down, 916, -0.04F, 0.04F);
    fill_uniform(up, 917, -0.25F, 0.25F);
    fill_uniform(inject, 918, -0.02F, 0.02F);
    round_to_bf16(down);
    round_to_bf16(up);
    round_to_bf16(inject);
    Fp8Rows down_fp8, up_fp8;
    if (fp8) {
        down_fp8 = random_fp8_rows(kRank, kHyper, 0.04F, 919, down);
        up_fp8   = random_fp8_rows(kHyper, kRank, 0.25F, 920, up);
    }

    std::vector<double> normalized(hyper.size());
    std::vector<double> block_reference(kHidden * kTokens);
    std::vector<double> injection_reference(kStreams * kTokens);
    for (int token = 0; token < kTokens; ++token) {
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
    DeviceBuffer d_down = fp8 ? to_device(fp8_payload(down_fp8)) : to_device(down_bits);
    DeviceBuffer d_up = fp8 ? to_device(fp8_payload(up_fp8)) : to_device(up_bits);
    DeviceBuffer d_inject = to_device(inject_bits);
    GuardedDeviceBuffer d_block(block_reference.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_injection(injection_reference.size() * sizeof(std::uint16_t));
    Tensor hyper_tensor(d_hyper.p, DType::BF16, {kHyper, kTokens});
    Tensor norm_tensor(d_norm.p, DType::BF16, {kHyper});
    Tensor block_tensor(d_block.data(), DType::BF16, {kHidden, kTokens});
    Tensor injection_tensor(d_injection.data(), DType::BF16, {kStreams, kTokens});
    ops::HyperConnectionWeights weights{
        .norm = norm_tensor,
        .down = fp8 ? fp8_weight(d_down, kRank, kHyper) : bf16_weight(d_down, kRank, kHyper),
        .up = fp8 ? fp8_weight(d_up, kHyper, kRank) : bf16_weight(d_up, kHyper, kRank),
        .injection = bf16_weight(d_inject, kStreams, kHyper),
    };
    WorkspaceArena workspace(ops::hyperconnection_mix_workspace_capacity_bytes(kTokens, true));
    ops::hyperconnection_mix(hyper_tensor, weights, block_tensor, &injection_tensor, workspace, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise("HyperConnection mix", from_device_bf16(d_block.data(), block_reference.size()), block_reference, {/*absolute*/ 7.0e-3, /*relative*/ 2.0e-2});
    failures += verify_pointwise("HyperConnection injection", from_device_bf16(d_injection.data(), injection_reference.size()), injection_reference, {/*absolute*/ 7.0e-3, /*relative*/ 2.0e-2});

    std::vector<float> block_output(kHidden * kTokens);
    fill_uniform(block_output, 915, -0.5F, 0.5F);
    round_to_bf16(block_output);
    std::vector<double> combined(hyper.size());
    for (int token = 0; token < kTokens; ++token) {
        for (int stream = 0; stream < kStreams; ++stream) {
            const double scale = 2.0 / (1.0 + std::exp(-injection_reference[token * kStreams + stream] / kStreams));
            for (int d = 0; d < kHidden; ++d) { combined[token * kHyper + stream * kHidden + d] = hyper[token * kHyper + stream * kHidden + d] + scale * block_output[token * kHidden + d]; }
        }
    }
    DeviceBuffer d_block_output = to_device(encode(block_output));
    Tensor block_output_tensor(d_block_output.p, DType::BF16, {kHidden, kTokens});
    DeviceBuffer d_hyper_fused = to_device(hyper_bits);
    Tensor hyper_fused_tensor(d_hyper_fused.p, DType::BF16, {kHyper, kTokens});
    ops::hyperconnection_combine(hyper_tensor, block_output_tensor, injection_tensor, nullptr);
    GuardedDeviceBuffer d_next_block(block_reference.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_next_injection(injection_reference.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_fused_block(block_reference.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_fused_injection(injection_reference.size() * sizeof(std::uint16_t));
    Tensor next_block(d_next_block.data(), DType::BF16, {kHidden, kTokens});
    Tensor next_injection(d_next_injection.data(), DType::BF16, {kStreams, kTokens});
    Tensor fused_block(d_fused_block.data(), DType::BF16, {kHidden, kTokens});
    Tensor fused_injection(d_fused_injection.data(), DType::BF16, {kStreams, kTokens});
    ops::hyperconnection_mix(hyper_tensor, weights, next_block, &next_injection, workspace,
                             nullptr);
    ops::hyperconnection_combine_mix(hyper_fused_tensor, block_output_tensor, injection_tensor,
                                     weights, fused_block, &fused_injection, workspace, nullptr);
    cuda_synchronize();
    failures += verify_pointwise("HyperConnection combine", from_device_bf16(d_hyper, hyper.size()), combined, {/*absolute*/ 1.2e-2, /*relative*/ 2.0e-2});
    failures += verify_pointwise("HyperConnection fused state",
                                 from_device_bf16(d_hyper_fused, hyper.size()), combined,
                                 {/*absolute*/ 1.2e-2, /*relative*/ 2.0e-2});
    failures += verify_pointwise(
        "HyperConnection fused block", from_device_bf16(d_fused_block.data(), block_reference.size()),
        from_device_bf16(d_next_block.data(), block_reference.size()),
        {/*absolute*/ 0.0, /*relative*/ 0.0});
    failures += verify_pointwise(
        "HyperConnection fused injection",
        from_device_bf16(d_fused_injection.data(), injection_reference.size()),
        from_device_bf16(d_next_injection.data(), injection_reference.size()),
        {/*absolute*/ 0.0, /*relative*/ 0.0});
    failures += d_block.verify_guards("HyperConnection block input");
    failures += d_injection.verify_guards("HyperConnection injection");
    failures += d_next_block.verify_guards("HyperConnection next block input");
    failures += d_next_injection.verify_guards("HyperConnection next injection");
    failures += d_fused_block.verify_guards("HyperConnection fused block input");
    failures += d_fused_injection.verify_guards("HyperConnection fused injection");
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) { return 77; }
    try {
        int failures = 0;
        for (const bool fp8 : {false, true}) {
            for (const int tokens : {1, 2, 8, 17}) { failures += run(tokens, fp8); }
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " HyperConnection\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "HyperConnection: " << error.what() << '\n';
        return 1;
    }
}
