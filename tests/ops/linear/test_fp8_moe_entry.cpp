// Flash-Next FP8 MoE entry (1-8 tokens): router scores and shared SwiGLU against an FP64 oracle over
// the decoded row-scaled weights, and bitwise against the unfused linear + silu_mul route.
#include "ninfer/ops/linear.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/linear/fp8/flash_next_launch.h"
#include "ops/op_tester.h"

#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr int kHidden = 2560;
constexpr int kExperts = 512;
constexpr int kIntermediate = 640;

struct Fp8Matrix {
    std::vector<std::uint8_t> payload;  // codes, then BF16 row scales at a 256-byte boundary
    std::vector<double> values;         // decoded weights
};

Fp8Matrix random_fp8(int rows, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> code(0, 255);
    std::uniform_real_distribution<float> scale(0.5F, 1.0F);
    const std::size_t codes = static_cast<std::size_t>(rows) * kHidden;
    const std::size_t offset = (codes + 255) / 256 * 256;
    Fp8Matrix out{std::vector<std::uint8_t>(offset + 2 * rows), std::vector<double>(codes)};
    for (int row = 0; row < rows; ++row) {
        const std::uint16_t scale_bits = f32_to_bf16(0.05F / 448.0F * scale(rng));
        std::memcpy(out.payload.data() + offset + 2 * row, &scale_bits, 2);
        for (int k = 0; k < kHidden; ++k) {
            std::uint8_t word = 0;
            do { word = static_cast<std::uint8_t>(code(rng)); } while ((word & 0x7F) == 0x7F);
            __nv_fp8_e4m3 decoded;
            decoded.__x = word;
            const std::size_t i = static_cast<std::size_t>(row) * kHidden + k;
            out.payload[i] = word;
            out.values[i] = static_cast<double>(static_cast<float>(decoded)) * bf16_to_f32(scale_bits);
        }
    }
    return out;
}

Weight fp8_weight(const DeviceBuffer& payload, int rows) {
    Weight out{};
    out.payload = out.qdata = payload.p;
    out.payload_bytes = payload.bytes;
    out.qtype = QType::FP8_E4M3FN_ROW_BF16;
    out.layout = QuantLayout::RowScale;
    out.n = out.shape[0] = out.padded_shape[0] = rows;
    out.k = out.shape[1] = out.padded_shape[1] = kHidden;
    out.ndim = 2;
    out.group_size = kHidden;
    out.group = kHidden;
    out.scales = static_cast<const std::uint8_t*>(payload.p) +
                 (static_cast<std::size_t>(rows) * kHidden + 255) / 256 * 256;
    out.scale_dtype = DType::BF16;
    out.scale_ne[0] = rows;
    out.scale_nb[0] = 2;
    out.scale_nb[1] = out.scale_nb[2] = out.scale_nb[3] = static_cast<std::int64_t>(rows) * 2;
    return out;
}

// [rows, tokens] token-major dot products.
std::vector<double> dot_rows(const Fp8Matrix& matrix, int rows, const std::vector<float>& x,
                             int tokens) {
    std::vector<double> out(static_cast<std::size_t>(rows) * tokens);
    for (int token = 0; token < tokens; ++token) {
        for (int row = 0; row < rows; ++row) {
            double sum = 0.0;
            for (int k = 0; k < kHidden; ++k) {
                sum += matrix.values[static_cast<std::size_t>(row) * kHidden + k] *
                       x[static_cast<std::size_t>(token) * kHidden + k];
            }
            out[static_cast<std::size_t>(token) * rows + row] = sum;
        }
    }
    return out;
}

// Records each token count's fused outputs to check that a token's result does not depend on how
// many tokens share the launch.
std::vector<std::vector<double>> recorded(17);

int run(int tokens) {
    std::vector<float> input(static_cast<std::size_t>(kHidden) * tokens);
    fill_uniform(input, 931, -1.0F, 1.0F);
    round_to_bf16(input);
    const Fp8Matrix router = random_fp8(kExperts, 932);
    const Fp8Matrix gate = random_fp8(kIntermediate, 933);
    const Fp8Matrix up = random_fp8(kIntermediate, 934);

    const std::vector<double> scores_reference = dot_rows(router, kExperts, input, tokens);
    const std::vector<double> gate_reference = dot_rows(gate, kIntermediate, input, tokens);
    const std::vector<double> up_reference = dot_rows(up, kIntermediate, input, tokens);
    std::vector<double> activation_reference(gate_reference.size());
    for (std::size_t row = 0; row < activation_reference.size(); ++row) {
        const double g = gate_reference[row];
        activation_reference[row] = g / (1.0 + std::exp(-g)) * up_reference[row];
    }

    std::vector<std::uint16_t> input_bits(input.size());
    for (std::size_t k = 0; k < input.size(); ++k) { input_bits[k] = f32_to_bf16(input[k]); }
    DeviceBuffer d_input = to_device(input_bits);
    DeviceBuffer d_router = to_device(router.payload);
    DeviceBuffer d_gate = to_device(gate.payload);
    DeviceBuffer d_up = to_device(up.payload);
    GuardedDeviceBuffer d_scores(static_cast<std::size_t>(kExperts) * tokens * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_activation(static_cast<std::size_t>(kIntermediate) * tokens * sizeof(std::uint16_t));
    DeviceBuffer d_unfused_scores(static_cast<std::size_t>(kExperts) * tokens * sizeof(std::uint16_t));
    DeviceBuffer d_unfused_gate(static_cast<std::size_t>(kIntermediate) * tokens * sizeof(std::uint16_t));
    DeviceBuffer d_unfused_up(static_cast<std::size_t>(kIntermediate) * tokens * sizeof(std::uint16_t));
    DeviceBuffer d_unfused_activation(static_cast<std::size_t>(kIntermediate) * tokens * sizeof(std::uint16_t));

    const Tensor x(d_input.p, DType::BF16, {kHidden, tokens});
    const Weight router_weight = fp8_weight(d_router, kExperts);
    const Weight gate_weight = fp8_weight(d_gate, kIntermediate);
    const Weight up_weight = fp8_weight(d_up, kIntermediate);
    Tensor scores(d_scores.data(), DType::BF16, {kExperts, tokens});
    Tensor activation(d_activation.data(), DType::BF16, {kIntermediate, tokens});
    ops::detail::flash_next::launch_fp8_moe_entry_decode(x, router_weight, gate_weight, up_weight,
                                                         scores, activation, nullptr);
    Tensor unfused_scores(d_unfused_scores.p, DType::BF16, {kExperts, tokens});
    Tensor unfused_gate(d_unfused_gate.p, DType::BF16, {kIntermediate, tokens});
    Tensor unfused_up(d_unfused_up.p, DType::BF16, {kIntermediate, tokens});
    Tensor unfused_activation(d_unfused_activation.p, DType::BF16, {kIntermediate, tokens});
    ops::linear(x, router_weight, unfused_scores, nullptr);
    ops::linear(x, gate_weight, unfused_gate, nullptr);
    ops::linear(x, up_weight, unfused_up, nullptr);
    ops::silu_mul(unfused_gate, unfused_up, unfused_activation, nullptr);
    cuda_synchronize();

    const auto got_scores = from_device_bf16(d_scores.data(), scores_reference.size());
    const auto got_activation = from_device_bf16(d_activation.data(), activation_reference.size());
    recorded[tokens].clear();
    for (int token = 0; token < tokens; ++token) {
        recorded[tokens].insert(recorded[tokens].end(), got_scores.begin() + token * kExperts,
                                got_scores.begin() + (token + 1) * kExperts);
        recorded[tokens].insert(recorded[tokens].end(),
                                got_activation.begin() + token * kIntermediate,
                                got_activation.begin() + (token + 1) * kIntermediate);
    }
    int failures = verify_pointwise("FP8 MoE entry scores", got_scores, scores_reference,
                                    {/*absolute*/ 2.0e-3, /*relative*/ 1.0e-2});
    failures += verify_pointwise("FP8 MoE entry SwiGLU", got_activation, activation_reference,
                                 {/*absolute*/ 2.0e-3, /*relative*/ 2.0e-2});
    failures += verify_pointwise("FP8 MoE entry scores vs unfused", got_scores,
                                 from_device_bf16(d_unfused_scores.p, scores_reference.size()),
                                 {/*absolute*/ 0.0, /*relative*/ 0.0});
    failures += verify_pointwise("FP8 MoE entry SwiGLU vs unfused", got_activation,
                                 from_device_bf16(d_unfused_activation.p, activation_reference.size()),
                                 {/*absolute*/ 0.0, /*relative*/ 0.0});
    failures += d_scores.verify_guards("FP8 MoE entry scores");
    failures += d_activation.verify_guards("FP8 MoE entry SwiGLU");
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) { return 77; }
    try {
        int failures = 0;
        for (const int tokens : {1, 2, 4, 8, 9, 12, 16}) { failures += run(tokens); }
        // The 4-token (8-token tile) and 12-token (16-token tile) launches share their inputs'
        // first four tokens.
        std::vector<double> wide(recorded[12].begin(),
                                 recorded[12].begin() + static_cast<std::ptrdiff_t>(recorded[4].size()));
        failures += verify_pointwise("FP8 MoE entry token-tile invariance", wide, recorded[4],
                                     {/*absolute*/ 0.0, /*relative*/ 0.0});
        std::cout << (failures == 0 ? "OK" : "FAIL") << " FP8 MoE entry\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FP8 MoE entry: " << error.what() << '\n';
        return 1;
    }
}
