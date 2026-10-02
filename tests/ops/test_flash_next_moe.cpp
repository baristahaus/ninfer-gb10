#include "ninfer/ops/flash_next_moe.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <string>
#include <span>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr int kHidden = 2560;
constexpr int kExperts = 512;
constexpr int kIntermediate = 640;
constexpr int kTop = 10;
constexpr int kGroupedTokens = 17;

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

// A row-scaled FP8 matrix whose only nonzero codes are the given exact E4M3 values, with every
// row multiplier 1.0.
quantized_weight::PackedWeight
sparse_fp8(int rows, int columns,
           std::initializer_list<std::pair<std::size_t, std::uint8_t>> codes) {
    auto packed =
        quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16, rows, columns, 0U);
    std::fill(packed.payload.begin(), packed.payload.end(), 0);
    for (const auto& [element, code] : codes) { packed.payload[element] = code; }
    const std::uint16_t one = f32_to_bf16(1.0F);
    for (int row = 0; row < rows; ++row) {
        std::memcpy(packed.payload.data() + packed.scale_plane_offset + 2 * row, &one, 2);
    }
    return packed;
}

int run() {
    std::vector<float> input(static_cast<std::size_t>(kHidden) * kGroupedTokens, 0.0F);
    for (int token = 0; token < kGroupedTokens; ++token) {
        input[static_cast<std::size_t>(token) * kHidden] = 0.5F;
        input[static_cast<std::size_t>(token) * kHidden + 1] = 0.25F;
        input[static_cast<std::size_t>(token) * kHidden + 2] = -0.5F;
    }
    round_to_bf16(input);
    std::vector<std::uint16_t> input_bits(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) { input_bits[i] = f32_to_bf16(input[i]); }

    DeviceBuffer d_input = to_device(input_bits);
    DeviceBuffer d_router(static_cast<std::size_t>(kExperts) * kHidden * sizeof(std::uint16_t));
    DeviceBuffer d_shared_gate(static_cast<std::size_t>(kIntermediate) * kHidden * sizeof(std::uint16_t));
    DeviceBuffer d_shared_up(static_cast<std::size_t>(kIntermediate) * kHidden * sizeof(std::uint16_t));
    DeviceBuffer d_shared_down(static_cast<std::size_t>(kHidden) * kIntermediate * sizeof(std::uint16_t));
    DeviceBuffer d_shared_scale(static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t));
    DeviceBuffer d_routed_gate_up(static_cast<std::size_t>(kExperts) * 2 * kIntermediate *
                                  kHidden * sizeof(std::uint16_t));
    DeviceBuffer d_routed_down(static_cast<std::size_t>(kExperts) * kHidden * kIntermediate *
                               sizeof(std::uint16_t));
    d_router.fill();
    d_shared_gate.fill();
    d_shared_up.fill();
    d_shared_down.fill();
    d_shared_scale.fill();
    d_routed_gate_up.fill();
    d_routed_down.fill();

    store_bf16(d_shared_gate, 0, 1.0F);
    store_bf16(d_shared_up, 1, 2.0F);
    store_bf16(d_shared_down, 0, 1.0F);
    store_bf16(d_shared_scale, 2, 1.0F);
    const std::size_t gate_up_expert_stride = static_cast<std::size_t>(2 * kIntermediate) * kHidden;
    const std::size_t down_expert_stride = static_cast<std::size_t>(kHidden) * kIntermediate;
    for (int expert = 0; expert < kTop; ++expert) {
        const std::size_t gate_up_base = static_cast<std::size_t>(expert) * gate_up_expert_stride;
        store_bf16(d_routed_gate_up, gate_up_base, 1.0F);
        store_bf16(d_routed_gate_up,
                   gate_up_base + static_cast<std::size_t>(kIntermediate) * kHidden + 1, 2.0F);
        store_bf16(d_routed_down, static_cast<std::size_t>(expert) * down_expert_stride, 1.0F);
    }

    GuardedDeviceBuffer d_scalar_output(static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_grouped_output(static_cast<std::size_t>(kHidden) * kGroupedTokens *
                                         sizeof(std::uint16_t));
    Tensor scalar_input(d_input.p, DType::BF16, {kHidden, 1});
    Tensor scalar_output(d_scalar_output.data(), DType::BF16, {kHidden, 1});
    Tensor grouped_input(d_input.p, DType::BF16, {kHidden, kGroupedTokens});
    Tensor grouped_output(d_grouped_output.data(), DType::BF16, {kHidden, kGroupedTokens});
    ops::FlashNextMoeWeights weights{
        .router = bf16_weight(d_router, kExperts, kHidden),
        .shared_gate_up =
            ops::FlashNextSharedGateUpPair{
                .gate = bf16_weight(d_shared_gate, kIntermediate, kHidden),
                .up   = bf16_weight(d_shared_up, kIntermediate, kHidden),
            },
        .shared_down    = bf16_weight(d_shared_down, kHidden, kIntermediate),
        .shared_scale   = bf16_weight(d_shared_scale, 1, kHidden),
        .routed_gate_up = {.codes   = d_routed_gate_up.p,
                           .qtype   = QType::BF16,
                           .experts = kExperts,
                           .rows    = 2 * kIntermediate,
                           .columns = kHidden},
        .routed_down    = {.codes   = d_routed_down.p,
                           .qtype   = QType::BF16,
                           .experts = kExperts,
                           .rows    = kHidden,
                           .columns = kIntermediate},
    };
    const double activation = (0.5 / (1.0 + std::exp(-0.5))) * 0.5;
    std::vector<double> expected(kHidden, 0.0);
    expected[0] = activation * (1.0 + 1.0 / (1.0 + std::exp(0.5)));
    std::vector<double> grouped_expected;
    grouped_expected.reserve(static_cast<std::size_t>(kHidden) * kGroupedTokens);
    for (int token = 0; token < kGroupedTokens; ++token) {
        grouped_expected.insert(grouped_expected.end(), expected.begin(), expected.end());
    }

    WorkspaceArena scalar_workspace(ops::flash_next_moe_workspace_capacity_bytes(1));
    WorkspaceArena grouped_workspace(ops::flash_next_moe_workspace_capacity_bytes(kGroupedTokens));
    const auto run_weights = [&](const std::string& label) {
        ops::flash_next_moe(scalar_input, weights, scalar_output, scalar_workspace, nullptr);
        ops::flash_next_moe(grouped_input, weights, grouped_output, grouped_workspace, nullptr);
        cuda_synchronize();
        int result =
            verify_pointwise(label + " scalar", from_device_bf16(d_scalar_output.data(), kHidden),
                             expected, {/*absolute*/ 4.0e-3, /*relative*/ 2.0e-2});
        result +=
            verify_pointwise(label + " grouped",
                             from_device_bf16(d_grouped_output.data(),
                                              static_cast<std::size_t>(kHidden) * kGroupedTokens),
                             grouped_expected, {/*absolute*/ 4.0e-3, /*relative*/ 2.0e-2});
        result += d_scalar_output.verify_guards(label + " scalar output");
        result += d_grouped_output.verify_guards(label + " grouped output");
        return result;
    };
    int failures = run_weights("Flash-Next BF16 MoE");

    // The FP8 recipe's shared expert: one packed gate/up parent (gate row 0 reads x0 with 1.0,
    // up row 640 reads x1 with 2.0) and an FP8 down projection, representing the same values.
    constexpr std::uint8_t kOne = 0x38, kTwo = 0x40;
    const auto fp8_gate_up =
        sparse_fp8(2 * kIntermediate, kHidden,
                   {{0, kOne}, {static_cast<std::size_t>(kIntermediate) * kHidden + 1, kTwo}});
    const auto fp8_down        = sparse_fp8(kHidden, kIntermediate, {{0, kOne}});
    DeviceBuffer d_fp8_gate_up = to_device(fp8_gate_up.payload);
    DeviceBuffer d_fp8_down    = to_device(fp8_down.payload);
    weights.shared_gate_up     = fp8_gate_up.device_weight(d_fp8_gate_up.p);
    weights.shared_down        = fp8_down.device_weight(d_fp8_down.p);
    failures += run_weights("Flash-Next FP8 shared-expert MoE");
    return failures;
}

// The production NVFP4 W4A4 routes against an FP64 oracle: 1 and 2 rows (the per-assignment
// decode route), 8 and 16 rows (the expert-grouped decode tile) and 17 rows (the grouped prefill
// tile). W4A4 has
// two explicit activation quantizations, the expert input and the down-projection input, each
// NVFP4 per 16 values with an E4M3 scale RNE(divisor * max|x| / 6) and E2M1 codes RNE(x *
// divisor / scale), both saturating. The oracle applies them to the BF16 values they quantize
// (the input, and the BF16 SiLU(gate) * up activation) and evaluates everything else in FP64 on
// the decoded weights. Routing is exact: each token's 10 experts score distinct BF16 values on a
// dedicated input dimension, and tokens share experts from a pool of 12, as decode rows do.
// Token t's routing dimension is kRoutingDim + t; every token needs one inside the hidden row.
constexpr int kMaxTokens  = 17;
constexpr int kRoutingDim = kHidden - 32;
static_assert(kRoutingDim + kMaxTokens <= kHidden);
constexpr std::array<int, 12> kPool{3, 41, 88, 130, 177, 211, 260, 305, 349, 402, 455, 509};
// Residual differences are FP32 accumulation order against FP64, BF16 rounding of the gate, up
// and down outputs, and the rare activation block whose BF16 value lands on the other side of a
// code boundary because of them.
constexpr ReductionCriterion kNvfp4MoeCriterion{2.0e-2, 0.0, 5.0e-2};

std::uint8_t encode_e4m3_rne_satfinite(float value) {
    // Nearest positive E4M3FN value, ties to the even code; values above 448 saturate.
    if (value >= 448.0F) { return 0x7e; }
    std::uint8_t best = 0;
    double best_error = std::abs(static_cast<double>(value));
    for (int code = 1; code < 0x7f; ++code) {
        const double error =
            std::abs(quantized_weight::detail::decode_e4m3fn(static_cast<std::uint8_t>(code)) -
                     static_cast<double>(value));
        if (error < best_error || (error == best_error && (code & 1) == 0)) {
            best       = static_cast<std::uint8_t>(code);
            best_error = error;
        }
    }
    return best;
}

double quantize_e2m1_rne_satfinite(float value) {
    constexpr std::array<double, 8> kMagnitudes{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double magnitude = std::min(std::abs(static_cast<double>(value)), 6.0);
    int best               = 0;
    for (int code = 1; code < 8; ++code) {
        const double error = std::abs(kMagnitudes[code] - magnitude);
        const double held  = std::abs(kMagnitudes[best] - magnitude);
        if (error < held || (error == held && (code & 1) == 0)) { best = code; }
    }
    return value < 0.0F ? -kMagnitudes[best] : kMagnitudes[best];
}

// The logical value the W4A4 product sees for each element of a BF16 vector: code * scale /
// divisor, per 16-element block.
std::vector<double> nvfp4_activation(const std::vector<float>& bf16_values, float divisor) {
    std::vector<double> out(bf16_values.size());
    for (std::size_t block = 0; block < bf16_values.size(); block += 16) {
        float max_abs = 0.0F;
        for (std::size_t i = 0; i < 16; ++i) {
            max_abs = std::max(max_abs, std::abs(bf16_values[block + i]));
        }
        const std::uint8_t scale_code = encode_e4m3_rne_satfinite(divisor * max_abs / 6.0F);
        const auto scale = static_cast<float>(quantized_weight::detail::decode_e4m3fn(scale_code));
        for (std::size_t i = 0; i < 16; ++i) {
            out[block + i] =
                scale_code == 0
                    ? 0.0
                    : quantize_e2m1_rne_satfinite(bf16_values[block + i] * divisor / scale) *
                          static_cast<double>(scale) / static_cast<double>(divisor);
        }
    }
    return out;
}

float bf16_round(double value) { return bf16_to_f32(f32_to_bf16(static_cast<float>(value))); }

struct Nvfp4Bank {
    DeviceBuffer device;
    std::vector<quantized_weight::PackedWeight> experts; // parallel to kPool
    std::vector<float> input_divisors;                   // indexed by expert id
    WeightGeometry geometry;

    Nvfp4Bank(int rows, int columns, std::uint32_t seed) : input_divisors(kExperts, 1.0F) {
        const std::array<std::uint64_t, 3> shape{kExperts, static_cast<std::uint64_t>(rows),
                                                 static_cast<std::uint64_t>(columns)};
        geometry = weight_geometry(QType::NVFP4, QuantLayout::ExpertBlockScaleK16M128x4, shape);
        device   = DeviceBuffer(geometry.bytes);
        device.fill(); // unused experts: zero codes and scales
        std::vector<float> divisors(kExperts, 1.0F);
        const std::uint64_t codes  = static_cast<std::uint64_t>(rows) * columns / 2;
        const std::uint64_t scales = static_cast<std::uint64_t>(rows) * columns / 16;
        for (std::size_t slot = 0; slot < kPool.size(); ++slot) {
            const int expert           = kPool[slot];
            const float weight_divisor = 0.75F + 0.125F * static_cast<float>(slot % 4);
            const float input_divisor  = 0.5F + 0.25F * static_cast<float>(slot % 3);
            experts.push_back(quantized_weight::make_patterned_weight(
                QType::NVFP4, rows, columns, seed + static_cast<std::uint32_t>(slot),
                {.weight_scale_divisor = weight_divisor, .input_scale_divisor = input_divisor}));
            const auto& packed = experts.back();
            device.copy_from_host(packed.payload.data(), codes, expert * codes);
            device.copy_from_host(packed.payload.data() + packed.scale_plane_offset, scales,
                                  geometry.scale_offset + expert * scales);
            divisors[expert]       = weight_divisor;
            input_divisors[expert] = input_divisor;
        }
        device.copy_from_host(divisors.data(), divisors.size() * sizeof(float),
                              geometry.divisor_offset);
    }

    ops::FlashNextExpertBank view(const DeviceBuffer& input_divisor_buffer, int rows,
                                  int columns) const {
        const auto* base = static_cast<const std::uint8_t*>(device.p);
        return {.codes  = base,
                .scales = base + geometry.scale_offset,
                .weight_scale_divisors =
                    reinterpret_cast<const float*>(base + geometry.divisor_offset),
                .input_scale_divisors = static_cast<const float*>(input_divisor_buffer.p),
                .qtype                = QType::NVFP4,
                .experts              = kExperts,
                .rows                 = rows,
                .columns              = columns};
    }
};

int run_nvfp4() {
    std::vector<float> input(static_cast<std::size_t>(kHidden) * kMaxTokens);
    fill_uniform(input, 4101, -1.0F, 1.0F);
    for (int t = 0; t < kMaxTokens; ++t) {
        for (int d = kRoutingDim; d < kHidden; ++d) {
            input[static_cast<std::size_t>(t) * kHidden + d] = d == kRoutingDim + t ? 1.0F : 0.0F;
        }
    }
    round_to_bf16(input);

    const Nvfp4Bank gate_up(2 * kIntermediate, kHidden, 4200);
    const Nvfp4Bank down(kHidden, kIntermediate, 4300);
    DeviceBuffer d_gate_divisors = to_device(gate_up.input_divisors);
    DeviceBuffer d_down_divisors = to_device(down.input_divisors);
    DeviceBuffer d_router(static_cast<std::size_t>(kExperts) * kHidden * sizeof(std::uint16_t));
    DeviceBuffer d_input = to_device_bf16(input);
    // Zero shared expert: shared_alpha * shared(x) is exactly zero, isolating the routed paths.
    DeviceBuffer d_shared_gate(static_cast<std::size_t>(kIntermediate) * kHidden * 2);
    DeviceBuffer d_shared_up(static_cast<std::size_t>(kIntermediate) * kHidden * 2);
    DeviceBuffer d_shared_down(static_cast<std::size_t>(kHidden) * kIntermediate * 2);
    DeviceBuffer d_shared_scale(static_cast<std::size_t>(kHidden) * 2);
    d_shared_gate.fill();
    d_shared_up.fill();
    d_shared_down.fill();
    d_shared_scale.fill();
    const ops::FlashNextMoeWeights weights{
        .router = bf16_weight(d_router, kExperts, kHidden),
        .shared_gate_up =
            ops::FlashNextSharedGateUpPair{
                .gate = bf16_weight(d_shared_gate, kIntermediate, kHidden),
                .up   = bf16_weight(d_shared_up, kIntermediate, kHidden),
            },
        .shared_down    = bf16_weight(d_shared_down, kHidden, kIntermediate),
        .shared_scale   = bf16_weight(d_shared_scale, 1, kHidden),
        .routed_gate_up = gate_up.view(d_gate_divisors, 2 * kIntermediate, kHidden),
        .routed_down    = down.view(d_down_divisors, kHidden, kIntermediate),
    };

    // Each token's 10 experts (pool slots) and their router scores (exact BF16 values on the
    // token's routing dimension; every other expert scores 0).
    using Selection          = std::vector<std::array<std::pair<int, float>, kTop>>;
    const auto upload_router = [&](const Selection& selection) {
        std::vector<std::uint16_t> router(static_cast<std::size_t>(kExperts) * kHidden,
                                          f32_to_bf16(0.0F));
        for (std::size_t t = 0; t < selection.size(); ++t) {
            for (const auto& [slot, score] : selection[t]) {
                router[static_cast<std::size_t>(kPool[slot]) * kHidden + kRoutingDim + t] =
                    f32_to_bf16(score);
            }
        }
        d_router.copy_from_host(router.data(), router.size() * sizeof(std::uint16_t));
    };

    // FP64 oracle with the two explicit activation quantizations.
    const auto oracle = [&](const Selection& selection) {
        std::vector<double> expected(static_cast<std::size_t>(kHidden) * selection.size(), 0.0);
        std::vector<std::int32_t> gate_up_rows(2 * kIntermediate);
        for (int r = 0; r < 2 * kIntermediate; ++r) { gate_up_rows[r] = r; }
        std::vector<std::int32_t> down_rows(kHidden);
        for (int r = 0; r < kHidden; ++r) { down_rows[r] = r; }
        for (std::size_t slot = 0; slot < kPool.size(); ++slot) {
            std::vector<std::pair<std::size_t, double>> users; // token, alpha
            for (std::size_t t = 0; t < selection.size(); ++t) {
                double maximum = -1.0e30;
                for (const auto& entry : selection[t]) {
                    maximum = std::max(maximum, static_cast<double>(entry.second));
                }
                double denominator = 0.0;
                for (const auto& entry : selection[t]) {
                    denominator += std::exp(static_cast<double>(entry.second) - maximum);
                }
                for (const auto& [chosen, score] : selection[t]) {
                    if (chosen == static_cast<int>(slot)) {
                        users.emplace_back(t, std::exp(static_cast<double>(score) - maximum) /
                                                  denominator);
                    }
                }
            }
            if (users.empty()) { continue; }
            const int expert            = kPool[slot];
            const std::vector<float> w1 = quantized_weight::materialize_rows_fp32(
                gate_up.experts[slot], std::span<const std::int32_t>(gate_up_rows));
            const std::vector<float> w2 = quantized_weight::materialize_rows_fp32(
                down.experts[slot], std::span<const std::int32_t>(down_rows));
            for (const auto& [t, alpha] : users) {
                const std::vector<float> row(
                    input.begin() + static_cast<std::ptrdiff_t>(t) * kHidden,
                    input.begin() + static_cast<std::ptrdiff_t>(t + 1) * kHidden);
                const std::vector<double> x = nvfp4_activation(row, gate_up.input_divisors[expert]);
                std::vector<float> activation(kIntermediate);
                for (int r = 0; r < kIntermediate; ++r) {
                    double gate = 0.0, up = 0.0;
                    for (int k = 0; k < kHidden; ++k) {
                        gate += static_cast<double>(w1[static_cast<std::size_t>(r) * kHidden + k]) *
                                x[k];
                        up += static_cast<double>(
                                  w1[static_cast<std::size_t>(r + kIntermediate) * kHidden + k]) *
                              x[k];
                    }
                    // The quantizer reads the BF16 activation built from the BF16 gate and up.
                    const double g = bf16_round(gate);
                    activation[r]  = bf16_round(g / (1.0 + std::exp(-g)) * bf16_round(up));
                }
                const std::vector<double> hidden =
                    nvfp4_activation(activation, down.input_divisors[expert]);
                for (int r = 0; r < kHidden; ++r) {
                    double value = 0.0;
                    for (int k = 0; k < kIntermediate; ++k) {
                        value += static_cast<double>(
                                     w2[static_cast<std::size_t>(r) * kIntermediate + k]) *
                                 hidden[k];
                    }
                    expected[t * kHidden + static_cast<std::size_t>(r)] += alpha * value;
                }
            }
        }
        return expected;
    };

    const auto run_rows = [&](int tokens) {
        GuardedDeviceBuffer d_output(static_cast<std::size_t>(kHidden) * tokens * 2);
        Tensor in(d_input.p, DType::BF16, {kHidden, tokens});
        Tensor out(d_output.data(), DType::BF16, {kHidden, tokens});
        WorkspaceArena workspace(ops::flash_next_moe_workspace_capacity_bytes(tokens));
        ops::flash_next_moe(in, weights, out, workspace, nullptr);
        cuda_synchronize();
        std::vector<double> got =
            from_device_bf16(d_output.data(), static_cast<std::size_t>(kHidden) * tokens);
        return std::pair{std::move(got), d_output.verify_guards("Flash-Next NVFP4 MoE output")};
    };

    // Per-token error, with the token's pool slots, so a fault can be tied to an expert.
    const auto report_tokens = [&](const std::string& label, const Selection& selection,
                                   const std::vector<double>& got,
                                   const std::vector<double>& want) {
        for (std::size_t t = 0; t * kHidden < got.size(); ++t) {
            const ReductionStats stats = compute_reduction_stats(
                got.data() + t * kHidden, want.data() + t * kHidden, kHidden);
            std::cout << "  " << label << " token " << t << ": relative L2 " << std::setprecision(4)
                      << stats.relative_l2 << ", max |err| " << stats.maximum_absolute_error
                      << " at row " << stats.maximum_error_index << " (" << stats.actual_at_maximum
                      << " vs " << stats.reference_at_maximum << "), slots";
            for (const auto& entry : selection[t]) { std::cout << ' ' << entry.first; }
            std::cout << '\n';
        }
    };

    int failures = 0;

    // Shared experts: token t selects slots (3t + j) % 12 with score 4 + j/4.
    Selection shared(kMaxTokens);
    for (int t = 0; t < kMaxTokens; ++t) {
        for (int j = 0; j < kTop; ++j) {
            shared[t][j] = {(3 * t + j) % static_cast<int>(kPool.size()),
                            4.0F + 0.25F * static_cast<float>(j)};
        }
    }
    upload_router(shared);
    const std::vector<double> expected = oracle(shared);
    for (const int tokens : {1, 2, 8, 16, 17}) {
        auto [got, guard_failures] = run_rows(tokens);
        failures += guard_failures;
        const std::vector<double> want(expected.begin(),
                                       expected.begin() + static_cast<std::ptrdiff_t>(got.size()));
        const std::string label = "Flash-Next NVFP4 MoE T=" + std::to_string(tokens);
        const ReductionStats stats =
            compute_reduction_stats(got.data(), want.data(), static_cast<std::int64_t>(got.size()));
        std::cout << label << ": relative L2 " << std::setprecision(4) << stats.relative_l2
                  << ", max |err| " << stats.maximum_absolute_error << " of max |ref| "
                  << stats.maximum_absolute_reference << '\n';
        if (tokens == kMaxTokens) { report_tokens(label, shared, got, want); }
        failures += verify_reduction(label, got, want, kNvfp4MoeCriterion);
    }

    // One dominant expert per token: token t gives slot t score 30 and nine other slots
    // 4 + j/4, so the others weigh about e^-26 and token t's error is slot t's.
    constexpr int kIsolated = static_cast<int>(kPool.size());
    Selection isolated(kIsolated);
    for (int t = 0; t < kIsolated; ++t) {
        isolated[t][0] = {t, 30.0F};
        for (int j = 1; j < kTop; ++j) {
            isolated[t][j] = {(t + j) % kIsolated, 4.0F + 0.25F * static_cast<float>(j)};
        }
    }
    upload_router(isolated);
    {
        const std::vector<double> want = oracle(isolated);
        auto [got, guard_failures]     = run_rows(kIsolated);
        failures += guard_failures;
        report_tokens("isolated expert", isolated, got, want);
        failures += verify_reduction("Flash-Next NVFP4 MoE isolated experts", got, want,
                                     kNvfp4MoeCriterion);
    }
    upload_router(shared);

    // Each of 8 rows alone against the same rows batched: a row's result does not depend on the
    // rows it is batched with, whichever route each call takes.
    constexpr int kBatch = 8;
    std::vector<double> single(static_cast<std::size_t>(kHidden) * kBatch);
    for (int t = 0; t < kBatch; ++t) {
        // Row t alone: its routing dimension kRoutingDim + t still selects token t's experts.
        DeviceBuffer d_row(static_cast<std::size_t>(kHidden) * 2);
        CUDA_CHECK(cudaMemcpy(d_row.p,
                              static_cast<const std::uint8_t*>(d_input.p) +
                                  static_cast<std::size_t>(t) * kHidden * 2,
                              static_cast<std::size_t>(kHidden) * 2, cudaMemcpyDeviceToDevice));
        GuardedDeviceBuffer d_out(static_cast<std::size_t>(kHidden) * 2);
        Tensor in(d_row.p, DType::BF16, {kHidden, 1});
        Tensor out(d_out.data(), DType::BF16, {kHidden, 1});
        WorkspaceArena workspace(ops::flash_next_moe_workspace_capacity_bytes(1));
        ops::flash_next_moe(in, weights, out, workspace, nullptr);
        cuda_synchronize();
        const std::vector<double> row = from_device_bf16(d_out.data(), kHidden);
        std::copy(row.begin(), row.end(),
                  single.begin() + static_cast<std::ptrdiff_t>(t) * kHidden);
    }
    GuardedDeviceBuffer d_batch(static_cast<std::size_t>(kHidden) * kBatch * 2);
    Tensor in(d_input.p, DType::BF16, {kHidden, kBatch});
    Tensor out(d_batch.data(), DType::BF16, {kHidden, kBatch});
    WorkspaceArena workspace(ops::flash_next_moe_workspace_capacity_bytes(kBatch));
    ops::flash_next_moe(in, weights, out, workspace, nullptr);
    cuda_synchronize();
    const std::vector<double> batch =
        from_device_bf16(d_batch.data(), static_cast<std::size_t>(kHidden) * kBatch);
    const ReductionStats parity = compute_reduction_stats(batch.data(), single.data(),
                                                          static_cast<std::int64_t>(batch.size()));
    std::cout << "Flash-Next NVFP4 MoE 8 rows batched vs one by one: relative L2 "
              << parity.relative_l2 << ", max |diff| " << parity.maximum_absolute_error << '\n';
    failures += verify_reduction("Flash-Next NVFP4 MoE route parity", batch, single,
                                 ReductionCriterion{1.0e-2, 0.0, 2.0e-2});
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) { return 77; }
    try {
        const int failures = run() + run_nvfp4();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Flash-Next MoE\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Flash-Next MoE: " << error.what() << '\n';
        return 1;
    }
}
