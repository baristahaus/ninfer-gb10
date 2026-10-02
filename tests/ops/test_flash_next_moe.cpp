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

// The production NVFP4 W4A4 routes against the FP64 MoE on the decoded expert weights: one row
// (the per-assignment route), 2..16 rows (the expert-grouped decode tile) and 17 rows (the
// grouped prefill tile). The two NVFP4 activation quantizations are the routes' arithmetic
// profile, so the criterion is distribution-level, as for the A4 linear Ops. Routing is exact:
// each token's 10 experts score distinct BF16 values on a dedicated input dimension, and tokens
// share experts from a pool of 12, as decode rows do.
constexpr int kRoutingDim = 2544;
constexpr std::array<int, 12> kPool{3, 41, 88, 130, 177, 211, 260, 305, 349, 402, 455, 509};
constexpr ReductionCriterion kNvfp4MoeCriterion{0.25, 0.0, 0.5};

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
    constexpr int kMaxTokens = 17;
    // Token t selects pool[(3t + j) % 12] for j in [0,10) with router score 4 + j/4 (exact BF16).
    std::vector<std::array<int, kTop>> selected(kMaxTokens);
    std::vector<std::uint16_t> router(static_cast<std::size_t>(kExperts) * kHidden,
                                      f32_to_bf16(0.0F));
    for (int t = 0; t < kMaxTokens; ++t) {
        for (int j = 0; j < kTop; ++j) {
            const int expert = kPool[(3 * t + j) % kPool.size()];
            selected[t][j]   = expert;
            router[static_cast<std::size_t>(expert) * kHidden + kRoutingDim + t] =
                f32_to_bf16(4.0F + 0.25F * static_cast<float>(j));
        }
    }
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
    DeviceBuffer d_router        = to_device(router);
    DeviceBuffer d_input         = to_device_bf16(input);
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

    // FP64 oracle: softmax over the selected scores, SiLU(gate) * up, down projection.
    std::vector<double> expected(static_cast<std::size_t>(kHidden) * kMaxTokens, 0.0);
    for (std::size_t slot = 0; slot < kPool.size(); ++slot) {
        const int expert = kPool[slot];
        std::vector<int> users;
        for (int t = 0; t < kMaxTokens; ++t) {
            if (std::find(selected[t].begin(), selected[t].end(), expert) != selected[t].end()) {
                users.push_back(t);
            }
        }
        if (users.empty()) { continue; }
        std::vector<std::int32_t> all_rows(2 * kIntermediate);
        for (int r = 0; r < 2 * kIntermediate; ++r) { all_rows[r] = r; }
        const std::vector<float> w1 = quantized_weight::materialize_rows_fp32(
            gate_up.experts[slot], std::span<const std::int32_t>(all_rows));
        std::vector<std::int32_t> down_rows(kHidden);
        for (int r = 0; r < kHidden; ++r) { down_rows[r] = r; }
        const std::vector<float> w2 = quantized_weight::materialize_rows_fp32(
            down.experts[slot], std::span<const std::int32_t>(down_rows));
        for (const int t : users) {
            double maximum = 0.0;
            for (int j = 0; j < kTop; ++j) { maximum = std::max(maximum, 4.0 + 0.25 * j); }
            double denominator = 0.0;
            for (int j = 0; j < kTop; ++j) { denominator += std::exp(4.0 + 0.25 * j - maximum); }
            const int rank = static_cast<int>(
                std::find(selected[t].begin(), selected[t].end(), expert) - selected[t].begin());
            const double alpha = std::exp(4.0 + 0.25 * rank - maximum) / denominator;
            const float* x     = input.data() + static_cast<std::size_t>(t) * kHidden;
            std::vector<double> hidden(kIntermediate);
            for (int r = 0; r < kIntermediate; ++r) {
                double gate = 0.0, up = 0.0;
                for (int k = 0; k < kHidden; ++k) {
                    gate +=
                        static_cast<double>(w1[static_cast<std::size_t>(r) * kHidden + k]) * x[k];
                    up += static_cast<double>(
                              w1[static_cast<std::size_t>(r + kIntermediate) * kHidden + k]) *
                          x[k];
                }
                hidden[r] = gate / (1.0 + std::exp(-gate)) * up;
            }
            for (int r = 0; r < kHidden; ++r) {
                double value = 0.0;
                for (int k = 0; k < kIntermediate; ++k) {
                    value +=
                        static_cast<double>(w2[static_cast<std::size_t>(r) * kIntermediate + k]) *
                        hidden[k];
                }
                expected[static_cast<std::size_t>(t) * kHidden + r] += alpha * value;
            }
        }
    }

    int failures = 0;
    std::vector<double> one_row_outputs;
    for (const int tokens : {1, 2, 8, 16, 17}) {
        GuardedDeviceBuffer d_output(static_cast<std::size_t>(kHidden) * tokens * 2);
        Tensor in(d_input.p, DType::BF16, {kHidden, tokens});
        Tensor out(d_output.data(), DType::BF16, {kHidden, tokens});
        WorkspaceArena workspace(ops::flash_next_moe_workspace_capacity_bytes(tokens));
        ops::flash_next_moe(in, weights, out, workspace, nullptr);
        cuda_synchronize();
        const std::vector<double> got =
            from_device_bf16(d_output.data(), static_cast<std::size_t>(kHidden) * tokens);
        const std::vector<double> want(expected.begin(),
                                       expected.begin() + static_cast<std::ptrdiff_t>(got.size()));
        const std::string label = "Flash-Next NVFP4 MoE T=" + std::to_string(tokens);
        const ReductionStats stats =
            compute_reduction_stats(got.data(), want.data(), static_cast<std::int64_t>(got.size()));
        std::cout << label << ": relative L2 " << std::setprecision(4) << stats.relative_l2
                  << ", max |err| " << stats.maximum_absolute_error << " of max |ref| "
                  << stats.maximum_absolute_reference << '\n';
        failures += verify_reduction(label, got, want, kNvfp4MoeCriterion);
        failures += d_output.verify_guards(label);
    }

    // Each of 8 rows alone (the per-assignment route) against the same rows batched (the
    // expert-grouped route): both production routes, same quantized arithmetic per (row, expert).
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
