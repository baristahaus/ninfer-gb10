#include "ninfer/ops/flash_next_moe.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <string>
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

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) { return 77; }
    try {
        const int failures = run();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Flash-Next MoE\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Flash-Next MoE: " << error.what() << '\n';
        return 1;
    }
}
