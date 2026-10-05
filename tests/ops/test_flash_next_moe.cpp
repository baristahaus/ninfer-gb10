#include "ninfer/ops/flash_next_moe.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <iostream>
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

// Row-scaled FP8 [rows, kHidden] with unit row scales; `ones`/`twos` list elements set to 1 and 2.
struct Fp8Rows {
    DeviceBuffer payload;
    Weight weight;
};

Fp8Rows fp8_rows(int rows, std::initializer_list<std::size_t> ones,
                 std::initializer_list<std::size_t> twos) {
    const std::size_t codes = static_cast<std::size_t>(rows) * kHidden;
    const std::size_t offset = (codes + 255) / 256 * 256;
    std::vector<std::uint8_t> bytes(offset + 2 * static_cast<std::size_t>(rows), 0);
    for (const std::size_t i : ones) { bytes[i] = 0x38; }  // E4M3 1.0
    for (const std::size_t i : twos) { bytes[i] = 0x40; }  // E4M3 2.0
    for (int row = 0; row < rows; ++row) {
        bytes[offset + 2 * row] = 0x80;  // BF16 1.0 = 0x3F80, little-endian
        bytes[offset + 2 * row + 1] = 0x3F;
    }
    Fp8Rows out{to_device(bytes), {}};
    Weight& w = out.weight;
    w.payload = w.qdata = out.payload.p;
    w.payload_bytes = out.payload.bytes;
    w.qtype = QType::FP8_E4M3FN_ROW_BF16;
    w.layout = QuantLayout::RowScale;
    w.n = w.shape[0] = w.padded_shape[0] = rows;
    w.k = w.shape[1] = w.padded_shape[1] = kHidden;
    w.ndim = 2;
    w.group_size = kHidden;
    w.group = kHidden;
    w.scales = static_cast<const std::uint8_t*>(out.payload.p) + offset;
    w.scale_dtype = DType::BF16;
    w.scale_ne[0] = rows;
    w.scale_nb[0] = 2;
    w.scale_nb[1] = w.scale_nb[2] = w.scale_nb[3] = static_cast<std::int64_t>(rows) * 2;
    return out;
}

void store_bf16(DeviceBuffer& storage, std::size_t element, float value) {
    const std::uint16_t bits = f32_to_bf16(value);
    storage.copy_from_host(&bits, sizeof(bits), element * sizeof(bits));
}

// `fp8` stores the router and shared gate/up as row-scaled FP8 (the one-token entry and fused
// route/shared-Down route) with the same values.
int run(bool fp8) {
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
    Fp8Rows router_fp8 = fp8_rows(kExperts, {}, {});
    Fp8Rows gate_fp8 = fp8_rows(kIntermediate, {0}, {});
    Fp8Rows up_fp8 = fp8_rows(kIntermediate, {}, {1});
    ops::FlashNextMoeWeights weights{
        .router = bf16_weight(d_router, kExperts, kHidden),
        .shared_gate = bf16_weight(d_shared_gate, kIntermediate, kHidden),
        .shared_up = bf16_weight(d_shared_up, kIntermediate, kHidden),
        .shared_down = bf16_weight(d_shared_down, kHidden, kIntermediate),
        .shared_scale = bf16_weight(d_shared_scale, 1, kHidden),
        .routed_gate_up = {.codes = d_routed_gate_up.p,
                           .qtype = QType::BF16,
                           .experts = kExperts,
                           .rows = 2 * kIntermediate,
                           .columns = kHidden},
        .routed_down = {.codes = d_routed_down.p,
                        .qtype = QType::BF16,
                        .experts = kExperts,
                        .rows = kHidden,
                        .columns = kIntermediate},
    };
    if (fp8) {
        weights.router = router_fp8.weight;
        weights.shared_gate = gate_fp8.weight;
        weights.shared_up = up_fp8.weight;
    }
    WorkspaceArena scalar_workspace(ops::flash_next_moe_workspace_capacity_bytes(1));
    ops::flash_next_moe(scalar_input, weights, scalar_output, scalar_workspace, nullptr);
    WorkspaceArena grouped_workspace(ops::flash_next_moe_workspace_capacity_bytes(kGroupedTokens));
    ops::flash_next_moe(grouped_input, weights, grouped_output, grouped_workspace, nullptr);
    // A verification-sized batch (MTP3 rows) takes the compact decode route.
    constexpr int kCompactTokens = 4;
    GuardedDeviceBuffer d_compact_output(static_cast<std::size_t>(kHidden) * kCompactTokens *
                                         sizeof(std::uint16_t));
    Tensor compact_input(d_input.p, DType::BF16, {kHidden, kCompactTokens});
    Tensor compact_output(d_compact_output.data(), DType::BF16, {kHidden, kCompactTokens});
    WorkspaceArena compact_workspace(ops::flash_next_moe_workspace_capacity_bytes(kCompactTokens));
    ops::flash_next_moe(compact_input, weights, compact_output, compact_workspace, nullptr);
    cuda_synchronize();

    const double activation = (0.5 / (1.0 + std::exp(-0.5))) * 0.5;
    std::vector<double> expected(kHidden, 0.0);
    expected[0] = activation * (1.0 + 1.0 / (1.0 + std::exp(0.5)));
    int failures = verify_pointwise(
        "Flash-Next BF16 MoE scalar", from_device_bf16(d_scalar_output.data(), kHidden),
        expected, {/*absolute*/ 4.0e-3, /*relative*/ 2.0e-2});
    std::vector<double> grouped_expected;
    grouped_expected.reserve(static_cast<std::size_t>(kHidden) * kGroupedTokens);
    for (int token = 0; token < kGroupedTokens; ++token) {
        grouped_expected.insert(grouped_expected.end(), expected.begin(), expected.end());
    }
    failures += verify_pointwise(
        "Flash-Next BF16 MoE grouped",
        from_device_bf16(d_grouped_output.data(), static_cast<std::size_t>(kHidden) * kGroupedTokens),
        grouped_expected, {/*absolute*/ 4.0e-3, /*relative*/ 2.0e-2});
    failures += verify_pointwise(
        "Flash-Next MoE compact",
        from_device_bf16(d_compact_output.data(), static_cast<std::size_t>(kHidden) * kCompactTokens),
        std::vector<double>(grouped_expected.begin(),
                            grouped_expected.begin() + static_cast<std::ptrdiff_t>(kHidden) * kCompactTokens),
        {/*absolute*/ 4.0e-3, /*relative*/ 2.0e-2});
    failures += d_compact_output.verify_guards("Flash-Next MoE compact output");
    failures += d_scalar_output.verify_guards("Flash-Next BF16 MoE scalar output");
    failures += d_grouped_output.verify_guards("Flash-Next BF16 MoE grouped output");
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) { return 77; }
    try {
        const int failures = run(false) + run(true);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Flash-Next MoE\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Flash-Next MoE: " << error.what() << '\n';
        return 1;
    }
}
