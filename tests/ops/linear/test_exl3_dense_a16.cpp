// The dense EXL3 route of the Linear Op (SP-7 contract sections 4 and 7, dense step).
//
// Oracle: the test-owned quantized-weight fixture composes the logical FP64 weight
// diag(suh) Hn_k W_hat Hn_n diag(svh) through the exact codec oracle's inner decode and exact
// butterfly Hadamard blocks; the Linear FP64 GEMM oracle consumes its FP32 materialization.
// Criterion: the A16 reduction tolerance shared by every Linear format. The kernel never
// materializes the inner weight (section 7); the codec itself is pinned by ninfer_exl3_codec_test.

#include "ops/linear/linear_test_common.h"

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ops/quantized_weight.h"

#include "core/device.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::linear;
using quantized_weight::make_exl3_dense_weight;
using Codebook = ninfer::test::exl3_ref::Codebook;

int admission_refusals() {
    int failures = 0;
    // An unregistered problem is refused at capacity and admission alike.
    try {
        (void)ops::linear_workspace_capacity_bytes(QType::EXL3, 5120, 5120,
                                                   ops::LinearPolicy::A16Only, 1, 8);
        std::cerr << "exl3 dense: unregistered problem accepted at capacity\n";
        ++failures;
    } catch (const std::exception&) {
    }
    // The token interval outside the registered decode shapes is refused.
    try {
        (void)ops::linear_workspace_capacity_bytes(QType::EXL3, 1024, 5120,
                                                   ops::LinearPolicy::A16Only, 9, 9);
        std::cerr << "exl3 dense: T = 9 accepted at capacity\n";
        ++failures;
    } catch (const std::exception&) {
    }
    // A 3INST bank has no qualified route (contract section 7).
    {
        auto packed = make_exl3_dense_weight(1024, 5120, 11U, 8, Codebook::Inst3);
        DeviceBuffer device_weight(packed.payload.size());
        device_weight.copy_from_host(packed.payload.data(), device_weight.bytes);
        DeviceBuffer activation(5120 * 8 * 2);
        DeviceBuffer output(1024 * 8 * 2);
        Tensor x(activation.p, DType::BF16, {5120, 8});
        Tensor out(output.p, DType::BF16, {1024, 8});
        DeviceArena workspace(1 << 20);
        try {
            ops::linear(x, packed.device_weight(device_weight.p), out,
                        ops::LinearPolicy::A16Only, workspace, nullptr);
            std::cerr << "exl3 dense: 3INST bank accepted\n";
            ++failures;
        } catch (const std::exception&) {
        }
    }
    // An EXL3 expert bank is not a dense linear: Linear refuses it outright.
    {
        Weight bank{};
        bank.qtype = QType::EXL3;
        bank.layout = QuantLayout::Exl3ExpertBank;
        bank.n = bank.shape[0] = bank.padded_shape[0] = 512;
        bank.k = bank.shape[1] = bank.padded_shape[1] = 2048;
        bank.ndim = 2;
        DeviceBuffer activation(2048 * 2);
        DeviceBuffer output(512 * 2);
        Tensor x(activation.p, DType::BF16, {2048, 1});
        Tensor out(output.p, DType::BF16, {512, 1});
        try {
            ops::linear(x, bank, out, nullptr);
            std::cerr << "exl3 dense: expert bank accepted by Linear\n";
            ++failures;
        } catch (const std::exception&) {
        }
    }
    return failures;
}

int exl3_dense_a16_conformance() {
    int failures = 0;

    // Full-output FP64 oracle across every registered decode token shape, with graph capture
    // on the boundaries the schedule keeps fixed.
    constexpr std::array kFull{
        Invocation{1}, Invocation{2, CallForm::Policy, ops::LinearPolicy::A16Only, true},
        Invocation{3}, Invocation{4}, Invocation{5}, Invocation{6},
        Invocation{7}, Invocation{8, CallForm::Policy, ops::LinearPolicy::A16Only, true},
    };
    failures += run_shape("EXL3_DENSE_A16", ActivationCompute::A16,
                          [](std::int32_t n, std::int32_t k, std::uint32_t seed) {
                              return make_exl3_dense_weight(n, k, seed, 8, Codebook::Mcg);
                          },
                          {1024, 5120, 211U, Comparison::Full, true, kFull});

    // The mul1 codebook at a half-integer rate, including graph capture.
    constexpr std::array kMul1Half{
        Invocation{1}, Invocation{4, CallForm::Policy, ops::LinearPolicy::A16Only, true},
        Invocation{8},
    };
    failures += run_shape("EXL3_DENSE_A16", ActivationCompute::A16,
                          [](std::int32_t n, std::int32_t k, std::uint32_t seed) {
                              return make_exl3_dense_weight(n, k, seed, 3, Codebook::Mul1);
                          },
                          {1024, 5120, 307U, Comparison::Full, true, kMul1Half});

    // Every remaining registered problem, sampled rows and columns at the decode shapes.
    constexpr std::array kSampled{
        Invocation{1}, Invocation{4}, Invocation{8},
    };
    const test::linear::WeightGenerator mcg =
        [](std::int32_t n, std::int32_t k, std::uint32_t seed) {
            return make_exl3_dense_weight(n, k, seed, 4, Codebook::Mcg);
        };
    failures += run_shape("EXL3_DENSE_A16", ActivationCompute::A16, mcg,
                          {5120, 6144, 401U, Comparison::Sampled, false, kSampled});
    failures += run_shape("EXL3_DENSE_A16", ActivationCompute::A16, mcg,
                          {6144, 5120, 409U, Comparison::Sampled, false, kSampled});
    failures += run_shape("EXL3_DENSE_A16", ActivationCompute::A16, mcg,
                          {10240, 5120, 503U, Comparison::Sampled, false, kSampled});
    failures += run_shape("EXL3_DENSE_A16", ActivationCompute::A16, mcg,
                          {17408, 5120, 601U, Comparison::Sampled, false, kSampled});
    failures += run_shape("EXL3_DENSE_A16", ActivationCompute::A16, mcg,
                          {5120, 17408, 701U, Comparison::Sampled, false, kSampled});

    // Capacity is monotone in the token interval and constant across the registered decode
    // shapes, so any declared interval inside 1..8 covers its points by construction; the
    // envelope helper applies only to formats registered at every token count.
    return failures;
}

// Real-converted-tensor check: the container (written by the converter's real-check script)
// carries one exl3_dense_v1 tensor from a real 27B quantization and the composed FP32 expected
// weight as a raw_bytes resource. The kernel runs on random activations and is compared
// against the FP64 GEMM of that matrix under the A16 criterion.
int real_artifact_check(const char* path) {
    ninfer::artifact::Reader reader(path);
    const auto tensor = reader.find("exl3_dense");
    const auto& geometry = reader.geometry(tensor);
    reader.validate_object(tensor);
    if (geometry.layout != QuantLayout::Exl3Dense || geometry.shape.size() != 2) {
        std::cerr << "real check: container tensor is not a dense EXL3 linear\n";
        return 1;
    }
    const auto bytes = reader.read_object(tensor);
    ninfer::DeviceBuffer device_weight(bytes.size());
    device_weight.copy_from_host(bytes.data(), bytes.size());

    Weight w{};
    w.qtype            = QType::EXL3;
    w.layout           = QuantLayout::Exl3Dense;
    w.ndim             = 2;
    w.n = w.shape[0] = w.padded_shape[0] = static_cast<std::int32_t>(geometry.shape[0]);
    w.k = w.shape[1] = w.padded_shape[1] = static_cast<std::int32_t>(geometry.shape[1]);
    w.payload        = device_weight.p;
    w.payload_bytes  = geometry.bytes;
    w.exl3_trellis_offset = geometry.exl3.trellis_offset;
    w.exl3_suh_offset      = geometry.exl3.suh_offset;
    w.exl3_svh_offset      = geometry.exl3.svh_offset;
    w.exl3_codebook        = geometry.exl3.codebook;
    w.exl3_experts         = geometry.exl3.experts;
    w.exl3_rows_per_expert = geometry.exl3.rows_per_expert;

    const auto expected_bytes = reader.read_object(reader.find("expected"));
    const std::int32_t n = w.n, k = w.k;
    if (expected_bytes.size() != static_cast<std::size_t>(n) * k * 4) {
        std::cerr << "real check: expected weight resource has the wrong size\n";
        return 1;
    }
    std::vector<float> expected(n * k);
    std::memcpy(expected.data(), expected_bytes.data(), expected_bytes.size());

    constexpr std::int32_t t = 4;
    std::vector<std::uint16_t> activation_bits(static_cast<std::size_t>(k) * t);
    for (std::size_t i = 0; i < activation_bits.size(); ++i) {
        const float value =
            static_cast<float>(static_cast<int>((i * 37 + 11) % 251) - 125) * (1.0F / 256.0F);
        activation_bits[i] = ninfer::test::f32_to_bf16(value);
    }
    ninfer::DeviceBuffer device_activation(activation_bits.size() * 2);
    device_activation.copy_from_host(activation_bits.data(), device_activation.bytes);
    ninfer::test::GuardedDeviceBuffer output(static_cast<std::size_t>(n) * t * 2);
    Tensor x(device_activation.p, DType::BF16, {k, t});
    Tensor out(output.data(), DType::BF16, {n, t});
    const std::size_t capacity = ops::linear_workspace_capacity_bytes(
        QType::EXL3, n, k, ops::LinearPolicy::A16Only, t, t);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    ops::linear(x, w, out, ops::LinearPolicy::A16Only, workspace, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    int guard_failures = output.verify_guards("real check");
    if (guard_failures != 0) {
        std::cerr << "real check: output guards were overwritten\n";
        return 1;
    }

    // FP64 oracle over the composed matrix and the represented activations.
    std::vector<float> activation_fp32(static_cast<std::size_t>(k) * t);
    for (std::size_t i = 0; i < activation_bits.size(); ++i) {
        activation_fp32[i] = bf16_to_f32(activation_bits[i]);
    }
    std::vector<double> reference(static_cast<std::size_t>(n) * t);
    cpu_linear_gemm_fp64(expected.data(), activation_fp32.data(), reference.data(), n, k, t);

    std::vector<std::uint16_t> host_out(static_cast<std::size_t>(n) * t);
    CUDA_CHECK(cudaMemcpy(host_out.data(), output.data(), host_out.size() * 2,
                          cudaMemcpyDeviceToHost));
    double diff_l2 = 0.0, ref_l2 = 0.0, worst = 0.0;
    int outliers = 0;
    for (std::size_t i = 0; i < host_out.size(); ++i) {
        const double actual = bf16_to_f32(host_out[i]);
        const double delta = actual - reference[i];
        diff_l2 += delta * delta;
        ref_l2 += reference[i] * reference[i];
        worst = std::max(worst, std::fabs(delta));
        if (std::fabs(delta) > 1.0) {
            ++outliers;
            if (outliers <= 8) {
                std::cout << "outlier index " << i << " (n=" << i % n << ", t=" << i / n
                          << ") got " << actual << " expect " << reference[i] << "\n";
            }
        }
    }
    if (outliers != 0) {
        std::cerr << "real check: " << outliers << " outputs off by more than 1.0\n";
        return 1;
    }
    const double relative = std::sqrt(diff_l2) / std::sqrt(std::max(ref_l2, 1e-30));
    std::cout << "real check: relative L2 " << relative << " max abs " << worst << "\n";
    if (!(relative <= 2.0 / 256.0)) {
        std::cerr << "real check: relative L2 exceeds the A16 criterion\n";
        return 1;
    }
    std::cout << "exl3 dense real-tensor check passed\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (!cuda_available()) {
        std::cout << "exl3 dense linear: no CUDA device, skipped\n";
        return 0;
    }
    if (argc == 2) {
        try {
            return real_artifact_check(argv[1]);
        } catch (const std::exception& error) {
            std::cerr << "exl3 dense real check: unexpected " << error.what() << "\n";
            return 1;
        }
    }
    int failures = 0;
    try {
        failures += admission_refusals();
        failures += exl3_dense_a16_conformance();
    } catch (const std::exception& error) {
        std::cerr << "exl3 dense linear: unexpected " << error.what() << "\n";
        return 1;
    }
    if (failures != 0) {
        std::cerr << "exl3 dense linear: " << failures << " failures\n";
        return 1;
    }
    std::cout << "exl3 dense linear checks passed\n";
    return 0;
}
