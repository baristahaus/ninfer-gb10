// The dense EXL3 route of the Linear Op (SP-7 contract sections 4 and 7, dense step).
//
// Oracle: the test-owned quantized-weight fixture composes the logical FP64 weight
// diag(suh) Hn_k W_hat Hn_n diag(svh) through the exact codec oracle's inner decode and exact
// butterfly Hadamard blocks; the Linear FP64 GEMM oracle consumes its FP32 materialization.
// Criterion: the A16 reduction tolerance shared by every Linear format. The kernel never
// materializes the inner weight (section 7); the codec itself is pinned by ninfer_exl3_codec_test.

#include "ops/linear/linear_test_common.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <array>
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

} // namespace

int main() {
    if (!cuda_available()) {
        std::cout << "exl3 dense linear: no CUDA device, skipped\n";
        return 0;
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
