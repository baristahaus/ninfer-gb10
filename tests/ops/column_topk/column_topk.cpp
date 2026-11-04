#include "ops/suite.h"
#include "ninfer/ops/column_topk.h"
#include "cuda_fixture.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::uint16_t to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>((bits + 0x8000u) >> 16);
}

float bf16_value(float value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(to_bf16(value)) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// The contract declares ids/scores [K,T,B] with the rank axis fastest: column c's ranks are
// contiguous at rank + K*c. A column-major writer interleaves the columns' ranks instead.
int run_case(const char* label, std::int32_t V, std::int32_t T, std::int32_t B) {
    constexpr std::int32_t K = 16;
    const std::int32_t columns = T * B;
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(columns) * V);
    for (std::int32_t c = 0; c < columns; ++c) {
        for (std::int32_t v = 0; v < V; ++v) {
            // Descending with the row index; each column is offset so ties never cross columns.
            logits[static_cast<std::size_t>(c) * V + v] = to_bf16(static_cast<float>(V - v) + 0.25f * c);
        }
    }

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_ids(static_cast<std::size_t>(K) * columns * sizeof(std::int32_t));
    GuardedDeviceBuffer device_scores(static_cast<std::size_t>(K) * columns * sizeof(float));
    device_logits.copy_from_host(logits.data(), device_logits.bytes());

    Tensor logits_tensor(device_logits.span(), DType::BF16, {V, T, B});
    Tensor ids_tensor(device_ids.span(), DType::I32, {K, T, B});
    Tensor scores_tensor(device_scores.span(), DType::FP32, {K, T, B});
    ops::column_topk(logits_tensor, ids_tensor, scores_tensor,
                     cuda_environment().default_execution);
    cuda_synchronize();

    const auto ids    = from_device<std::int32_t>(device_ids.data(), static_cast<std::size_t>(K) * columns);
    const auto scores = from_device<float>(device_scores.data(), static_cast<std::size_t>(K) * columns);
    int failures      = 0;
    for (std::int32_t c = 0; c < columns; ++c) {
        for (std::int32_t r = 0; r < K; ++r) {
            const std::size_t at = static_cast<std::size_t>(r) + static_cast<std::size_t>(K) * c;
            if (ids[at] != r) {
                std::cerr << label << ": ids[" << at << "]=" << ids[at] << " expected " << r << '\n';
                ++failures;
            }
            // The op returns the represented BF16 value, not the pre-rounding float.
            const float expected = bf16_value(static_cast<float>(V - r) + 0.25f * c);
            if (std::fabs(scores[at] - expected) > 1e-3f) {
                std::cerr << label << ": scores[" << at << "]=" << scores[at] << " expected "
                          << expected << '\n';
                ++failures;
            }
        }
    }
    failures += device_logits.verify_guards("column_topk logits");
    failures += device_ids.verify_guards("column_topk ids");
    failures += device_scores.verify_guards("column_topk scores");
    return failures;
}

} // namespace

void add_column_topk_cases(ninfer::test::OpSuite& suite) {
    suite.add("column_topk/v64/t3/b1", [] { return run_case("column_topk [64,3,1]", 64, 3, 1); });
    suite.add("column_topk/v257/t4/b2", [] { return run_case("column_topk [257,4,2]", 257, 4, 2); });
}
