#include "ninfer/ops/argmax.h"
#include "core/device.h"
#include <algorithm>
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::vector<std::int32_t> argmax_oracle(const std::vector<std::uint16_t>& logits,
                                        std::int32_t physical_rows, std::int32_t tokens,
                                        std::int32_t valid_rows) {
    std::vector<std::int32_t> expected(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        std::int32_t best      = 0;
        float best_value       = bf16_to_f32(logits[base]);
        for (std::int32_t row = 1; row < valid_rows; ++row) {
            const float value = bf16_to_f32(logits[base + row]);
            if (value > best_value) {
                best       = row;
                best_value = value;
            }
        }
        expected[static_cast<std::size_t>(token)] = best;
    }
    return expected;
}

std::vector<std::uint16_t> make_logits(std::int32_t physical_rows, std::int32_t tokens,
                                       std::int32_t valid_rows) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        for (std::int32_t row = 0; row < physical_rows; ++row) {
            const std::uint32_t mixed = static_cast<std::uint32_t>(row) * 1664525u +
                                        static_cast<std::uint32_t>(token + 1) * 1013904223u;
            const float value  = -24.0f + static_cast<float>(mixed % 3072u) * (1.0f / 256.0f);
            logits[base + row] = f32_to_bf16(value);
        }

        std::int32_t first  = 17 + token * 7919;
        std::int32_t second = valid_rows - 1 - token * 65537;
        first %= valid_rows;
        second %= valid_rows;
        if (second < 0) { second += valid_rows; }
        if (first == second) { second = (second + 1) % valid_rows; }
        if (second < first) {
            const std::int32_t temporary = first;
            first                        = second;
            second                       = temporary;
        }
        logits[base + first]  = f32_to_bf16(token % 3 == 0 ? -0.5f : 32.0f + static_cast<float>(token));
        logits[base + second] = logits[base + first];

        if (token == 0) {
            std::fill(logits.begin() + base, logits.begin() + base + valid_rows, f32_to_bf16(-4.0f));
        }
        if (valid_rows < physical_rows) {
            logits[base + valid_rows]        = f32_to_bf16(32768.0f);
            logits[base + physical_rows - 1] = f32_to_bf16(65536.0f);
        }
    }
    return logits;
}

int run_case(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t tokens) {
    auto logits   = make_logits(physical_rows, tokens, valid_rows);
    auto expected = argmax_oracle(logits, physical_rows, tokens, valid_rows);

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_output(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    device_logits.copy_from_host(logits.data(), logits.size() * sizeof(std::uint16_t));
    device_output.fill(0xcd);

    Tensor logits_tensor(device_logits.data(), DType::BF16, {physical_rows, tokens});
    Tensor output_tensor(device_output.data(), DType::I32, {tokens});
    ops::argmax(logits_tensor, output_tensor, valid_rows, nullptr);
    cuda_synchronize();

    if (physical_rows == 248320 && tokens == 128) {
        cudaStream_t stream;
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        ops::argmax(logits_tensor, output_tensor, valid_rows, stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int t = 0; t < tokens; ++t)
            logits[static_cast<std::size_t>(t) * physical_rows + valid_rows - 1] = f32_to_bf16(8192.0f);
        CUDA_CHECK(cudaMemcpyAsync(device_logits.data(), logits.data(), device_logits.bytes(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(stream));
        expected = argmax_oracle(logits, physical_rows, tokens, valid_rows);
    }

    const auto actual =
        from_device<std::int32_t>(device_output.data(), static_cast<std::size_t>(tokens));
    const auto logits_after = from_device<std::uint16_t>(device_logits.data(), logits.size());
    const std::string label = "argmax rows=" + std::to_string(physical_rows) +
                              " valid=" + std::to_string(valid_rows) +
                              " T=" + std::to_string(tokens);

    int failures = 0;
    failures += verify_exact(label.c_str(), actual, expected);
    failures += verify_exact((label + " preserves logits").c_str(), logits_after, logits);
    failures += device_logits.verify_guards((label + " logits").c_str());
    failures += device_output.verify_guards((label + " output").c_str());
    return failures;
}

// Shortlist rerank: each 512-row tile of the approximate logits contributes its two best rows
// (lower row on ties), mapped through id_map; the exact head rescores those ids and the best
// exact score wins (lower id on ties). The FP64 reference evaluates the represented head rows.
int run_shortlist_case(bool fp8_head) {
    constexpr std::int32_t kHidden        = 2560;
    constexpr std::int32_t kHeadRows      = 8192;
    constexpr std::int32_t kPhysicalRows  = 1536;
    constexpr std::int32_t kShortlistRows = 1500;
    constexpr std::int32_t kTokens        = 3;
    constexpr std::int32_t kTile          = 512;
    constexpr std::int32_t kCandidateRows = (kShortlistRows + kTile - 1) / kTile * 2;
    const std::string label = std::string("shortlist exact argmax ") + (fp8_head ? "FP8" : "BF16");

    std::vector<std::int32_t> all_rows(kHeadRows);
    for (std::int32_t row = 0; row < kHeadRows; ++row) { all_rows[row] = row; }
    const auto fp8 = quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16,
                                                             kHeadRows, kHidden, 931U);
    std::vector<float> head(static_cast<std::size_t>(kHeadRows) * kHidden);
    if (fp8_head) {
        head = quantized_weight::materialize_rows_fp32(fp8, all_rows);
    } else {
        fill_uniform(head, 932U, -0.5F, 0.5F);
        round_to_bf16(head);
    }
    std::vector<float> hidden(static_cast<std::size_t>(kHidden) * kTokens);
    fill_uniform(hidden, 933U, -1.0F, 1.0F);
    round_to_bf16(hidden);
    std::vector<float> approximate(static_cast<std::size_t>(kPhysicalRows) * kTokens);
    fill_uniform(approximate, 934U, -8.0F, 8.0F);
    round_to_bf16(approximate);
    std::vector<std::int32_t> id_map(kPhysicalRows);
    for (std::int32_t row = 0; row < kPhysicalRows; ++row) {
        id_map[row] = (row * 7 + 3) % kHeadRows;
    }

    std::vector<std::int32_t> expected_ids(static_cast<std::size_t>(kCandidateRows) * kTokens);
    std::vector<double> expected_scores(expected_ids.size());
    std::vector<std::int32_t> expected_out(kTokens);
    const auto better = [](double value, std::int32_t index, double best, std::int32_t best_index) {
        return value > best || (value == best && index < best_index);
    };
    for (std::int32_t token = 0; token < kTokens; ++token) {
        const float* logits = approximate.data() + static_cast<std::size_t>(token) * kPhysicalRows;
        for (std::int32_t tile = 0; tile * kTile < kShortlistRows; ++tile) {
            std::int32_t first = -1, second = -1;
            for (std::int32_t row = tile * kTile;
                 row < std::min(kShortlistRows, (tile + 1) * kTile); ++row) {
                if (first < 0 || better(logits[row], row, logits[first], first)) {
                    second = first;
                    first  = row;
                } else if (second < 0 || better(logits[row], row, logits[second], second)) {
                    second = row;
                }
            }
            const std::size_t base = static_cast<std::size_t>(token) * kCandidateRows + tile * 2;
            expected_ids[base]     = id_map[first];
            expected_ids[base + 1] = id_map[second];
        }
        double best_score      = -1.0e300;
        std::int32_t best_id   = 0;
        for (std::int32_t candidate = 0; candidate < kCandidateRows; ++candidate) {
            const std::size_t index =
                static_cast<std::size_t>(token) * kCandidateRows + candidate;
            const std::int32_t id = expected_ids[index];
            double score          = 0.0;
            for (std::int32_t k = 0; k < kHidden; ++k) {
                score += double(head[static_cast<std::size_t>(id) * kHidden + k]) *
                         hidden[static_cast<std::size_t>(token) * kHidden + k];
            }
            expected_scores[index] = score;
            if (candidate == 0 || better(score, id, best_score, best_id)) {
                best_score = score;
                best_id    = id;
            }
        }
        expected_out[token] = best_id;
    }

    std::vector<std::uint16_t> head_bits(head.size());
    for (std::size_t i = 0; i < head.size(); ++i) { head_bits[i] = f32_to_bf16(head[i]); }
    DeviceBuffer d_head = fp8_head ? to_device(fp8.payload) : to_device(head_bits);
    Weight exact{};
    if (fp8_head) {
        exact = fp8.device_weight(d_head.p);
    } else {
        exact.payload = exact.qdata = d_head.p;
        exact.payload_bytes         = d_head.bytes;
        exact.qtype                 = QType::BF16;
        exact.layout                = QuantLayout::Contiguous;
        exact.n = exact.shape[0] = exact.padded_shape[0] = kHeadRows;
        exact.k = exact.shape[1] = exact.padded_shape[1] = kHidden;
        exact.ndim                                       = 2;
    }
    DeviceBuffer d_hidden      = to_device_bf16(hidden);
    DeviceBuffer d_approximate = to_device_bf16(approximate);
    DeviceBuffer d_id_map      = to_device(id_map);
    GuardedDeviceBuffer d_ids(expected_ids.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_scores(expected_scores.size() * sizeof(float));
    GuardedDeviceBuffer d_out(expected_out.size() * sizeof(std::int32_t));
    Tensor hidden_tensor(d_hidden.p, DType::BF16, {kHidden, kTokens});
    Tensor approximate_tensor(d_approximate.p, DType::BF16, {kPhysicalRows, kTokens});
    Tensor ids_tensor(d_ids.data(), DType::I32, {kCandidateRows, kTokens});
    Tensor scores_tensor(d_scores.data(), DType::FP32, {kCandidateRows, kTokens});
    Tensor out_tensor(d_out.data(), DType::I32, {kTokens});
    ops::shortlist_exact_argmax(hidden_tensor, approximate_tensor, kShortlistRows, exact,
                                static_cast<const std::int32_t*>(d_id_map.p), ids_tensor,
                                scores_tensor, out_tensor, nullptr);
    cuda_synchronize();

    int failures = 0;
    failures += verify_exact((label + " candidates").c_str(),
                             from_device<std::int32_t>(d_ids.data(), expected_ids.size()),
                             expected_ids);
    const auto scores = from_device<float>(d_scores.data(), expected_scores.size());
    const std::vector<double> actual_scores(scores.begin(), scores.end());
    failures += verify_pointwise(label + " scores", actual_scores, expected_scores,
                                 {/*absolute*/ 1.0e-3, /*relative*/ 1.0e-4});
    failures += verify_exact((label + " selection").c_str(),
                             from_device<std::int32_t>(d_out.data(), expected_out.size()),
                             expected_out);
    failures += d_ids.verify_guards((label + " candidates").c_str());
    failures += d_scores.verify_guards((label + " scores").c_str());
    failures += d_out.verify_guards((label + " selection").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case(248320, 248077, 1);
    failures += run_case(248320, 248077, 2);
    failures += run_case(248320, 248077, 8);
    failures += run_case(248320, 248077, 9);
    failures += run_case(248320, 248077, 64);
    failures += run_case(248320, 248077, 15);
    failures += run_case(248320, 248077, 128);
    failures += run_case(131072, 131072, 1);
    failures += run_case(131072, 131072, 15);
    failures += run_case(131072, 131072, 120);
    failures += run_shortlist_case(false);
    failures += run_shortlist_case(true);
    std::cout << (failures ? "FAIL" : "OK") << " argmax\n";
    return failures ? 1 : 0;
}
