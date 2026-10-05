// Shortlist re-ranking for the optimized proposal head: per 512-row tile the two best approximate
// logits become candidates, and the exact output-head score picks the token. Qualified against an
// FP64 oracle for BF16 and row-scaled FP8 heads.
#include "ninfer/ops/argmax.h"
#include "ops/op_tester.h"

#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr int kHidden = 2560, kVocab = 6000, kShortlist = 1300, kTokens = 3;
constexpr int kTile = 512, kPerTile = 2;
constexpr int kCandidates = (kShortlist + kTile - 1) / kTile * kPerTile;

double fp8_value(std::uint8_t code) {
    __nv_fp8_e4m3 value;
    value.__x = code;
    return static_cast<double>(static_cast<float>(value));
}

int run_case(bool fp8) {
    std::mt19937 rng(fp8 ? 77 : 41);
    std::vector<float> hidden(kHidden * kTokens), approximate(kShortlist * kTokens);
    fill_uniform(hidden, fp8 ? 3 : 5, -1.0F, 1.0F);
    round_to_bf16(hidden);
    // Distinct approximate logits so tile candidates are unambiguous.
    std::vector<int> order(kShortlist * kTokens);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    for (std::size_t i = 0; i < order.size(); ++i) approximate[order[i]] = 0.01F * static_cast<float>(i % 256) + 0.0001F * static_cast<float>(i / 256);
    round_to_bf16(approximate);
    std::vector<int> id_map(kVocab);
    std::iota(id_map.begin(), id_map.end(), 0);
    std::shuffle(id_map.begin(), id_map.end(), rng);
    id_map.resize(kShortlist);

    std::vector<double> head(static_cast<std::size_t>(kVocab) * kHidden);
    std::vector<std::uint16_t> bf16_rows;
    std::vector<std::uint8_t> codes;
    std::vector<std::uint16_t> scales;
    if (fp8) {
        codes.resize(head.size());
        scales.resize(kVocab);
        std::uniform_int_distribution<int> byte(0, 255);
        for (std::size_t i = 0; i < codes.size(); ++i) {
            std::uint8_t c;
            do { c = static_cast<std::uint8_t>(byte(rng)); } while ((c & 0x7F) == 0x7F);
            codes[i] = c;
        }
        std::uniform_real_distribution<float> scale(0.0005F, 0.002F);
        for (int r = 0; r < kVocab; ++r) {
            scales[r] = f32_to_bf16(scale(rng));
            for (int k = 0; k < kHidden; ++k)
                head[static_cast<std::size_t>(r) * kHidden + k] =
                    fp8_value(codes[static_cast<std::size_t>(r) * kHidden + k]) * bf16_to_f32(scales[r]);
        }
    } else {
        std::vector<float> values(head.size());
        fill_uniform(values, 9, -0.05F, 0.05F);
        bf16_rows.resize(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            bf16_rows[i] = f32_to_bf16(values[i]);
            head[i]      = bf16_to_f32(bf16_rows[i]);
        }
    }

    // Oracle: per tile the best and second-best approximate rows (value, then lower row), then the
    // largest exact FP64 score.
    std::vector<int> expected(kTokens);
    std::vector<double> best_score(kTokens);
    std::vector<std::vector<int>> candidates(kTokens);
    for (int t = 0; t < kTokens; ++t) {
        for (int tile = 0; tile * kTile < kShortlist; ++tile) {
            std::vector<int> rows;
            for (int r = tile * kTile; r < std::min(kShortlist, (tile + 1) * kTile); ++r) rows.push_back(r);
            std::sort(rows.begin(), rows.end(), [&](int a, int b) {
                const float va = approximate[t * kShortlist + a], vb = approximate[t * kShortlist + b];
                return va > vb || (va == vb && a < b);
            });
            candidates[t].push_back(id_map[rows[0]]);
            candidates[t].push_back(id_map[rows[1]]);
        }
        best_score[t] = -INFINITY;
        for (const int id : candidates[t]) {
            double s = 0;
            for (int k = 0; k < kHidden; ++k)
                s += static_cast<double>(hidden[t * kHidden + k]) * head[static_cast<std::size_t>(id) * kHidden + k];
            if (s > best_score[t] || (s == best_score[t] && id < expected[t])) {
                best_score[t] = s;
                expected[t]   = id;
            }
        }
    }

    DeviceBuffer d_hidden = to_device_bf16(hidden);
    DeviceBuffer d_approx = to_device_bf16(approximate);
    DeviceBuffer d_ids    = to_device_i32(id_map);
    DeviceBuffer d_codes  = fp8 ? to_device(codes) : to_device(bf16_rows);
    DeviceBuffer d_scales = fp8 ? to_device(scales) : to_device(std::vector<std::uint16_t>(1));
    DeviceBuffer d_cand_ids(static_cast<std::size_t>(kCandidates) * kTokens * 4);
    DeviceBuffer d_cand_scores(static_cast<std::size_t>(kCandidates) * kTokens * 4);
    DeviceBuffer d_out(kTokens * 4);
    Weight w{};
    w.qtype       = fp8 ? QType::FP8_E4M3FN_ROW_BF16 : QType::BF16;
    w.layout      = fp8 ? QuantLayout::RowScale : QuantLayout::Contiguous;
    w.qdata       = d_codes.p;
    w.scales      = fp8 ? d_scales.p : nullptr;
    w.scale_dtype = DType::BF16;
    w.n           = kVocab;
    w.k           = kHidden;
    Tensor hidden_t(d_hidden.p, DType::BF16, {kHidden, kTokens});
    Tensor approx_t(d_approx.p, DType::BF16, {kShortlist, kTokens});
    Tensor cand_ids(d_cand_ids.p, DType::I32, {kCandidates, kTokens});
    Tensor cand_scores(d_cand_scores.p, DType::FP32, {kCandidates, kTokens});
    Tensor out(d_out.p, DType::I32, {kTokens});
    ops::shortlist_exact_argmax(hidden_t, approx_t, kShortlist, w,
                                static_cast<const std::int32_t*>(d_ids.p), cand_ids, cand_scores,
                                out, nullptr);
    cuda_synchronize();
    const auto got = from_device<std::int32_t>(d_out, kTokens);
    int failures   = 0;
    for (int t = 0; t < kTokens; ++t) {
        if (got[t] == expected[t]) continue;
        // A different pick is acceptable only within FP32 accumulation error of the best score.
        double s = 0;
        for (int k = 0; k < kHidden; ++k)
            s += static_cast<double>(hidden[t * kHidden + k]) * head[static_cast<std::size_t>(got[t]) * kHidden + k];
        const bool candidate = std::find(candidates[t].begin(), candidates[t].end(), got[t]) != candidates[t].end();
        if (!candidate || std::abs(s - best_score[t]) > 1e-4 * std::abs(best_score[t]) + 1e-5) {
            std::cerr << (fp8 ? "FP8" : "BF16") << " token " << t << ": got " << got[t]
                      << " expected " << expected[t] << '\n';
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) { return 77; }
    try {
        const int failures = run_case(false) + run_case(true);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " shortlist exact argmax (BF16, FP8 rows)\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "shortlist exact argmax: " << error.what() << '\n';
        return 1;
    }
}
