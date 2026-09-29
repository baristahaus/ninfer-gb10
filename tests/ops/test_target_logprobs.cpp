#include "ninfer/ops/target_logprobs.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr ReductionCriterion kTargetLogprobsFp32Criterion{
    /*relative_l2=*/2.0e-5,
    /*gross_absolute=*/2.0e-4,
    /*gross_relative_to_max_reference=*/0.0,
};

std::vector<std::int32_t> make_targets(std::int32_t valid_rows, std::int32_t columns) {
    std::vector<std::int32_t> targets(static_cast<std::size_t>(columns));
    for (std::int32_t column = 0; column < columns; ++column) {
        if (column % 4 == 0) {
            targets[static_cast<std::size_t>(column)] = 0;
        } else if (column % 4 == 1) {
            targets[static_cast<std::size_t>(column)] = valid_rows - 1;
        } else {
            targets[static_cast<std::size_t>(column)] =
                static_cast<std::int32_t>((static_cast<std::uint64_t>(column + 1) * 7919u) %
                                          static_cast<std::uint32_t>(valid_rows));
        }
    }
    return targets;
}

std::vector<std::uint16_t> make_random_logits(std::int32_t physical_rows, std::int32_t valid_rows,
                                              std::int32_t columns) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            const std::uint32_t mixed = static_cast<std::uint32_t>(row) * 1664525u +
                                        static_cast<std::uint32_t>(column + 1) * 1013904223u;
            const float value = -24.0f + static_cast<float>(mixed % 6144u) * (1.0f / 128.0f);
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(value);
        }
        for (std::int32_t row = valid_rows; row < physical_rows; ++row) {
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(96.0f);
        }
    }
    return logits;
}

std::vector<std::uint16_t> make_uniform_logits(std::int32_t physical_rows, std::int32_t valid_rows,
                                               std::int32_t columns, float value) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns,
                                      f32_to_bf16(value));
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        for (std::int32_t row = valid_rows; row < physical_rows; ++row) {
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(112.0f);
        }
    }
    return logits;
}

std::vector<std::uint16_t> make_shift_logits(std::int32_t physical_rows, std::int32_t valid_rows,
                                             std::int32_t columns, float shift) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            const int centered = (row * 37 + column * 11) % 65 - 32;
            const float value  = static_cast<float>(centered) * 0.25f + shift;
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(value);
        }
        for (std::int32_t row = valid_rows; row < physical_rows; ++row) {
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(120.0f);
        }
    }
    return logits;
}

std::vector<std::uint16_t> make_extreme_logits(std::int32_t physical_rows, std::int32_t valid_rows,
                                               std::int32_t columns) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns);
    constexpr float values[] = {-80.0f, -32.0f, -1.0f, 0.0f, 1.0f, 32.0f, 80.0f};
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            const auto index = static_cast<std::size_t>(row + column) % std::size(values);
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(values[index]);
        }
        for (std::int32_t row = valid_rows; row < physical_rows; ++row) {
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(120.0f);
        }
    }
    return logits;
}

std::vector<double> target_logprobs_oracle(const std::vector<std::uint16_t>& logits,
                                           const std::vector<std::int32_t>& targets,
                                           std::int32_t physical_rows, std::int32_t valid_rows) {
    std::vector<double> expected(targets.size());
    for (std::size_t column = 0; column < targets.size(); ++column) {
        const std::size_t base = column * static_cast<std::size_t>(physical_rows);
        double maximum         = -std::numeric_limits<double>::infinity();
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            maximum = std::max(maximum, static_cast<double>(bf16_to_f32(logits[base + row])));
        }
        double sum = 0.0;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            sum += std::exp(static_cast<double>(bf16_to_f32(logits[base + row])) - maximum);
        }
        const double target = static_cast<double>(bf16_to_f32(logits[base + targets[column]]));
        expected[column]    = target - maximum - std::log(sum);
    }
    return expected;
}

std::vector<double> fp32_as_double(const void* device, std::size_t count) {
    const auto values = from_device<float>(device, count);
    return {values.begin(), values.end()};
}

int run_case(const std::string& label, std::int32_t physical_rows, std::int32_t valid_rows,
             std::int32_t columns, const std::vector<std::uint16_t>& logits) {
    const auto targets  = make_targets(valid_rows, columns);
    const auto expected = target_logprobs_oracle(logits, targets, physical_rows, valid_rows);

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_targets(targets.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_output(targets.size() * sizeof(float));
    device_logits.copy_from_host(logits.data(), device_logits.bytes());
    device_targets.copy_from_host(targets.data(), device_targets.bytes());
    device_output.fill(0xcd);

    Tensor logits_tensor(device_logits.data(), DType::BF16, {physical_rows, columns});
    Tensor targets_tensor(device_targets.data(), DType::I32, {columns});
    Tensor output_tensor(device_output.data(), DType::FP32, {columns});
    ops::target_logprobs(logits_tensor, targets_tensor, valid_rows, output_tensor, nullptr,
                         nullptr, nullptr);
    cuda_synchronize();

    int failures = verify_reduction(label, fp32_as_double(device_output.data(), targets.size()),
                                    expected, kTargetLogprobsFp32Criterion);
    failures +=
        verify_exact((label + " preserves logits").c_str(),
                     from_device<std::uint16_t>(device_logits.data(), logits.size()), logits);
    failures +=
        verify_exact((label + " preserves targets").c_str(),
                     from_device<std::int32_t>(device_targets.data(), targets.size()), targets);
    failures += device_logits.verify_guards(label + " logits guards");
    failures += device_targets.verify_guards(label + " target guards");
    failures += device_output.verify_guards(label + " output guards");
    return failures;
}

// The report is the model's own distribution, so the oracle is a plain FP64 log-softmax over the
// represented BF16 logits: no temperature, no penalties, and nothing from the sampler or the round.
// Fixture values are spaced far enough apart that an FP32 and an FP64 ordering cannot disagree, which
// is what lets the ranking ids be compared exactly.
struct RankingFixture {
    std::int32_t top_k = 0;
};

double raw_value(const std::vector<std::uint16_t>& logits, std::size_t base, std::int32_t row) {
    return static_cast<double>(bf16_to_f32(logits[base + static_cast<std::size_t>(row)]));
}

// The Op's ranking rule: descending value, lower token id breaking an exact tie.
bool ranking_before(std::pair<double, std::int32_t> lhs, std::pair<double, std::int32_t> rhs) {
    if (lhs.first != rhs.first) { return lhs.first > rhs.first; }
    return lhs.second < rhs.second;
}

int run_ranked_case(const std::string& label, std::int32_t physical_rows, std::int32_t valid_rows,
                    std::int32_t columns, const std::vector<std::uint16_t>& logits,
                    const RankingFixture& fixture) {
    const auto targets = make_targets(valid_rows, columns);

    std::vector<double> expected_output(static_cast<std::size_t>(columns));
    std::vector<std::int32_t> expected_ids(static_cast<std::size_t>(fixture.top_k) * columns,
                                           ops::kNoReportedLogprobRank);
    std::vector<double> expected_rank_logprobs(
        static_cast<std::size_t>(fixture.top_k) * columns, -std::numeric_limits<double>::infinity());
    std::vector<std::int32_t> reported_count(static_cast<std::size_t>(columns), 0);

    for (std::size_t column = 0; column < static_cast<std::size_t>(columns); ++column) {
        const std::size_t base = column * static_cast<std::size_t>(physical_rows);
        double maximum         = -std::numeric_limits<double>::infinity();
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            maximum = std::max(maximum, raw_value(logits, base, row));
        }
        double sum = 0.0;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            sum += std::exp(raw_value(logits, base, row) - maximum);
        }
        const double normalizer = maximum + std::log(sum);
        expected_output[column] = raw_value(logits, base, targets[column]) - normalizer;

        std::vector<std::pair<double, std::int32_t>> ranked;
        ranked.reserve(static_cast<std::size_t>(valid_rows));
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            ranked.emplace_back(raw_value(logits, base, row), row);
        }
        std::stable_sort(ranked.begin(), ranked.end(), ranking_before);
        const std::int32_t reported =
            fixture.top_k < valid_rows ? fixture.top_k : valid_rows;
        reported_count[column] = reported;
        for (std::int32_t rank = 0; rank < reported; ++rank) {
            expected_ids[static_cast<std::size_t>(rank) + column * fixture.top_k] =
                ranked[static_cast<std::size_t>(rank)].second;
            expected_rank_logprobs[static_cast<std::size_t>(rank) +
                                   column * fixture.top_k] =
                ranked[static_cast<std::size_t>(rank)].first - normalizer;
        }
    }

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_targets(targets.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_output(targets.size() * sizeof(float));
    device_logits.copy_from_host(logits.data(), device_logits.bytes());
    device_targets.copy_from_host(targets.data(), device_targets.bytes());
    device_output.fill(0xcd);

    const std::size_t ranking_elements =
        static_cast<std::size_t>(fixture.top_k) * static_cast<std::size_t>(columns);
    GuardedDeviceBuffer device_top_ids(ranking_elements * sizeof(std::int32_t));
    GuardedDeviceBuffer device_top_logprobs(ranking_elements * sizeof(float));
    device_top_ids.fill(0x5a);
    device_top_logprobs.fill(0x5a);

    Tensor logits_tensor(device_logits.data(), DType::BF16, {physical_rows, columns});
    Tensor targets_tensor(device_targets.data(), DType::I32, {columns});
    Tensor output_tensor(device_output.data(), DType::FP32, {columns});
    Tensor top_ids_tensor(device_top_ids.data(), DType::I32, {fixture.top_k, columns});
    Tensor top_logprobs_tensor(device_top_logprobs.data(), DType::FP32, {fixture.top_k, columns});

    Tensor* top_ids_pointer      = &top_ids_tensor;
    Tensor* top_logprobs_pointer = &top_logprobs_tensor;
    ops::target_logprobs(logits_tensor, targets_tensor, valid_rows, output_tensor, top_ids_pointer,
                         top_logprobs_pointer, nullptr);
    cuda_synchronize();

    int failures = verify_reduction(label, fp32_as_double(device_output.data(), targets.size()),
                                    expected_output, kTargetLogprobsFp32Criterion);
    failures += verify_exact(
        (label + " ranking ids").c_str(),
        from_device<std::int32_t>(device_top_ids.data(), ranking_elements), expected_ids);

    // Compare the reported prefix under the reduction criterion and check the sentinel tail
    // exactly: an out-of-range rank is a defined value, not an approximation.
    const auto actual_rank_logprobs =
        from_device<float>(device_top_logprobs.data(), ranking_elements);
    for (std::size_t column = 0; column < expected_output.size(); ++column) {
        const std::int32_t reported = reported_count[column];
        std::vector<double> actual_prefix(static_cast<std::size_t>(reported));
        std::vector<double> expected_prefix(static_cast<std::size_t>(reported));
        for (std::int32_t rank = 0; rank < reported; ++rank) {
            const std::size_t index = static_cast<std::size_t>(rank) + column * fixture.top_k;
            actual_prefix[static_cast<std::size_t>(rank)]   = actual_rank_logprobs[index];
            expected_prefix[static_cast<std::size_t>(rank)] = expected_rank_logprobs[index];
        }
        failures += verify_reduction(label + " ranking logprobs", actual_prefix, expected_prefix,
                                     kTargetLogprobsFp32Criterion);
        for (std::int32_t rank = reported; rank < fixture.top_k; ++rank) {
            const std::size_t index = static_cast<std::size_t>(rank) + column * fixture.top_k;
            const double reported_value = static_cast<double>(actual_rank_logprobs[index]);
            if (!(std::isinf(reported_value) && reported_value < 0.0)) {
                std::cerr << label << ": rank " << rank << " of column " << column
                          << " expected -inf, got " << actual_rank_logprobs[index] << '\n';
                ++failures;
            }
        }
    }

    failures += device_logits.verify_guards(label + " logits guards");
    failures += device_targets.verify_guards(label + " target guards");
    failures += device_output.verify_guards(label + " output guards");
    failures += device_top_ids.verify_guards(label + " top_ids guards");
    failures += device_top_logprobs.verify_guards(label + " top_logprobs guards");
    return failures;
}

template <class Function>
int expect_invalid(const char* label, Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& error) {
        std::cerr << label << ": expected invalid_argument, got " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

int run_validation_cases() {
    DeviceBuffer logits_data(8 * 3 * sizeof(std::uint16_t));
    DeviceBuffer targets_data(3 * sizeof(std::int32_t));
    DeviceBuffer output_data(3 * sizeof(float));
    DeviceBuffer ranking_data(20 * 3 * sizeof(std::int32_t));
    DeviceBuffer rank_logprob_data(20 * 3 * sizeof(float));
    Tensor logits(logits_data.p, DType::BF16, {8, 3});
    Tensor targets(targets_data.p, DType::I32, {3});
    Tensor output(output_data.p, DType::FP32, {3});
    Tensor top_ids(ranking_data.p, DType::I32, {20, 3});
    Tensor top_logprobs(rank_logprob_data.p, DType::FP32, {20, 3});

    int failures = 0;
    failures += expect_invalid(
        "target_logprobs rejects valid_rows=0",
        [&] { ops::target_logprobs(logits, targets, 0, output, nullptr, nullptr, nullptr); });
    failures += expect_invalid("target_logprobs rejects valid_rows>physical_rows", [&] {
        ops::target_logprobs(logits, targets, 9, output, nullptr, nullptr, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects target shape mismatch", [&] {
        Tensor wrong_targets(targets_data.p, DType::I32, {2});
        ops::target_logprobs(logits, wrong_targets, 8, output, nullptr, nullptr, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects output dtype", [&] {
        Tensor wrong_output(output_data.p, DType::BF16, {3});
        ops::target_logprobs(logits, targets, 8, wrong_output, nullptr, nullptr, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects non-contiguous logits", [&] {
        Tensor strided_logits = logits;
        strided_logits.nb[1] += 2;
        ops::target_logprobs(strided_logits, targets, 8, output, nullptr, nullptr, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects null output", [&] {
        Tensor null_output(nullptr, DType::FP32, {3});
        ops::target_logprobs(logits, targets, 8, null_output, nullptr, nullptr, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects output alias", [&] {
        Tensor alias_output(logits_data.p, DType::FP32, {3});
        ops::target_logprobs(logits, targets, 8, alias_output, nullptr, nullptr, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects non-matrix logits", [&] {
        Tensor rank_three = logits;
        rank_three.ne[2]  = 2;
        ops::target_logprobs(rank_three, targets, 8, output, nullptr, nullptr, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects a half ranking request", [&] {
        ops::target_logprobs(logits, targets, 8, output, &top_ids, nullptr, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects a rank count above the ceiling", [&] {
        Tensor wide_ids(ranking_data.p, DType::I32, {21, 3});
        Tensor wide_logprobs(rank_logprob_data.p, DType::FP32, {21, 3});
        ops::target_logprobs(logits, targets, 8, output, &wide_ids, &wide_logprobs, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects a ranking column mismatch", [&] {
        Tensor narrow_ids(ranking_data.p, DType::I32, {20, 2});
        Tensor narrow_logprobs(rank_logprob_data.p, DType::FP32, {20, 2});
        ops::target_logprobs(logits, targets, 8, output, &narrow_ids, &narrow_logprobs, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects mismatched ranking shapes", [&] {
        Tensor narrow_logprobs(rank_logprob_data.p, DType::FP32, {19, 3});
        ops::target_logprobs(logits, targets, 8, output, &top_ids, &narrow_logprobs, nullptr);
    });
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case("target_logprobs full vocabulary C=3", 248320, 248077, 3,
                         make_random_logits(248320, 248077, 3));
    failures += run_case("target_logprobs non-aligned rows C=1025", 523, 509, 1025,
                         make_random_logits(523, 509, 1025));
    failures += run_case("target_logprobs uniform logits", 263, 257, 1024,
                         make_uniform_logits(263, 257, 1024, 3.5f));
    failures +=
        run_case("target_logprobs one valid row", 13, 1, 7, make_uniform_logits(13, 1, 7, -7.0f));
    failures += run_case("target_logprobs extreme finite logits", 263, 257, 17,
                         make_extreme_logits(263, 257, 17));
    failures += run_case("target_logprobs base for constant shift", 257, 257, 31,
                         make_shift_logits(257, 257, 31, 0.0f));
    failures += run_case("target_logprobs shifted logits", 257, 257, 31,
                         make_shift_logits(257, 257, 31, 32.0f));

    // Generation reporting: the full ranking at the ceiling, a narrower request, and a domain
    // smaller than the request. Ranking is the model's own order, so the reported top-1 of a raw
    // distribution is the argmax of the same logits the chosen value came from.
    RankingFixture ceiling;
    ceiling.top_k = 20;
    failures += run_ranked_case("target_logprobs ranked at the ceiling", 257, 257, 5,
                                make_shift_logits(257, 257, 5, 0.0f), ceiling);

    RankingFixture narrow;
    narrow.top_k = 4;
    failures += run_ranked_case("target_logprobs ranked below the ceiling", 257, 257, 5,
                                make_shift_logits(257, 257, 5, 4.0f), narrow);

    RankingFixture tiny;
    tiny.top_k = 3;
    failures += run_ranked_case("target_logprobs ranked over a small domain", 17, 17, 3,
                                make_random_logits(17, 17, 3), tiny);

    // The vocabulary is smaller than the report: the tail is the defined sentinel, not a value.
    RankingFixture wide;
    wide.top_k = 20;
    failures += run_ranked_case("target_logprobs ranking beyond the vocabulary", 17, 13, 3,
                                make_random_logits(17, 13, 3), wide);

    failures += run_validation_cases();

    std::cout << (failures ? "FAIL" : "OK") << " target_logprobs\n";
    return failures ? 1 : 0;
}
