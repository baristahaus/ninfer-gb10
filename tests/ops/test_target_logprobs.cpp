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
    ops::target_logprobs(logits_tensor, targets_tensor, valid_rows, {}, output_tensor, nullptr,
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

// The reported distribution is defined by the committed-token counts as they stood when the column
// was drawn. The Op receives the counts the engine actually holds at that point, which already
// include this round's own published tokens, so a fixture states the prior history plus the lane's
// published tokens and the oracle adds only the prefix before the column. A non-positive temperature
// reports the un-scaled distribution. Values in the fixtures are spaced far enough apart, with
// dyadic penalties and temperatures, that an FP32 and an FP64 ordering cannot disagree.
struct RankingFixture {
    float temperature      = 1.0F;
    float presence_penalty = 0.0F;
    float frequency_penalty= 0.0F;
    std::vector<std::int32_t> history;   // committed counts before this round, per vocabulary row
    std::vector<std::int32_t> published; // [columns_per_lane, lanes], column-major
    std::vector<std::int32_t> produced;  // published entries per lane, or empty for full columns
    std::int32_t columns_per_lane = 1;
    std::int32_t top_k            = 0;
};

double scaled_value(const std::vector<std::uint16_t>& logits, std::size_t base, std::int32_t row,
                    const RankingFixture& fixture, std::size_t column) {
    int count =
        fixture.history.empty() ? 0 : fixture.history[static_cast<std::size_t>(row)];
    if (!fixture.published.empty()) {
        const std::size_t lanes     = fixture.published.size() /
                                  static_cast<std::size_t>(fixture.columns_per_lane);
        const std::size_t lane      = column / static_cast<std::size_t>(fixture.columns_per_lane);
        const std::int32_t within   = static_cast<std::int32_t>(
            column - lane * static_cast<std::size_t>(fixture.columns_per_lane));
        const std::int32_t produced =
            fixture.produced.empty()
                ? fixture.columns_per_lane
                : fixture.produced[lane < lanes ? lane : lanes - 1U];
        const std::int32_t prefix   = within < produced ? within : produced;
        for (std::int32_t entry = 0; entry < prefix; ++entry) {
            if (fixture.published[static_cast<std::size_t>(entry) +
                                  lane * static_cast<std::size_t>(fixture.columns_per_lane)] ==
                row) {
                ++count;
            }
        }
    }
    double value = static_cast<double>(bf16_to_f32(logits[base + static_cast<std::size_t>(row)]));
    if (count > 0) { value -= static_cast<double>(fixture.presence_penalty); }
    value -= static_cast<double>(fixture.frequency_penalty) * static_cast<double>(count);
    return fixture.temperature > 0.0F
               ? value / static_cast<double>(fixture.temperature)
               : value;
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
            maximum = std::max(maximum, scaled_value(logits, base, row, fixture, column));
        }
        double sum = 0.0;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            sum += std::exp(scaled_value(logits, base, row, fixture, column) - maximum);
        }
        const double normalizer = maximum + std::log(sum);
        expected_output[column] =
            scaled_value(logits, base, targets[column], fixture, column) - normalizer;

        std::vector<std::pair<double, std::int32_t>> ranked;
        ranked.reserve(static_cast<std::size_t>(valid_rows));
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            ranked.emplace_back(scaled_value(logits, base, row, fixture, column), row);
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

    // The engine's committed counts at the time this Op runs already contain the tokens the round
    // published, up to each lane's produced count. Building them that way is the point of the test:
    // the Op has to subtract the ones that postdate each column.
    std::vector<std::int32_t> counts = fixture.history;
    if (!counts.empty() && !fixture.published.empty()) {
        const std::size_t lane_count =
            fixture.published.size() / static_cast<std::size_t>(fixture.columns_per_lane);
        for (std::size_t lane = 0; lane < lane_count; ++lane) {
            const std::int32_t produced =
                fixture.produced.empty()
                    ? fixture.columns_per_lane
                    : fixture.produced[lane < fixture.produced.size()
                                          ? lane
                                          : fixture.produced.size() - 1U];
            for (std::int32_t entry = 0; entry < produced; ++entry) {
                const std::int32_t token = fixture.published[static_cast<std::size_t>(entry) +
                                                             lane * static_cast<std::size_t>(
                                                                       fixture.columns_per_lane)];
                ++counts[static_cast<std::size_t>(token)];
            }
        }
    }
    GuardedDeviceBuffer device_counts(counts.size() * sizeof(std::int32_t));
    if (!counts.empty()) { device_counts.copy_from_host(counts.data(), device_counts.bytes()); }
    GuardedDeviceBuffer device_published(fixture.published.size() * sizeof(std::int32_t));
    if (!fixture.published.empty()) {
        device_published.copy_from_host(fixture.published.data(), device_published.bytes());
    }
    GuardedDeviceBuffer device_produced(fixture.produced.size() * sizeof(std::int32_t));
    if (!fixture.produced.empty()) {
        device_produced.copy_from_host(fixture.produced.data(), device_produced.bytes());
    }
    const std::size_t lanes =
        fixture.columns_per_lane > 0
            ? static_cast<std::size_t>(columns) / static_cast<std::size_t>(fixture.columns_per_lane)
            : 1;
    std::vector<ops::SamplingConfig> configs(lanes == 0 ? 1 : lanes);
    for (auto& config : configs) {
        config.temperature       = fixture.temperature;
        config.presence_penalty  = fixture.presence_penalty;
        config.frequency_penalty = fixture.frequency_penalty;
        config.token_counts = counts.empty() ? nullptr
                                             : static_cast<std::int32_t*>(device_counts.data());
    }
    GuardedDeviceBuffer device_configs(configs.size() * sizeof(ops::SamplingConfig));
    device_configs.copy_from_host(configs.data(), device_configs.bytes());

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

    ops::TargetLogprobOptions options;
    options.configs           = static_cast<const ops::SamplingConfig*>(device_configs.data());
    options.columns_per_lane  = fixture.columns_per_lane;
    options.round_tokens      = fixture.published.empty()
                                    ? nullptr
                                    : static_cast<const std::int32_t*>(device_published.data());
    options.round_produced    = fixture.produced.empty()
                                    ? nullptr
                                    : static_cast<const std::int32_t*>(device_produced.data());

    Tensor* top_ids_pointer   = &top_ids_tensor;
    Tensor* top_logprobs_pointer = &top_logprobs_tensor;
    ops::target_logprobs(logits_tensor, targets_tensor, valid_rows, options, output_tensor,
                         top_ids_pointer, top_logprobs_pointer, nullptr);
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
        [&] { ops::target_logprobs(logits, targets, 0, {}, output, nullptr, nullptr, nullptr); });
    failures += expect_invalid("target_logprobs rejects valid_rows>physical_rows", [&] {
        ops::target_logprobs(logits, targets, 9, {}, output, nullptr, nullptr, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects target shape mismatch", [&] {
        Tensor wrong_targets(targets_data.p, DType::I32, {2});
        ops::target_logprobs(logits, wrong_targets, 8, {}, output, nullptr, nullptr, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects output dtype", [&] {
        Tensor wrong_output(output_data.p, DType::BF16, {3});
        ops::target_logprobs(logits, targets, 8, {}, wrong_output, nullptr, nullptr, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects non-contiguous logits", [&] {
        Tensor strided_logits = logits;
        strided_logits.nb[1] += 2;
        ops::target_logprobs(strided_logits, targets, 8, {}, output, nullptr, nullptr, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects null output", [&] {
        Tensor null_output(nullptr, DType::FP32, {3});
        ops::target_logprobs(logits, targets, 8, {}, null_output, nullptr, nullptr, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects output alias", [&] {
        Tensor alias_output(logits_data.p, DType::FP32, {3});
        ops::target_logprobs(logits, targets, 8, {}, alias_output, nullptr, nullptr, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects non-matrix logits", [&] {
        Tensor rank_three = logits;
        rank_three.ne[2]  = 2;
        ops::target_logprobs(rank_three, targets, 8, {}, output, nullptr, nullptr, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects a half ranking request", [&] {
        ops::target_logprobs(logits, targets, 8, {}, output, &top_ids, nullptr, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects a rank count above the ceiling", [&] {
        Tensor wide_ids(ranking_data.p, DType::I32, {21, 3});
        Tensor wide_logprobs(rank_logprob_data.p, DType::FP32, {21, 3});
        ops::target_logprobs(logits, targets, 8, {}, output, &wide_ids, &wide_logprobs, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects a ranking column mismatch", [&] {
        Tensor narrow_ids(ranking_data.p, DType::I32, {20, 2});
        Tensor narrow_logprobs(rank_logprob_data.p, DType::FP32, {20, 2});
        ops::target_logprobs(logits, targets, 8, {}, output, &narrow_ids, &narrow_logprobs, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects mismatched ranking shapes", [&] {
        Tensor narrow_logprobs(rank_logprob_data.p, DType::FP32, {19, 3});
        ops::target_logprobs(logits, targets, 8, {}, output, &top_ids, &narrow_logprobs, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects lane configs without configs", [&] {
        ops::TargetLogprobOptions options;
        options.columns_per_lane = 2;
        ops::target_logprobs(logits, targets, 8, options, output, nullptr, nullptr, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects round tokens without configs", [&] {
        ops::TargetLogprobOptions options;
        DeviceBuffer published_data(4 * 3 * sizeof(std::int32_t));
        options.round_tokens = static_cast<const std::int32_t*>(published_data.p);
        ops::target_logprobs(logits, targets, 8, options, output, nullptr, nullptr, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects produced counts without round tokens", [&] {
        ops::TargetLogprobOptions options;
        DeviceBuffer config_data(3 * sizeof(ops::SamplingConfig));
        DeviceBuffer produced_data(3 * sizeof(std::int32_t));
        options.configs        = static_cast<const ops::SamplingConfig*>(config_data.p);
        options.round_produced = static_cast<const std::int32_t*>(produced_data.p);
        ops::target_logprobs(logits, targets, 8, options, output, nullptr, nullptr, nullptr);
    });
    failures += expect_invalid("target_logprobs rejects a lane width that misses columns", [&] {
        ops::TargetLogprobOptions options;
        DeviceBuffer config_data(3 * sizeof(ops::SamplingConfig));
        DeviceBuffer published_data(4 * 3 * sizeof(std::int32_t));
        options.configs          = static_cast<const ops::SamplingConfig*>(config_data.p);
        options.columns_per_lane = 2; // three columns do not divide by two
        options.round_tokens     = static_cast<const std::int32_t*>(published_data.p);
        ops::target_logprobs(logits, targets, 8, options, output, nullptr, nullptr, nullptr);
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

    // Sampled generation: temperature scaling and a full 20-rank report at the ceiling.
    RankingFixture temperature;
    temperature.temperature = 0.5F;
    temperature.top_k       = 20;
    failures += run_ranked_case("target_logprobs ranked at temperature 0.5", 257, 257, 5,
                                make_shift_logits(257, 257, 5, 0.0f), temperature);

    // Penalties with a published round prefix, and a narrower report than the ceiling. One lane of
    // five verify columns: the engine has already counted all five published tokens, and column c
    // was drawn against only the first c of them, so the repeated row carries a frequency count of
    // two from column 2 on and none before it.
    RankingFixture penalties;
    penalties.temperature       = 0.7F;
    penalties.presence_penalty  = 0.5F;
    penalties.frequency_penalty = 0.25F;
    penalties.history.resize(257);
    for (std::int32_t row = 0; row < 257; ++row) {
        penalties.history[static_cast<std::size_t>(row)] = row % 3;
    }
    penalties.columns_per_lane = 5;
    penalties.published        = {3, 3, 100, 256, 7};
    penalties.top_k            = 4;
    failures += run_ranked_case("target_logprobs ranked with penalties and a published prefix", 257,
                                257, 5, make_shift_logits(257, 257, 5, 4.0f), penalties);

    // The same round truncated: the accept step counted only the first three published tokens, so the
    // last two are absent from the counts the Op receives and must not be subtracted from them.
    RankingFixture truncated = penalties;
    truncated.produced       = {3};
    failures += run_ranked_case("target_logprobs ranked with a truncated published round", 257, 257,
                                5, make_shift_logits(257, 257, 5, 4.0f), truncated);

    // Ordinary decode shape - one column per lane - and the regression this reconciliation exists
    // for: ops::sample has already counted each lane's drawn token, so reporting the draw's own view
    // means that token is penalized exactly once, not once more for the report.
    RankingFixture drawn;
    drawn.temperature       = 1.0F;
    drawn.presence_penalty  = 1.0F;
    drawn.frequency_penalty = 1.0F;
    drawn.history.assign(17, 0);
    drawn.published         = {4, 9, 4};
    drawn.top_k             = 3;
    failures += run_ranked_case("target_logprobs does not penalize a column for its own token", 17,
                                17, 3, make_random_logits(17, 17, 3), drawn);

    // A greedy config has no temperature, so tau is 1 while penalties still apply.
    RankingFixture greedy;
    greedy.temperature        = 0.0F;
    greedy.presence_penalty   = 0.5F;
    greedy.history.assign(13, 1);
    greedy.top_k              = 5;
    failures += run_ranked_case("target_logprobs greedy reports tau=1", 13, 13, 3,
                                make_extreme_logits(13, 13, 3), greedy);

    // The vocabulary is smaller than the report: the tail is the defined sentinel, not a value.
    RankingFixture wide;
    wide.temperature = 1.0F;
    wide.top_k       = 20;
    failures += run_ranked_case("target_logprobs ranking beyond the vocabulary", 17, 13, 3,
                                make_random_logits(17, 13, 3), wide);

    failures += run_validation_cases();

    std::cout << (failures ? "FAIL" : "OK") << " target_logprobs\n";
    return failures ? 1 : 0;
}
