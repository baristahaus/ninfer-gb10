// Host-only numerical gate on the sharded forward's composition rules.
//
// The sharded forward is three claims about arithmetic, and none of them needs a GPU to check:
//
//   1. ROW-PARALLEL. Splitting a projection's OUTPUT rows and computing each rank's block from the
//      full input reproduces the unsharded projection exactly — the blocks concatenate.
//   2. COLUMN-PARALLEL. Splitting a projection's INPUT columns, computing each rank's partial, and
//      summing the partials reproduces the unsharded projection.
//   3. THE RESIDUAL IS ADDED ONCE, AFTER THE SUM. A column-parallel projection must not use the
//      fused `linear_add`: the residual is replicated, so adding it on both ranks and then summing
//      counts it twice. This is the one rule in the design that silently produces plausible wrong
//      output rather than an error.
//
// The oracle is a naive FP64 evaluation of the complete projection. The split arithmetic is the
// production `artifact::split_evenly`, so the test uses the same split the shard table does.
//
// WHAT THIS TEST DOES NOT CLAIM: that the width-2 result is BIT-IDENTICAL to the width-1 result.
// It is not. Each rank rounds its partial to BF16 before the all-reduce, and the reduce rounds the
// sum once more, so a column-parallel projection carries two BF16 storage roundings the unsharded
// one does not. The last group measures that error against the FP64 oracle instead of pretending
// it away.

#include "artifact/shard.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

// Round-to-nearest-even BF16 of a float, as the all-reduce's store does. Implemented here rather
// than pulled from a CUDA header because this test runs without a device.
float to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7F800000U) == 0x7F800000U) { return value; } // inf / nan pass through
    const std::uint32_t lsb      = (bits >> 16) & 1U;
    const std::uint32_t rounding = 0x7FFFU + lsb;
    bits += rounding;
    bits &= 0xFFFF0000U;
    float out = 0.0F;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// y[n, t] = sum_k w[n, k] * x[k, t], in double. The independent oracle.
std::vector<double> reference(const std::vector<double>& w, const std::vector<double>& x, int rows,
                              int inner, int columns) {
    std::vector<double> y(static_cast<std::size_t>(rows) * columns, 0.0);
    for (int n = 0; n < rows; ++n) {
        for (int t = 0; t < columns; ++t) {
            double sum = 0.0;
            for (int k = 0; k < inner; ++k) {
                sum += w[static_cast<std::size_t>(n) * inner + k] *
                       x[static_cast<std::size_t>(k) * columns + t];
            }
            y[static_cast<std::size_t>(n) * columns + t] = sum;
        }
    }
    return y;
}

std::vector<double> sample(std::size_t count, double seed, double scale) {
    std::vector<double> out(count);
    std::uint64_t state = static_cast<std::uint64_t>(seed);
    for (std::size_t i = 0; i < count; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const double unit = static_cast<double>(state >> 11) / 9007199254740992.0; // [0,1)
        out[i]            = (unit * 2.0 - 1.0) * scale;
    }
    return out;
}

bool equal_within(const std::vector<double>& a, const std::vector<double>& b, double tolerance) {
    if (a.size() != b.size()) { return false; }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!(std::fabs(a[i] - b[i]) <= tolerance)) { return false; }
    }
    return true;
}

double worst_relative(const std::vector<double>& got, const std::vector<double>& want) {
    double worst = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double magnitude = std::max(1.0, std::fabs(want[i]));
        worst                  = std::max(worst, std::fabs(got[i] - want[i]) / magnitude);
    }
    return worst;
}

// The 27B's dense MLP shape, shrunk so the oracle runs in FP64 without a 17408x5120 matrix: the
// ratio of rows to inner to columns is what the composition argument depends on, not the size.
constexpr int kRows     = 8;
constexpr int kInner    = 6;
constexpr int kColumns  = 3;
constexpr int kWidth    = 2;

// ------------------------------------------------------------------ 1. row-parallel
void check_row_parallel() {
    const auto w = sample(static_cast<std::size_t>(kRows) * kInner, 1, 1.0);
    const auto x = sample(static_cast<std::size_t>(kInner) * kColumns, 2, 1.0);
    const auto whole = reference(w, x, kRows, kInner, kColumns);

    const auto placement = ninfer::artifact::split_evenly(ninfer::artifact::ShardAxis::Rows, kRows,
                                                          kWidth);
    std::vector<double> composed(whole.size(), 0.0);
    for (int rank = 0; rank < kWidth; ++rank) {
        const auto range = placement.ranges[static_cast<std::size_t>(rank)];
        // Rank `rank` computes ONLY its own output rows, from the FULL input.
        std::vector<double> shard_w(
            w.begin() + range.begin * kInner,
            w.begin() + (range.begin + range.extent) * kInner);
        const auto partial = reference(shard_w, x, static_cast<int>(range.extent), kInner, kColumns);
        for (int n = 0; n < range.extent; ++n) {
            for (int t = 0; t < kColumns; ++t) {
                composed[static_cast<std::size_t>(range.begin + n) * kColumns + t] =
                    partial[static_cast<std::size_t>(n) * kColumns + t];
            }
        }
    }
    check(equal_within(composed, whole, 1e-12),
          "row-parallel blocks concatenate to the unsharded projection");
    std::cout << "  row-parallel: " << kRows << " rows -> " << placement.ranges[0].extent << " + "
              << placement.ranges[1].extent << ", exact\n";
}

// ------------------------------------------------------------------ 2. column-parallel
void check_column_parallel() {
    const auto w = sample(static_cast<std::size_t>(kRows) * kInner, 3, 1.0);
    const auto x = sample(static_cast<std::size_t>(kInner) * kColumns, 4, 1.0);
    const auto whole = reference(w, x, kRows, kInner, kColumns);

    const auto placement = ninfer::artifact::split_evenly(ninfer::artifact::ShardAxis::Columns,
                                                          kInner, kWidth);
    std::vector<double> sum(whole.size(), 0.0);
    for (int rank = 0; rank < kWidth; ++rank) {
        const auto range = placement.ranges[static_cast<std::size_t>(rank)];
        // Rank `rank` computes a FULL-WIDTH partial over ONLY its own input columns.
        std::vector<double> shard_x(
            x.begin() + range.begin * kColumns,
            x.begin() + (range.begin + range.extent) * kColumns);
        std::vector<double> shard_w(static_cast<std::size_t>(kRows) * range.extent);
        for (int n = 0; n < kRows; ++n) {
            for (int k = 0; k < range.extent; ++k) {
                shard_w[static_cast<std::size_t>(n) * range.extent + k] =
                    w[static_cast<std::size_t>(n) * kInner + range.begin + k];
            }
        }
        const auto partial =
            reference(shard_w, shard_x, kRows, static_cast<int>(range.extent), kColumns);
        for (std::size_t i = 0; i < sum.size(); ++i) { sum[i] += partial[i]; }
    }
    check(equal_within(sum, whole, 1e-12),
          "column-parallel partials sum to the unsharded projection");
    std::cout << "  column-parallel: " << kInner << " inner -> " << placement.ranges[0].extent
              << " + " << placement.ranges[1].extent << ", exact\n";
}

// ------------------------------------------------------------------ 3. the residual rule
void check_residual_added_once() {
    const auto w = sample(static_cast<std::size_t>(kRows) * kInner, 5, 1.0);
    const auto x = sample(static_cast<std::size_t>(kInner) * kColumns, 6, 1.0);
    const auto residual = sample(static_cast<std::size_t>(kRows) * kColumns, 7, 0.5);
    const auto whole    = reference(w, x, kRows, kInner, kColumns);

    const auto placement = ninfer::artifact::split_evenly(ninfer::artifact::ShardAxis::Columns,
                                                          kInner, kWidth);
    std::vector<double> partial_sum(whole.size(), 0.0);
    for (int rank = 0; rank < kWidth; ++rank) {
        const auto range = placement.ranges[static_cast<std::size_t>(rank)];
        std::vector<double> shard_x(
            x.begin() + range.begin * kColumns,
            x.begin() + (range.begin + range.extent) * kColumns);
        std::vector<double> shard_w(static_cast<std::size_t>(kRows) * range.extent);
        for (int n = 0; n < kRows; ++n) {
            for (int k = 0; k < range.extent; ++k) {
                shard_w[static_cast<std::size_t>(n) * range.extent + k] =
                    w[static_cast<std::size_t>(n) * kInner + range.begin + k];
            }
        }
        const auto partial =
            reference(shard_w, shard_x, kRows, static_cast<int>(range.extent), kColumns);
        for (std::size_t i = 0; i < partial_sum.size(); ++i) { partial_sum[i] += partial[i]; }
    }

    // Correct: sum first, then add the replicated residual once.
    std::vector<double> after(partial_sum.size());
    for (std::size_t i = 0; i < after.size(); ++i) { after[i] = partial_sum[i] + residual[i]; }
    std::vector<double> want(whole.size());
    for (std::size_t i = 0; i < want.size(); ++i) { want[i] = whole[i] + residual[i]; }
    check(equal_within(after, want, 1e-12), "the residual added after the sum appears exactly once");

    // The trap: `linear_add` on both ranks, then the sum. Each rank's partial carries the residual,
    // so the composed result carries it twice.
    std::vector<double> doubled(want.size());
    for (std::size_t i = 0; i < doubled.size(); ++i) { doubled[i] = whole[i] + 2.0 * residual[i]; }
    const bool same_as_correct = equal_within(doubled, want, 1e-12);
    check(!same_as_correct, "adding the residual before the sum is detectably wrong");
    std::cout << "  residual: after-sum matches; before-sum differs by |residual| = "
              << std::fabs(residual[0]) << " at element 0\n";
}

// ------------------------------------------------------------------ 4. the BF16 storage cost
// The all-reduce contract is `RNE_bf16(fl32(partial_0) + fl32(partial_1))` over BF16 partials, so a
// column-parallel projection is NOT bit-identical to the unsharded one. This measures the gap
// against the FP64 oracle rather than asserting it away.
void check_bf16_storage_cost() {
    const auto w = sample(static_cast<std::size_t>(kRows) * kInner, 8, 1.0);
    const auto x = sample(static_cast<std::size_t>(kInner) * kColumns, 9, 1.0);
    const auto whole = reference(w, x, kRows, kInner, kColumns);

    const auto placement = ninfer::artifact::split_evenly(ninfer::artifact::ShardAxis::Columns,
                                                          kInner, kWidth);
    std::vector<double> composed(whole.size(), 0.0);
    for (int rank = 0; rank < kWidth; ++rank) {
        const auto range = placement.ranges[static_cast<std::size_t>(rank)];
        std::vector<double> shard_x(
            x.begin() + range.begin * kColumns,
            x.begin() + (range.begin + range.extent) * kColumns);
        std::vector<double> shard_w(static_cast<std::size_t>(kRows) * range.extent);
        for (int n = 0; n < kRows; ++n) {
            for (int k = 0; k < range.extent; ++k) {
                shard_w[static_cast<std::size_t>(n) * range.extent + k] =
                    w[static_cast<std::size_t>(n) * kInner + range.begin + k];
            }
        }
        const auto partial =
            reference(shard_w, shard_x, kRows, static_cast<int>(range.extent), kColumns);
        for (std::size_t i = 0; i < composed.size(); ++i) {
            composed[i] += static_cast<double>(to_bf16(static_cast<float>(partial[i])));
        }
    }
    for (double& value : composed) {
        value = static_cast<double>(to_bf16(static_cast<float>(value)));
    }
    const double worst = worst_relative(composed, whole);
    std::printf("  bf16 storage: worst relative error vs FP64 oracle %.3e\n", worst);
    // Two BF16 storage roundings on a partial of magnitude <= |whole| bound the error near 2^-8;
    // 1e-2 leaves room for cancellation while still failing a wrong composition.
    check(worst < 1.0e-2, "the two-storage-rounding cost stays within the BF16 bound");
}

} // namespace

int main() {
    std::printf("sharded forward composition (host, FP64 oracle)\n");
    check_row_parallel();
    check_column_parallel();
    check_residual_added_once();
    check_bf16_storage_cost();
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures,
                failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
