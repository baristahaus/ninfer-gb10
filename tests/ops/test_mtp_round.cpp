#include "ninfer/ops/mtp_round.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

int run_case(int k, const std::vector<std::int32_t>& accepted) {
    const int batch           = static_cast<int>(accepted.size());
    const int T               = k + 1;
    constexpr int max_context = 128;

    std::vector<std::int32_t> verify(static_cast<std::size_t>(T * batch));
    std::vector<std::int32_t> anchors(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> frontiers(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> budgets(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> licensed(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> rope_deltas(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> expected_alignment(static_cast<std::size_t>(T * batch));
    std::vector<std::int32_t> expected_extents(static_cast<std::size_t>(batch));
    const int steps = std::max(k - 1, 1);
    std::vector<std::int32_t> expected_positions(static_cast<std::size_t>(batch * steps));
    std::vector<std::int32_t> expected_rope_positions(static_cast<std::size_t>(batch * steps));
    std::vector<std::int32_t> expected_valid(static_cast<std::size_t>(batch * steps));
    for (int b = 0; b < batch; ++b) {
        anchors[static_cast<std::size_t>(b)]  = 90000 + 31 * b;
        licensed[static_cast<std::size_t>(b)] = accepted[static_cast<std::size_t>(b)] + 1;
        frontiers[static_cast<std::size_t>(b)] =
            20 + 17 * b + licensed[static_cast<std::size_t>(b)];
        budgets[static_cast<std::size_t>(b)] =
            b == batch - 1 ? licensed[static_cast<std::size_t>(b)] : 12 - b;
        rope_deltas[static_cast<std::size_t>(b)] = 3 * b - 2;
        for (int j = 0; j < T; ++j) {
            verify[static_cast<std::size_t>(b * T + j)] = 1000 + 101 * b + 7 * j;
        }
        for (int j = 0; j < T; ++j) {
            expected_alignment[static_cast<std::size_t>(b * T + j)] =
                j < accepted[static_cast<std::size_t>(b)]
                    ? verify[static_cast<std::size_t>(b * T + j + 1)]
                    : anchors[static_cast<std::size_t>(b)];
        }
        const int budget_extent = std::max(
            budgets[static_cast<std::size_t>(b)] - licensed[static_cast<std::size_t>(b)] - 1, 0);
        const int context_extent =
            std::max(max_context - frontiers[static_cast<std::size_t>(b)] - 1, 0);
        expected_extents[static_cast<std::size_t>(b)] =
            std::min({k, budget_extent, context_extent});
        for (int s = 0; s < steps; ++s) {
            const std::size_t offset   = static_cast<std::size_t>(s * batch + b);
            expected_positions[offset] = frontiers[static_cast<std::size_t>(b)] + s;
            expected_rope_positions[offset] =
                expected_positions[offset] + rope_deltas[static_cast<std::size_t>(b)];
            expected_valid[offset] = s + 1 < expected_extents[static_cast<std::size_t>(b)] ? 1 : 0;
        }
    }

    DeviceBuffer d_verify      = to_device(verify);
    DeviceBuffer d_anchors     = to_device(anchors);
    DeviceBuffer d_accepted    = to_device(accepted);
    DeviceBuffer d_frontiers   = to_device(frontiers);
    DeviceBuffer d_budgets     = to_device(budgets);
    DeviceBuffer d_licensed    = to_device(licensed);
    DeviceBuffer d_rope_deltas = to_device(rope_deltas);
    GuardedDeviceBuffer d_alignment(expected_alignment.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_extents(expected_extents.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_positions(expected_positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_rope_positions(expected_rope_positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_valid(expected_valid.size() * sizeof(std::int32_t));
    d_alignment.fill(0xcd);
    d_extents.fill(0xcd);
    d_positions.fill(0xcd);
    d_rope_positions.fill(0xcd);
    d_valid.fill(0xcd);

    Tensor t_verify(d_verify.p, DType::I32, {T, batch});
    Tensor t_anchors(d_anchors.p, DType::I32, {batch});
    Tensor t_accepted(d_accepted.p, DType::I32, {batch});
    Tensor t_frontiers(d_frontiers.p, DType::I32, {batch});
    Tensor t_budgets(d_budgets.p, DType::I32, {batch});
    Tensor t_licensed(d_licensed.p, DType::I32, {batch});
    Tensor t_rope_deltas(d_rope_deltas.p, DType::I32, {batch});
    Tensor t_alignment(d_alignment.data(), DType::I32, {T, batch});
    Tensor t_extents(d_extents.data(), DType::I32, {batch});
    Tensor t_positions(d_positions.data(), DType::I32, {batch, steps});
    Tensor t_rope_positions(d_rope_positions.data(), DType::I32, {batch, steps});
    Tensor t_valid(d_valid.data(), DType::I32, {batch, steps});
    ops::mtp_prepare_next_round(t_verify, t_anchors, t_accepted, t_frontiers, t_budgets, t_licensed,
                                t_rope_deltas, t_alignment, t_extents, t_positions,
                                t_rope_positions, t_valid, max_context, nullptr);
    cuda_synchronize();

    const std::string label =
        "mtp next round K=" + std::to_string(k) + " B=" + std::to_string(batch);
    int failures =
        verify_exact((label + " alignment").c_str(),
                     from_device<std::int32_t>(d_alignment.data(), expected_alignment.size()),
                     expected_alignment);
    failures += verify_exact((label + " next extents").c_str(),
                             from_device<std::int32_t>(d_extents.data(), expected_extents.size()),
                             expected_extents);
    failures +=
        verify_exact((label + " AR positions").c_str(),
                     from_device<std::int32_t>(d_positions.data(), expected_positions.size()),
                     expected_positions);
    failures += verify_exact(
        (label + " AR rope positions").c_str(),
        from_device<std::int32_t>(d_rope_positions.data(), expected_rope_positions.size()),
        expected_rope_positions);
    failures += verify_exact((label + " AR valid columns").c_str(),
                             from_device<std::int32_t>(d_valid.data(), expected_valid.size()),
                             expected_valid);
    failures += d_alignment.verify_guards((label + " alignment guards").c_str());
    failures += d_extents.verify_guards((label + " extent guards").c_str());
    failures += d_positions.verify_guards((label + " position guards").c_str());
    failures += d_rope_positions.verify_guards((label + " rope position guards").c_str());
    failures += d_valid.verify_guards((label + " valid guards").c_str());
    return failures;
}

// The frame for the next round, from the committed round's outputs. next_drafts is the exact-B
// prefix of a fixed-capacity [capacity,K] frame, so its step stride exceeds B.
int run_advance_case(int k, int batch, int capacity) {
    const int T = k + 1;
    std::vector<std::int32_t> anchors(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> frontiers(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> licensed(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> next_extents(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> next_drafts(static_cast<std::size_t>(capacity * k));
    std::vector<std::int32_t> rope_deltas(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> slots(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> budgets(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> expected_budgets(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> expected_extents(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> expected_valid(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> expected_drafts(static_cast<std::size_t>(k * batch));
    std::vector<std::int32_t> expected_rope(static_cast<std::size_t>(T * batch));
    std::vector<std::int32_t> expected_folds(static_cast<std::size_t>(4 * batch));
    std::vector<std::int32_t> verify(static_cast<std::size_t>(T * batch));
    std::vector<std::int32_t> licensed_tokens(static_cast<std::size_t>(T * batch));
    std::vector<std::int32_t> history(static_cast<std::size_t>(2 * batch));
    std::vector<std::int32_t> expected_history(static_cast<std::size_t>(2 * batch));
    for (int s = 0; s < k; ++s) {
        for (int b = 0; b < capacity; ++b) {
            next_drafts[static_cast<std::size_t>(s * capacity + b)] = 50000 + 100 * b + s;
        }
    }
    for (int b = 0; b < batch; ++b) {
        const auto i   = static_cast<std::size_t>(b);
        anchors[i]     = 90000 + 31 * b;
        licensed[i]    = 1 + b % T;
        frontiers[i]   = 40 + 23 * b;
        rope_deltas[i] = 5 * b - 7;
        slots[i]       = 3 + 2 * b;
        // Cover an exhausted budget, a zero extent and the full extent.
        budgets[i]      = b == 0 ? licensed[i] - 1 : 20 + b;
        next_extents[i] = b % (k + 1);
        if (b == batch - 1) { next_extents[i] = k; }

        const int extent    = std::clamp(next_extents[i], 0, k);
        expected_budgets[i] = std::max(budgets[i] - licensed[i], 0);
        expected_extents[i] = extent;
        expected_valid[i]   = extent + 1;
        for (int j = 0; j < k; ++j) {
            expected_drafts[static_cast<std::size_t>(b * k + j)] =
                j < extent ? next_drafts[static_cast<std::size_t>(j * capacity + b)] : anchors[i];
        }
        for (int j = 0; j < T; ++j) {
            expected_rope[static_cast<std::size_t>(b * T + j)] =
                frontiers[i] + std::min(j, extent) + rope_deltas[i];
        }
        expected_folds[static_cast<std::size_t>(4 * b)]     = slots[i];
        expected_folds[static_cast<std::size_t>(4 * b + 1)] = slots[i];
        expected_folds[static_cast<std::size_t>(4 * b + 2)] = licensed[i];
        expected_folds[static_cast<std::size_t>(4 * b + 3)] = 0;

        // Row 0 starts at the sequence start (no history); the others carry two tokens.
        history[static_cast<std::size_t>(2 * b)]     = b == 0 ? -1 : 70000 + b;
        history[static_cast<std::size_t>(2 * b + 1)] = b == 0 ? -1 : 71000 + b;
        for (int j = 0; j < T; ++j) {
            verify[static_cast<std::size_t>(b * T + j)]          = 80000 + 10 * b + j;
            licensed_tokens[static_cast<std::size_t>(b * T + j)] = 60000 + 10 * b + j;
        }
        std::vector<std::int32_t> sequence = {history[static_cast<std::size_t>(2 * b + 1)],
                                              history[static_cast<std::size_t>(2 * b)],
                                              verify[static_cast<std::size_t>(b * T)]};
        for (int j = 0; j < licensed[i]; ++j) {
            sequence.push_back(licensed_tokens[static_cast<std::size_t>(b * T + j)]);
        }
        expected_history[static_cast<std::size_t>(2 * b)]     = sequence[sequence.size() - 2];
        expected_history[static_cast<std::size_t>(2 * b + 1)] = sequence[sequence.size() - 3];
    }

    DeviceBuffer d_anchors      = to_device(anchors);
    DeviceBuffer d_frontiers    = to_device(frontiers);
    DeviceBuffer d_licensed     = to_device(licensed);
    DeviceBuffer d_next_extents = to_device(next_extents);
    DeviceBuffer d_next_drafts  = to_device(next_drafts);
    DeviceBuffer d_rope_deltas  = to_device(rope_deltas);
    DeviceBuffer d_slots        = to_device(slots);
    DeviceBuffer d_budgets      = to_device(budgets);
    DeviceBuffer d_verify       = to_device(verify);
    DeviceBuffer d_licensed_tok = to_device(licensed_tokens);
    DeviceBuffer d_history      = to_device(history);
    GuardedDeviceBuffer d_extents(expected_extents.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_valid(expected_valid.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_drafts(expected_drafts.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_rope(expected_rope.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_folds(expected_folds.size() * sizeof(std::int32_t));
    d_extents.fill(0xcd);
    d_valid.fill(0xcd);
    d_drafts.fill(0xcd);
    d_rope.fill(0xcd);
    d_folds.fill(0xcd);

    Tensor t_anchors(d_anchors.p, DType::I32, {batch});
    Tensor t_frontiers(d_frontiers.p, DType::I32, {batch});
    Tensor t_licensed(d_licensed.p, DType::I32, {batch});
    Tensor t_next_extents(d_next_extents.p, DType::I32, {batch});
    Tensor t_next_drafts = Tensor(d_next_drafts.p, DType::I32, {capacity, k}).slice(0, 0, batch);
    Tensor t_rope_deltas(d_rope_deltas.p, DType::I32, {batch});
    Tensor t_slots(d_slots.p, DType::I32, {batch});
    Tensor t_budgets(d_budgets.p, DType::I32, {batch});
    Tensor t_extents(d_extents.data(), DType::I32, {batch});
    Tensor t_valid(d_valid.data(), DType::I32, {batch});
    Tensor t_drafts(d_drafts.data(), DType::I32, {k, batch});
    Tensor t_rope(d_rope.data(), DType::I32, {T, batch});
    Tensor t_folds(d_folds.data(), DType::I32, {4, batch});
    Tensor t_verify(d_verify.p, DType::I32, {T, batch});
    Tensor t_licensed_tokens(d_licensed_tok.p, DType::I32, {T, batch});
    Tensor t_history(d_history.p, DType::I32, {2, batch});
    ops::mtp_advance_round(t_anchors, t_frontiers, t_licensed, t_next_extents, t_next_drafts,
                           t_rope_deltas, t_slots, t_budgets, t_extents, t_valid, t_drafts, t_rope,
                           t_folds, t_verify, t_licensed_tokens, t_history, nullptr);
    cuda_synchronize();

    const std::string label =
        "mtp advance round K=" + std::to_string(k) + " B=" + std::to_string(batch);
    int failures = verify_exact((label + " budgets").c_str(),
                                from_device<std::int32_t>(d_budgets.p, expected_budgets.size()),
                                expected_budgets);
    failures += verify_exact((label + " extents").c_str(),
                             from_device<std::int32_t>(d_extents.data(), expected_extents.size()),
                             expected_extents);
    failures += verify_exact((label + " valid columns").c_str(),
                             from_device<std::int32_t>(d_valid.data(), expected_valid.size()),
                             expected_valid);
    failures += verify_exact((label + " drafts").c_str(),
                             from_device<std::int32_t>(d_drafts.data(), expected_drafts.size()),
                             expected_drafts);
    failures +=
        verify_exact((label + " rope positions").c_str(),
                     from_device<std::int32_t>(d_rope.data(), expected_rope.size()), expected_rope);
    failures += verify_exact((label + " pending folds").c_str(),
                             from_device<std::int32_t>(d_folds.data(), expected_folds.size()),
                             expected_folds);
    failures += verify_exact((label + " PLE history").c_str(),
                             from_device<std::int32_t>(d_history.p, expected_history.size()),
                             expected_history);
    failures += d_extents.verify_guards((label + " extent guards").c_str());
    failures += d_valid.verify_guards((label + " valid guards").c_str());
    failures += d_drafts.verify_guards((label + " draft guards").c_str());
    failures += d_rope.verify_guards((label + " rope guards").c_str());
    failures += d_folds.verify_guards((label + " fold guards").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "mtp_round: SKIP (CUDA unavailable)\n";
        return 77;
    }

    int failures = 0;
    failures += run_case(1, {0});
    failures += run_case(5, {0, 2, 5});
    failures += run_advance_case(1, 1, 1);
    failures += run_advance_case(3, 4, 8);
    failures += run_advance_case(5, 8, 8);

    if (failures != 0) {
        std::cerr << "mtp_round failures=" << failures << '\n';
        return 1;
    }
    std::cout << "mtp_round: PASS\n";
    return 0;
}
