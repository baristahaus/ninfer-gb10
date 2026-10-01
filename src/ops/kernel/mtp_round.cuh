#pragma once

// Implements: include/ninfer/ops/mtp_round.h
// Match: request-major fixed K=1..5 autoregressive MTP round transition.

#include <cstdint>

namespace ninfer::ops {

__global__ void mtp_prepare_next_round_kernel(
    const std::int32_t* verify_ids, const std::int32_t* next_anchors, const std::int32_t* accepted,
    const std::int32_t* updated_frontiers, const std::int32_t* remaining_budgets,
    const std::int32_t* licensed_counts, const std::int32_t* rope_deltas,
    std::int32_t* alignment_ids, std::int32_t* next_extents, std::int32_t* ar_positions,
    std::int32_t* ar_rope_positions, std::int32_t* ar_valid_columns, std::int32_t k,
    std::int32_t ar_step_stride, std::int32_t max_context) {
    const int row = static_cast<int>(blockIdx.y);
    const int T   = k + 1;
    int a         = accepted[row];
    a             = a < 0 ? 0 : (a > k ? k : a);
    for (int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x; j < T;
         j += blockDim.x * gridDim.x) {
        alignment_ids[row * T + j] = j < a ? verify_ids[row * T + j + 1] : next_anchors[row];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int licensed       = licensed_counts[row];
        const int remaining      = remaining_budgets[row] - licensed;
        const int budget_extent  = remaining > 1 ? remaining - 1 : 0;
        const int context_extent = max_context - updated_frontiers[row] - 1;
        int next                 = budget_extent < context_extent ? budget_extent : context_extent;
        next                     = next < 0 ? 0 : (next > k ? k : next);
        next_extents[row]        = next;
        const int steps          = k > 1 ? k - 1 : 1;
        for (int s = 0; s < steps; ++s) {
            const int offset          = s * ar_step_stride + row;
            const int position        = updated_frontiers[row] + s;
            ar_positions[offset]      = position;
            ar_rope_positions[offset] = position + rope_deltas[row];
            ar_valid_columns[offset]  = s + 1 < next ? 1 : 0;
        }
    }
}

__global__ void
mtp_advance_round_kernel(const std::int32_t* anchors, const std::int32_t* frontiers,
                         const std::int32_t* licensed_counts, const std::int32_t* next_extents,
                         const std::int32_t* next_drafts, const std::int32_t* rope_deltas,
                         const std::int32_t* state_slots, std::int32_t* remaining_budgets,
                         std::int32_t* current_extents, std::int32_t* target_valid_columns,
                         std::int32_t* current_drafts, std::int32_t* target_rope_positions,
                         std::int32_t* pending_folds, const std::int32_t* verify_ids,
                         const std::int32_t* licensed_tokens, std::int32_t* ple_history,
                         std::int32_t batch, std::int32_t k, std::int32_t draft_step_stride) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= batch) { return; }
    const int licensed        = licensed_counts[row];
    int extent                = next_extents[row];
    extent                    = extent < 0 ? 0 : (extent > k ? k : extent);
    const int budget          = remaining_budgets[row] - licensed;
    remaining_budgets[row]    = budget > 0 ? budget : 0;
    current_extents[row]      = extent;
    target_valid_columns[row] = extent + 1;
    const int anchor          = anchors[row];
    for (int j = 0; j < k; ++j) {
        current_drafts[row * k + j] =
            j < extent ? next_drafts[j * draft_step_stride + row] : anchor;
    }
    const int rope_base = frontiers[row] + rope_deltas[row];
    for (int j = 0; j <= k; ++j) {
        target_rope_positions[row * (k + 1) + j] = rope_base + (j < extent ? j : extent);
    }
    const int slot             = state_slots[row];
    pending_folds[row * 4]     = slot;
    pending_folds[row * 4 + 1] = slot;
    pending_folds[row * 4 + 2] = licensed;
    pending_folds[row * 4 + 3] = 0;
    // The row's sequence ends h2, h1, previous anchor, licensed tokens; the new anchor is its
    // last licensed token.
    const int width        = k + 1;
    const int history1     = ple_history[row * 2];
    const int history2     = ple_history[row * 2 + 1];
    const int previous     = verify_ids[row * width];
    const auto sequence_at = [&](int i) {
        return i == 0   ? history2
               : i == 1 ? history1
               : i == 2 ? previous
                        : licensed_tokens[row * width + i - 3];
    };
    ple_history[row * 2]     = sequence_at(licensed + 1);
    ple_history[row * 2 + 1] = sequence_at(licensed);
}

} // namespace ninfer::ops
