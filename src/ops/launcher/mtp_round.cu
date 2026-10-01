// Implements: include/ninfer/ops/mtp_round.h
// Match: validated request-major K=1..5 MTP round transition and frame advance.
#include "ops/launcher/mtp_round.h"

#include "core/device.h"
#include "ops/kernel/mtp_round.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

void mtp_prepare_next_round_launch(const Tensor& verify_ids, const Tensor& next_anchors,
                                   const Tensor& accepted, const Tensor& updated_frontiers,
                                   const Tensor& remaining_budgets, const Tensor& licensed_counts,
                                   const Tensor& rope_deltas, Tensor& alignment_ids,
                                   Tensor& next_extents, Tensor& ar_positions,
                                   Tensor& ar_rope_positions, Tensor& ar_valid_columns,
                                   std::int32_t max_context, cudaStream_t stream) {
    constexpr int kBlock = 32;
    const int k          = verify_ids.ne[0] - 1;
    const int batch      = verify_ids.ne[1];
    const int ar_step_stride =
        static_cast<int>(ar_positions.nb[1] / static_cast<std::int64_t>(sizeof(std::int32_t)));
    const dim3 grid(static_cast<unsigned int>((k + kBlock) / kBlock),
                    static_cast<unsigned int>(batch));
    mtp_prepare_next_round_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(verify_ids.data),
        static_cast<const std::int32_t*>(next_anchors.data),
        static_cast<const std::int32_t*>(accepted.data),
        static_cast<const std::int32_t*>(updated_frontiers.data),
        static_cast<const std::int32_t*>(remaining_budgets.data),
        static_cast<const std::int32_t*>(licensed_counts.data),
        static_cast<const std::int32_t*>(rope_deltas.data),
        static_cast<std::int32_t*>(alignment_ids.data),
        static_cast<std::int32_t*>(next_extents.data),
        static_cast<std::int32_t*>(ar_positions.data),
        static_cast<std::int32_t*>(ar_rope_positions.data),
        static_cast<std::int32_t*>(ar_valid_columns.data), k, ar_step_stride, max_context);
    CUDA_CHECK(cudaGetLastError());
}

void mtp_advance_round_launch(const Tensor& anchors, const Tensor& frontiers,
                              const Tensor& licensed_counts, const Tensor& next_extents,
                              const Tensor& next_drafts, const Tensor& rope_deltas,
                              Tensor& state_source_slots, Tensor& state_destination_slots,
                              Tensor& remaining_budgets, Tensor& current_extents,
                              Tensor& target_valid_columns, Tensor& current_drafts,
                              Tensor& target_rope_positions, Tensor& pending_folds,
                              const Tensor& verify_ids, const Tensor& licensed_tokens,
                              Tensor& ple_history, cudaStream_t stream) {
    constexpr int kBlock = 32;
    const int batch      = anchors.ne[0];
    const int k          = current_drafts.ne[0];
    const int draft_step_stride =
        static_cast<int>(next_drafts.nb[1] / static_cast<std::int64_t>(sizeof(std::int32_t)));
    const dim3 grid(static_cast<unsigned int>((batch + kBlock - 1) / kBlock));
    mtp_advance_round_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const std::int32_t*>(frontiers.data),
        static_cast<const std::int32_t*>(licensed_counts.data),
        static_cast<const std::int32_t*>(next_extents.data),
        static_cast<const std::int32_t*>(next_drafts.data),
        static_cast<const std::int32_t*>(rope_deltas.data),
        static_cast<std::int32_t*>(state_source_slots.data),
        static_cast<std::int32_t*>(state_destination_slots.data),
        static_cast<std::int32_t*>(remaining_budgets.data),
        static_cast<std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(target_valid_columns.data),
        static_cast<std::int32_t*>(current_drafts.data),
        static_cast<std::int32_t*>(target_rope_positions.data),
        static_cast<std::int32_t*>(pending_folds.data),
        static_cast<const std::int32_t*>(verify_ids.data),
        static_cast<const std::int32_t*>(licensed_tokens.data),
        static_cast<std::int32_t*>(ple_history.data), batch, k, draft_step_stride);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
