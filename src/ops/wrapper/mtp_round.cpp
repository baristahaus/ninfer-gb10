#include "ninfer/ops/mtp_round.h"
#include "ops/launcher/mtp_round.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_contiguous_nonnull(const Tensor& t, const char* op, const char* name) {
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

void require_matrix(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                    const char* op, const char* name) {
    if (t.dtype != dtype || rows <= 0 || cols <= 0 || t.ne[0] != rows || t.ne[1] != cols ||
        t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid matrix shape for " + name);
    }
    require_contiguous_nonnull(t, op, name);
}

void require_vector(const Tensor& t, DType dtype, std::int32_t count, const char* op,
                    const char* name) {
    require_matrix(t, dtype, count, 1, op, name);
}

void require_row_pitched_matrix(const Tensor& t, std::int32_t rows, std::int32_t cols,
                                const char* op, const char* name) {
    constexpr std::int64_t element = sizeof(std::int32_t);
    if (rows <= 0 || cols <= 0 || t.dtype != DType::I32 || t.ne[0] != rows || t.ne[1] != cols ||
        t.ne[2] != 1 || t.ne[3] != 1 || t.nb[0] != element || t.nb[1] < rows * element ||
        t.nb[1] % element != 0 || t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": invalid row-pitched matrix for " + name);
    }
}

} // namespace

void mtp_prepare_next_round(const Tensor& verify_ids, const Tensor& next_anchors,
                            const Tensor& accepted, const Tensor& updated_frontiers,
                            const Tensor& remaining_budgets, const Tensor& licensed_counts,
                            const Tensor& rope_deltas, Tensor& alignment_ids, Tensor& next_extents,
                            Tensor& ar_positions, Tensor& ar_rope_positions,
                            Tensor& ar_valid_columns, std::int32_t max_context,
                            cudaStream_t stream) {
    constexpr const char* op = "mtp_prepare_next_round";
    const std::int32_t T     = verify_ids.ne[0];
    const std::int32_t batch = verify_ids.ne[1];
    if (T < 2 || T > 6) {
        throw std::invalid_argument("mtp_prepare_next_round: T must be in [2,6]");
    }
    if (batch < 1) { throw std::invalid_argument("mtp_prepare_next_round: B must be positive"); }
    if (max_context <= 0) {
        throw std::invalid_argument("mtp_prepare_next_round: max_context must be positive");
    }
    require_matrix(verify_ids, DType::I32, T, batch, op, "verify_ids");
    require_vector(next_anchors, DType::I32, batch, op, "next_anchors");
    require_vector(accepted, DType::I32, batch, op, "accepted");
    require_vector(updated_frontiers, DType::I32, batch, op, "updated_frontiers");
    require_vector(remaining_budgets, DType::I32, batch, op, "remaining_budgets");
    require_vector(licensed_counts, DType::I32, batch, op, "licensed_counts");
    require_vector(rope_deltas, DType::I32, batch, op, "rope_deltas");
    require_matrix(alignment_ids, DType::I32, T, batch, op, "alignment_ids");
    require_vector(next_extents, DType::I32, batch, op, "next_extents");
    const std::int32_t steps = std::max(T - 2, 1);
    require_row_pitched_matrix(ar_positions, batch, steps, op, "ar_positions");
    require_row_pitched_matrix(ar_rope_positions, batch, steps, op, "ar_rope_positions");
    require_row_pitched_matrix(ar_valid_columns, batch, steps, op, "ar_valid_columns");
    if (ar_rope_positions.nb[1] != ar_positions.nb[1] ||
        ar_valid_columns.nb[1] != ar_positions.nb[1]) {
        throw std::invalid_argument(
            "mtp_prepare_next_round: AR outputs must share one step stride");
    }
    detail::mtp_prepare_next_round_launch(verify_ids, next_anchors, accepted, updated_frontiers,
                                          remaining_budgets, licensed_counts, rope_deltas,
                                          alignment_ids, next_extents, ar_positions,
                                          ar_rope_positions, ar_valid_columns, max_context, stream);
}

void mtp_advance_round(const Tensor& anchors, const Tensor& frontiers,
                       const Tensor& licensed_counts, const Tensor& next_extents,
                       const Tensor& next_drafts, const Tensor& rope_deltas,
                       const Tensor& state_slots, Tensor& remaining_budgets,
                       Tensor& current_extents, Tensor& target_valid_columns,
                       Tensor& current_drafts, Tensor& target_rope_positions, Tensor& pending_folds,
                       cudaStream_t stream) {
    constexpr const char* op = "mtp_advance_round";
    const std::int32_t batch = anchors.ne[0];
    const std::int32_t k     = current_drafts.ne[0];
    if (batch < 1) { throw std::invalid_argument("mtp_advance_round: B must be positive"); }
    if (k < 1 || k > 5) { throw std::invalid_argument("mtp_advance_round: K must be in [1,5]"); }
    require_vector(anchors, DType::I32, batch, op, "anchors");
    require_vector(frontiers, DType::I32, batch, op, "frontiers");
    require_vector(licensed_counts, DType::I32, batch, op, "licensed_counts");
    require_vector(next_extents, DType::I32, batch, op, "next_extents");
    require_row_pitched_matrix(next_drafts, batch, k, op, "next_drafts");
    require_vector(rope_deltas, DType::I32, batch, op, "rope_deltas");
    require_vector(state_slots, DType::I32, batch, op, "state_slots");
    require_vector(remaining_budgets, DType::I32, batch, op, "remaining_budgets");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_vector(target_valid_columns, DType::I32, batch, op, "target_valid_columns");
    require_matrix(current_drafts, DType::I32, k, batch, op, "current_drafts");
    require_matrix(target_rope_positions, DType::I32, k + 1, batch, op, "target_rope_positions");
    require_matrix(pending_folds, DType::I32, 4, batch, op, "pending_folds");
    detail::mtp_advance_round_launch(anchors, frontiers, licensed_counts, next_extents, next_drafts,
                                     rope_deltas, state_slots, remaining_budgets, current_extents,
                                     target_valid_columns, current_drafts, target_rope_positions,
                                     pending_folds, stream);
}

} // namespace ninfer::ops
