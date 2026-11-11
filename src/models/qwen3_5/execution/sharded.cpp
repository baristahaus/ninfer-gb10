#include "models/qwen3_5/execution/sharded.h"

#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::execution {
namespace {

// The two-rank lane is the adopted scope (include/ninfer/ops/allreduce.h), so a width other than 2
// is a caller error rather than something to fall back from: silently running an unsharded
// projection at width 2 would give a plausible-looking wrong answer.
void require_two_ranks(const ShardedCollective& collective) {
    if (collective.execution.width() != 2) {
        throw std::invalid_argument("sharded forward: the collective requires exactly two ranks");
    }
    if (collective.rank < 0 || collective.rank >= 2) {
        throw std::invalid_argument("sharded forward: the collective's rank is outside [0,2)");
    }
}

// Both partials must be the same full-width shape on their own devices, because the reduce reads
// and writes both. The peer's tensor is not this rank's to allocate, so this is a checked
// precondition rather than something the helper can fix.
void require_partial_pair(std::array<Tensor, 2>& partials, const ExecutionContext& execution,
                          std::int32_t rows, std::int32_t columns) {
    for (int rank = 0; rank < 2; ++rank) {
        const Tensor& partial = partials[static_cast<std::size_t>(rank)];
        if (partial.dtype != DType::BF16 || partial.ne[0] != rows || partial.ne[1] != columns ||
            partial.ne[2] != 1 || partial.ne[3] != 1) {
            throw std::invalid_argument(
                "sharded forward: a column-parallel partial is not BF16 [rows, columns]");
        }
        if (partial.device != execution.device(rank).device()) {
            throw std::invalid_argument(
                "sharded forward: a column-parallel partial is not on its own rank's device");
        }
    }
    if (partials[0].data == partials[1].data) {
        throw std::invalid_argument("sharded forward: the two ranks must not share one partial");
    }
}

// This rank's half of the cross-rank sum of the two full-width partials, through the collective's
// per-rank form: one slot and one exchange, on this rank's own stream, after the projection that
// feeds it. The peer's half is enqueued by the peer's own Program on its own stream.
void reduce_partials(std::array<Tensor, 2>& partials, const ShardedCollective& collective) {
    std::array<void*, 2> data{partials[0].data, partials[1].data};
    const std::size_t bytes = partials[0].bytes();
    if (bytes != partials[1].bytes()) {
        throw std::invalid_argument("sharded forward: the two ranks' partials differ in size");
    }
    if (bytes == 0 || bytes % 16 != 0) {
        throw std::invalid_argument(
            "sharded forward: a column-parallel partial must be a nonzero multiple of 16 bytes");
    }
    if (bytes > collective.mailbox.max_bytes()) {
        throw std::invalid_argument("sharded forward: the partial exceeds the mailbox slot");
    }
    collective.reduce(data, bytes);
}

} // namespace

LogitsGatherPlan plan_logits_gather(std::int32_t vocabulary, std::int32_t rank_rows, int rank,
                                    std::int32_t columns, std::size_t slot_bytes) {
    if (vocabulary <= 0 || rank_rows <= 0 || columns <= 0) {
        throw std::invalid_argument(
            "logits gather: the extents must be positive; got vocabulary=" +
            std::to_string(vocabulary) + " rank_rows=" + std::to_string(rank_rows) +
            " columns=" + std::to_string(columns));
    }
    // The gather is the concatenation of exactly two rank halves, so the full buffer has to be two
    // of them. A vocabulary that is not is a shard table and a head that disagree.
    if (vocabulary != 2 * rank_rows) {
        throw std::invalid_argument(
            "logits gather: the vocabulary is not two rank halves; vocabulary=" +
            std::to_string(vocabulary) + " rank_rows=" + std::to_string(rank_rows));
    }
    if (rank < 0 || rank >= 2) {
        throw std::invalid_argument("logits gather: the rank is outside [0,2): " +
                                    std::to_string(rank));
    }
    const std::size_t bytes_per_column = static_cast<std::size_t>(vocabulary) * sizeof(std::uint16_t);
    // One column is the smallest payload the gather can exchange, and the exchange carries whole
    // 16-byte units only.
    if (bytes_per_column % 16 != 0) {
        throw std::invalid_argument(
            "logits gather: one column of the full buffer is " + std::to_string(bytes_per_column) +
            " bytes, not a multiple of the exchange's 16-byte unit; vocabulary=" +
            std::to_string(vocabulary));
    }
    if (bytes_per_column > slot_bytes) {
        throw std::invalid_argument(
            "logits gather: one column of the full buffer does not fit the mailbox slot; column "
            "payload=" +
            std::to_string(bytes_per_column) + " bytes, slot=" + std::to_string(slot_bytes) +
            " bytes, vocabulary=" + std::to_string(vocabulary));
    }
    LogitsGatherPlan plan;
    plan.vocabulary       = vocabulary;
    plan.rank_rows        = rank_rows;
    plan.columns          = columns;
    // One exchange carries as many whole columns as the slot holds, but never more than the gather
    // itself has: a chunk wider than the call would report a payload no exchange carries.
    const auto per_slot = static_cast<std::int32_t>(slot_bytes / bytes_per_column);
    plan.chunk_columns  = std::min(per_slot, columns);
    plan.exchanges      = (columns + plan.chunk_columns - 1) / plan.chunk_columns;
    plan.bytes_per_column = bytes_per_column;
    plan.chunk_bytes      = bytes_per_column * static_cast<std::size_t>(plan.chunk_columns);
    return plan;
}

void all_gather_logits(const models::ShardedGroup& group, int rank, std::size_t& slot,
                       const Tensor& local, const Tensor& full, CudaExecutionView execution) {
    if (group.execution == nullptr || group.mailbox == nullptr) {
        throw std::invalid_argument("logits gather: the sharded group bundle is not built");
    }
    if (group.execution->width() != 2) {
        throw std::invalid_argument("logits gather: the gather requires exactly two ranks");
    }
    if (group.logits[0].data == nullptr || group.logits[1].data == nullptr) {
        throw std::logic_error(
            "logits gather: the group has no logits pair; size it at construction (output_rows, "
            "logits_columns)");
    }
    if (rank < 0 || rank >= 2) {
        throw std::invalid_argument("logits gather: the rank is outside [0,2): " +
                                    std::to_string(rank));
    }
    if (execution.device() != group.execution->device(rank).device()) {
        throw std::invalid_argument("logits gather: the view is not this rank's device");
    }
    // The destination must be this rank's member of the group pair: the zero-fill and the copy write
    // `full`, the exchange writes BOTH members of the pair, and if `full` were a different buffer the
    // copy and the sum would land in different places.
    if (full.data != group.logits[static_cast<std::size_t>(rank)].data) {
        throw std::invalid_argument(
            "logits gather: the destination is not this rank's member of the group's logits pair");
    }
    if (full.dtype != DType::BF16 || local.dtype != DType::BF16 || full.ne[2] != 1 ||
        full.ne[3] != 1 || local.ne[2] != 1 || local.ne[3] != 1) {
        throw std::invalid_argument("logits gather: the operands must be rank-2 BF16");
    }
    const std::int32_t columns = local.ne[1];
    if (full.ne[0] != group.logits[0].ne[0] || local.ne[0] != full.ne[0] / 2) {
        throw std::invalid_argument(
            "logits gather: the local half is not half the destination's rows; local=" +
            std::to_string(local.ne[0]) + " destination=" + std::to_string(full.ne[0]));
    }
    if (columns > full.ne[1]) {
        throw std::invalid_argument(
            "logits gather: the gather has more columns than the pair holds; columns=" +
            std::to_string(columns) + " pair columns=" + std::to_string(full.ne[1]));
    }
    const auto plan = plan_logits_gather(full.ne[0], local.ne[0], rank, columns,
                                         group.mailbox->max_bytes());
    auto* const destination      = static_cast<std::uint8_t*>(full.data);
    const auto* const source     = static_cast<const std::uint8_t*>(local.data);
    const std::size_t row_bytes  = static_cast<std::size_t>(plan.rank_rows) * sizeof(std::uint16_t);
    std::array<void*, 2> pair{group.logits[0].data, group.logits[1].data};
    for (std::int32_t chunk = 0; chunk < plan.exchanges; ++chunk) {
        const std::int32_t first = chunk * plan.chunk_columns;
        const std::int32_t count = std::min(plan.chunk_columns, columns - first);
        const std::size_t bytes =
            plan.bytes_per_column * static_cast<std::size_t>(count);
        auto* const base = destination + static_cast<std::size_t>(first) * plan.bytes_per_column;
        // Zero the whole slice first: every row outside this rank's half must be an exact BF16 zero
        // for the sum to be the concatenation. The copy below then writes this rank's half back, so
        // it has to come after — the zero-fill must not be the last thing that touches it.
        CUDA_CHECK(cudaMemsetAsync(base, 0, bytes, execution.stream()));
        auto* const half = base + static_cast<std::size_t>(rank) * row_bytes;
        for (std::int32_t column = 0; column < count; ++column) {
            // One contiguous copy per column: dim 0 is fastest, so a column's rows are contiguous
            // in both buffers, and the destination's column stride is the full vocabulary.
            CUDA_CHECK(cudaMemcpyAsync(half + static_cast<std::size_t>(column) * plan.bytes_per_column,
                                       source + static_cast<std::size_t>(first + column) * row_bytes,
                                       row_bytes, cudaMemcpyDeviceToDevice, execution.stream()));
        }
        ops::allreduce_sum_rank(*group.execution, *group.mailbox, rank,
                                group.mailbox->slot_for(slot, bytes), pair, bytes);
        ++slot;
    }
}

void sharded_project_columns(const Tensor& input, const LinearParameters& p,
                             std::array<Tensor, 2>& partials,
                             const ShardedCollective& collective, MemoryArena& workspace,
                             CudaExecutionView execution) {
    require_two_ranks(collective);
    const int rank = collective.rank;
    require_partial_pair(partials, collective.execution, p.weight.n, input.ne[1]);
    if (execution.device() != collective.execution.device(rank).device()) {
        throw std::invalid_argument("sharded forward: the view is not this rank's device");
    }
    {
        // No residual here: the projection is a partial sum over this rank's input columns, so the
        // fused `linear_add` would add the replicated residual before the cross-rank sum and count
        // it twice. See the header.
        auto scope = workspace.scope();
        ops::linear(input, p.weight, partials[static_cast<std::size_t>(rank)], p.policy, workspace,
                    execution);
    }
    reduce_partials(partials, collective);
}

void sharded_project_columns_add(const Tensor& input, const LinearParameters& p,
                                 std::array<Tensor, 2>& partials, Tensor& residual,
                                 const ShardedCollective& collective, MemoryArena& workspace,
                                 CudaExecutionView execution) {
    sharded_project_columns(input, p, partials, collective, workspace, execution);
    // The sum is complete on this rank's stream, so the residual is added exactly once, after it.
    ops::residual_add(partials[static_cast<std::size_t>(collective.rank)], residual, execution);
}

void sharded_dense_mlp(const Tensor& hidden, const DenseParameters& parameters, Tensor& residual,
                       std::array<Tensor, 2>& partials, const ShardedCollective& collective,
                       MemoryArena& workspace, CudaExecutionView execution, DenseMlpUse use) {
    require_two_ranks(collective);
    const std::int32_t columns = hidden.ne[1];

    // Row-parallel gate and up: each rank owns a contiguous block of the intermediate rows, so the
    // two intermediates below are this rank's alone and nothing is communicated.
    const auto intermediate_rows = [&](const ops::SingleProjectionWeight& w) {
        return w.weight.shape[0];
    };
    if (parameters.gate_up_roles) {
        const auto& gate = (*parameters.gate_up_roles)[0];
        const auto& up   = (*parameters.gate_up_roles)[1];
        Tensor gate_out  = workspace.alloc(DType::BF16, {intermediate_rows(gate), columns});
        Tensor up_out    = workspace.alloc(DType::BF16, {intermediate_rows(up), columns});
        {
            auto call = workspace.scope();
            ops::linear(hidden, gate.weight, gate_out, gate.policy, workspace, execution);
        }
        {
            auto call = workspace.scope();
            ops::linear(hidden, up.weight, up_out, up.policy, workspace, execution);
        }
        // Elementwise over this rank's own intermediate rows: no communication, and the SiLU is
        // applied to the same values the unsharded pass would apply it to.
        Tensor activation = workspace.alloc(DType::BF16, {intermediate_rows(gate), columns});
        ops::silu_mul(gate_out, up_out, activation, execution);
        sharded_project_columns_add(activation, parameters.down, partials, residual, collective,
                                    workspace, execution);
        return;
    }

    const auto& gu = parameters.gate_up.weight;
    if (use == DenseMlpUse::Mtp) {
        // The MTP stem's gate_up is a plain parent rather than a SwiGLU pair, so the single-device
        // lane projects it, slices gate from up, and applies SiLU itself. The sharded path does the
        // same over this rank's own intermediate rows.
        Tensor gate_up = workspace.alloc(DType::BF16, {gu.n, columns});
        {
            auto call = workspace.scope();
            ops::linear(hidden, gu, gate_up, parameters.gate_up.policy, workspace, execution);
        }
        Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
        ops::silu_mul(gate_up.slice(0, 0, gu.n / 2), gate_up.slice(0, gu.n / 2, gu.n / 2),
                      activation, execution);
        sharded_project_columns_add(activation, parameters.down, partials, residual, collective,
                                    workspace, execution);
        return;
    }
    // The fused gate_up parent keeps gate rows first and then up rows, so a rank's block of output
    // rows is NOT [gate_half | up_half] — it is a slice of the concatenated [2*I] rows. The rank's
    // slice is therefore handled by the same fused `linear_swiglu` the single-device lane uses, and
    // its output is this rank's own intermediate rows.
    Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
    {
        auto call = workspace.scope();
        ops::linear_swiglu(hidden, gu, activation, parameters.gate_up.policy, workspace, execution);
    }
    sharded_project_columns_add(activation, parameters.down, partials, residual, collective,
                                workspace, execution);
}

} // namespace ninfer::models::qwen3_5::execution
