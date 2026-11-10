#include "models/qwen3_5/execution/sharded.h"

#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"

#include <algorithm>
#include <stdexcept>

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
