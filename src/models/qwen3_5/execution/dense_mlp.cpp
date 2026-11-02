#include "models/qwen3_5/execution/dense_mlp.h"

#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {

std::size_t dense_mlp_workspace_bytes(const DenseProfile& parameters, std::int32_t first,
                                      std::int32_t last, DenseMlpUse use) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("Dense MLP: invalid column interval");
    }
    const auto& gu   = parameters.gate_up;
    const auto& down = parameters.down;
    WorkspaceLayoutBuilder layout;
    if (parameters.gate_up_role_profiles) {
        const auto& gate = (*parameters.gate_up_role_profiles)[0];
        const auto& up   = (*parameters.gate_up_role_profiles)[1];
        // The runtime holds gate_out, up_out and the SiLU activation simultaneously.
        (void)layout.alloc(DType::BF16, {gate.n, last});
        (void)layout.alloc(DType::BF16, {up.n, last});
        (void)layout.alloc(DType::BF16, {gate.n, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
                gate.qtype, gate.n, gate.k, gate.policy, first, last));
        }
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
                up.qtype, up.n, up.k, up.policy, first, last));
        }
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            down.qtype, down.n, down.k, parameters.down.policy, first, last));
        return layout.peak_bytes(1);
    }
    if (use == DenseMlpUse::Mtp) {
        (void)layout.alloc(DType::BF16, {gu.n, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, parameters.gate_up.policy, first, last));
        }
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        (void)layout.alloc(DType::BF16, {down.n, last});
        (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
            down.qtype, down.n, down.k, parameters.down.policy, first, last));
    } else {
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, parameters.gate_up.policy, first, last));
        }
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
                down.qtype, down.n, down.k, parameters.down.policy, first, last));
        }
    }
    return layout.peak_bytes(1);
}

void dense_mlp(const Tensor& hidden, const DenseParameters& parameters, Tensor& residual,
               DenseMlpUse use, MemoryArena& workspace, CudaExecutionView execution) {
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    const auto& gu     = parameters.gate_up.weight;
    const auto& down   = parameters.down.weight;
    if (parameters.gate_up_roles) {
        const auto& gate = (*parameters.gate_up_roles)[0];
        const auto& up   = (*parameters.gate_up_roles)[1];
        Tensor gate_out = workspace.alloc(DType::BF16, {gate.weight.shape[0], columns});
        Tensor up_out   = workspace.alloc(DType::BF16, {up.weight.shape[0], columns});
        {
            auto call = workspace.scope();
            ops::linear(hidden, gate.weight, gate_out, gate.policy, workspace, execution);
        }
        {
            auto call = workspace.scope();
            ops::linear(hidden, up.weight, up_out, up.policy, workspace, execution);
        }
        Tensor activation =
            workspace.alloc(DType::BF16, {gate.weight.shape[0], columns});
        ops::silu_mul(gate_out, up_out, activation, execution);
        ops::linear_add(activation, down, residual, parameters.down.policy, workspace,
                        execution);
        return;
    }
    if (use == DenseMlpUse::Mtp) {
        Tensor gate_up = workspace.alloc(DType::BF16, {gu.n, columns});
        {
            auto call = workspace.scope();
            ops::linear(hidden, gu, gate_up, parameters.gate_up.policy, workspace, execution);
        }
        Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
        ops::silu_mul(gate_up.slice(0, 0, gu.n / 2), gate_up.slice(0, gu.n / 2, gu.n / 2),
                      activation, execution);
        Tensor delta = workspace.alloc(DType::BF16, {down.n, columns});
        ops::linear(activation, down, delta, parameters.down.policy, workspace, execution);
        ops::residual_add(delta, residual, execution);
        return;
    }
    Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
    {
        auto call = workspace.scope();
        ops::linear_swiglu(hidden, gu, activation, parameters.gate_up.policy, workspace, execution);
    }
    ops::linear_add(activation, down, residual, parameters.down.policy, workspace, execution);
}

} // namespace ninfer::models::qwen3_5::execution
