#include "models/qwen3_5/execution/gdn.h"
#include "ninfer/ops/gdn_control.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_split.h"

#include <algorithm>

namespace ninfer::models::qwen3_5::execution {

std::size_t gdn_projection_workspace_bytes(const GdnProfile& parameters, std::int32_t first,
                                           std::int32_t last) {
    if (parameters.role_profiles) {
        // The two role dispatches run under their own scopes; the peak is the larger.
        std::size_t peak = 0;
        for (const auto& role : *parameters.role_profiles) {
            peak = std::max(peak, ops::linear_workspace_capacity_bytes(
                                      role.qtype, role.n, role.k, role.policy, first, last));
        }
        return peak;
    }
    if (is_exl3_gdn(parameters)) {
        const auto& packed = parameters.projection.first;
        return ops::gdn_input_proj_exl3_workspace_capacity_bytes(packed.n, packed.k, packed.policy,
                                                                 first, last);
    }
    return ops::linear_split_workspace_capacity_bytes(parameters.projection,
                                                      parameters.projection_rows, first, last);
}

std::size_t gdn_snapshot_workspace_bytes(const GdnProfile& parameters, std::int32_t batch,
                                         std::int32_t first_width, std::int32_t last_width) {
    return ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(parameters.projection, batch,
                                                                      first_width, last_width);
}

std::size_t gdn_record_workspace_bytes(const GdnProfile& parameters, std::int32_t batch,
                                       std::int32_t first_width, std::int32_t last_width) {
    return ops::gdn_input_proj_conv_record_workspace_capacity_bytes(parameters.projection, batch,
                                                                    first_width, last_width);
}

void gdn_projection(const Tensor& hidden, const GdnParameters& parameters, Tensor& qkv, Tensor& z,
                    MemoryArena& workspace, CudaExecutionView execution) {
    if (parameters.role_rows) {
        // Separate (qkv, z) parents (plan 4a): one linear per role into the same
        // planes the fused route produces.
        const auto& qkv_role = (*parameters.role_rows)[0];
        const auto& z_role   = (*parameters.role_rows)[1];
        {
            auto call = workspace.scope();
            ops::linear(hidden, qkv_role.weight, qkv, qkv_role.policy, workspace, execution);
        }
        {
            auto call = workspace.scope();
            ops::linear(hidden, z_role.weight, z, z_role.policy, workspace, execution);
        }
        return;
    }
    if (const auto* single = std::get_if<LinearParameters>(&parameters.projection);
        single != nullptr && single->weight.qtype == QType::EXL3) {
        ops::gdn_input_proj_exl3(hidden, single->weight, qkv, z, single->policy, workspace,
                                 execution.stream());
        return;
    }
    std::array outputs{qkv, z};
    ops::linear_split(hidden, parameters.projection, outputs, workspace, execution);
}

void gdn_norm_control(const Tensor& residual, const Tensor& norm, float epsilon,
                      const GdnParameters& parameters, Tensor& hidden, Tensor& g, Tensor& beta,
                      MemoryArena& workspace, CudaExecutionView execution) {
    ops::gdn_norm_control_proj(residual, norm, epsilon, parameters.control, parameters.a_log,
                               parameters.dt_bias, {hidden, {g, beta}}, workspace, execution);
}

void gdn_projection_snapshot(const Tensor& hidden, const GdnParameters& parameters,
                             Tensor& conv_states, const Tensor& valid_columns,
                             const Tensor& initial_slots, const Tensor& destination_slots,
                             Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                             MemoryArena& workspace, CudaExecutionView execution) {
    ops::gdn_input_proj_conv_snapshot(hidden, parameters.projection, parameters.convolution,
                                      {conv_states, initial_slots, destination_slots},
                                      valid_columns, {query, key, value, z}, workspace, execution);
}

void gdn_projection_record(const Tensor& hidden, const GdnParameters& parameters,
                           const Tensor& conv_states, const Tensor& valid_columns,
                           const Tensor& initial_slots, Tensor& conv_record, Tensor& query,
                           Tensor& key, Tensor& value, Tensor& z, MemoryArena& workspace,
                           CudaExecutionView execution) {
    ops::gdn_input_proj_conv_record(hidden, parameters.projection, parameters.convolution,
                                    {conv_states, initial_slots}, valid_columns,
                                    {query, key, value, z}, conv_record, workspace, execution);
}

} // namespace ninfer::models::qwen3_5::execution
