#include "models/qwen3_5/execution/attention.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/rope.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

void require_rope_axes(const Tensor& positions, const RopeConfig& config) {
    if (positions.ne[1] != 3) { return; }
    for (std::size_t i = 0; i < config.pair_axes.size(); ++i) {
        if (config.pair_axes[i] != i % 3) {
            throw std::invalid_argument("text RoPE: this MRoPE axis mapping has no native route");
        }
    }
}

} // namespace

workspace::AttentionExtents attention_extents(const AttentionParameters& parameters,
                                             const TextConfig& config) {
    workspace::AttentionExtents out;
    if (parameters.role_rows) {
        const auto& roles = *parameters.role_rows;
        out.query_width   = roles[0].weight.n;
        out.key_width     = roles[1].weight.n;
        out.gate_width    = roles[2].weight.n;
        out.value_width   = roles[3].weight.n;
    } else {
        out.query_width = parameters.projection_rows[0];
        out.key_width   = parameters.projection_rows[1];
        // The fused form has no separate gate: the gate is the query's own rows.
        out.gate_width  = parameters.projection_rows[0];
        out.value_width = parameters.projection_rows[3];
    }
    out.head_dim    = static_cast<std::int32_t>(config.attention->head_dim);
    out.hidden_size = static_cast<std::int32_t>(config.hidden_size);
    if (out.head_dim <= 0 || out.query_width % out.head_dim != 0 ||
        out.key_width % out.head_dim != 0) {
        throw std::invalid_argument("attention extents are not a whole number of heads");
    }
    out.query_heads = out.query_width / out.head_dim;
    out.kv_heads    = out.key_width / out.head_dim;
    return out;
}

std::size_t attention_projection_workspace_bytes(const AttentionProfile& parameters,
                                                 std::int32_t first, std::int32_t last) {
    if (parameters.role_profiles) {
        // Per-role dispatches run under their own scopes, so the peak is the largest
        // role's capacity, not the sum.
        std::size_t peak = 0;
        for (const auto& role : *parameters.role_profiles) {
            peak = std::max(peak, ops::linear_workspace_capacity_bytes(
                                      role.qtype, role.n, role.k, role.policy, first, last,
                                      "attention_role"));
        }
        return peak;
    }
    if (const auto& first_profile = parameters.projection.first;
        !parameters.projection.second && first_profile.qtype == QType::EXL3) {
        return ops::attn_input_proj_workspace_capacity_bytes(first_profile.qtype, first_profile.n,
                                                             first_profile.k, first_profile.policy,
                                                             first, last);
    }
    return ops::linear_split_workspace_capacity_bytes(parameters.projection,
                                                      parameters.projection_rows, first, last);
}

void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          MemoryArena& workspace, CudaExecutionView execution) {
    if (parameters.role_rows) {
        const std::array<Tensor*, 4> outputs{&query, &key, &gate, &value};
        for (std::size_t i = 0; i < parameters.role_rows->size(); ++i) {
            const auto& role = (*parameters.role_rows)[i];
            auto call = workspace.scope();
            ops::linear(hidden, role.weight, *outputs[i], role.policy, workspace, execution);
        }
        return;
    }
    if (const auto* single = std::get_if<LinearParameters>(&parameters.projection);
        single != nullptr && single->weight.qtype == QType::EXL3) {
        ops::attn_input_proj(hidden, single->weight, query, gate, key, value, single->policy,
                             workspace, execution.stream());
        return;
    }
    std::array outputs{query, key, gate, value};
    ops::linear_split(hidden, parameters.projection, outputs, workspace, execution);
}

namespace {

// The Op-level YaRN value of one model. Disabled whenever the config was not overridden, and the
// Op then takes exactly the plain-RoPE path.
ops::RopeYarn yarn_of(const RopeConfig& config) {
    ops::RopeYarn out;
    out.factor                = config.yarn_factor;
    out.original_max_position = static_cast<std::int32_t>(config.yarn_original_max_position);
    out.beta_fast             = config.yarn_beta_fast;
    out.beta_slow             = config.yarn_beta_slow;
    return out;
}

} // namespace

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query,
               CudaExecutionView execution) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, yarn_of(config), query,
              execution);
}

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query, Tensor& key,
               CudaExecutionView execution) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, yarn_of(config), query,
              key, execution);
}

} // namespace ninfer::models::qwen3_5::execution
