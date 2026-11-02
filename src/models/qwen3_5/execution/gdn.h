#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "core/device.h"

namespace ninfer::models::qwen3_5::execution {

// The decomposed EXL3 GDN route: the fused single-parent form or the separate
// (qkv, z) role form (campaign plan 4a). One predicate for the workspace, decode
// and planning selectors so their copies cannot drift.
[[nodiscard]] inline bool is_exl3_gdn(const GdnParameters& parameters) {
    if (parameters.role_rows) { return true; }
    const auto* single = std::get_if<LinearParameters>(&parameters.projection);
    return single != nullptr && single->weight.qtype == QType::EXL3;
}

[[nodiscard]] inline bool is_exl3_gdn(const GdnProfile& profile) {
    return profile.role_profiles.has_value() ||
           (!profile.projection.second && profile.projection.first.qtype == QType::EXL3);
}

[[nodiscard]] std::size_t gdn_projection_workspace_bytes(const GdnProfile& parameters,
                                                         std::int32_t first, std::int32_t last);
[[nodiscard]] std::size_t gdn_snapshot_workspace_bytes(const GdnProfile& parameters,
                                                       std::int32_t batch, std::int32_t first_width,
                                                       std::int32_t last_width);
[[nodiscard]] std::size_t gdn_record_workspace_bytes(const GdnProfile& parameters,
                                                     std::int32_t batch, std::int32_t first_width,
                                                     std::int32_t last_width);
void gdn_projection(const Tensor& hidden, const GdnParameters& parameters, Tensor& qkv, Tensor& z,
                    MemoryArena& workspace, CudaExecutionView execution);
void gdn_norm_control(const Tensor& residual, const Tensor& norm, float epsilon,
                      const GdnParameters& parameters, Tensor& hidden, Tensor& g, Tensor& beta,
                      MemoryArena& workspace, CudaExecutionView execution);
void gdn_projection_snapshot(const Tensor& hidden, const GdnParameters& parameters,
                             Tensor& conv_states, const Tensor& valid_columns,
                             const Tensor& initial_slots, const Tensor& destination_slots,
                             Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                             MemoryArena& workspace, CudaExecutionView execution);
void gdn_projection_record(const Tensor& hidden, const GdnParameters& parameters,
                           const Tensor& conv_states, const Tensor& valid_columns,
                           const Tensor& initial_slots, Tensor& conv_record, Tensor& query,
                           Tensor& key, Tensor& value, Tensor& z, MemoryArena& workspace,
                           CudaExecutionView execution);

} // namespace ninfer::models::qwen3_5::execution
