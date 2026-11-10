#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "core/device.h"
#include "models/qwen3_5/execution/workspace.h"

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
// The GDN extents this rank holds, read from the prepared weights rather than the config. The
// control weights are the a|b pair, so their first parent's rows are the rank's value heads; the
// input projection's rows are Q|K|V (2 key widths plus one value width) and its z rows are the
// value width. At width 1 these equal the config's numbers, which is why only the sharded lane
// ever disagreed with the ops.
[[nodiscard]] workspace::GdnExtents gdn_extents(const GdnParameters& parameters);

// The same extents from the profile, for the planning layer, which builds the per-rank state and
// workspace specs before any weight tensor exists. Both forms read the rank's own geometry: the
// profile carries the sharded shapes because it is described from the sharded bindings.
[[nodiscard]] workspace::GdnExtents gdn_extents(const GdnProfile& profile);

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
