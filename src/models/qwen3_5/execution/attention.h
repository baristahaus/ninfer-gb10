#pragma once

#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/workspace.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t attention_projection_workspace_bytes(const AttentionProfile& parameters,
                                                               std::int32_t first,
                                                               std::int32_t last);
// The attention extents this rank holds, read from the prepared weights rather than the config.
//
// `role_rows` carries four separate role weights in the documented order query, key, gate, value
// (`parameters.h`, the `role_rows` member), so the order is not inferred here. The fused form has
// no separate gate, so gate width is the query width. `head_dim` and the residual width are not
// sharded and come from the config, because every rank holds the whole head vector and the whole
// residual.
[[nodiscard]] workspace::AttentionExtents attention_extents(const AttentionParameters& parameters,
                                                           const TextConfig& config);

void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          MemoryArena& workspace, CudaExecutionView execution);

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query,
               CudaExecutionView execution);
void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query, Tensor& key,
               CudaExecutionView execution);

} // namespace ninfer::models::qwen3_5::execution
