#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

std::size_t mxfp8_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
                                                  LinearPolicy policy, std::int32_t min_tokens,
                                                  std::int32_t max_tokens);

void mxfp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    MemoryArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
