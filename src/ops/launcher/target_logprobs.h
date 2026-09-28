#pragma once

// ninfer::ops::detail - private launch prototype for target_logprobs.

#include "core/tensor.h"
#include "ninfer/ops/target_logprobs.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void target_logprobs_launch(const Tensor& logits, const Tensor& target_ids, std::int32_t valid_rows,
                            const TargetLogprobOptions& options, Tensor& output, Tensor* top_ids,
                            Tensor* top_logprobs, cudaStream_t stream);

} // namespace ninfer::ops::detail
