#pragma once
#include "core/weight.h"

// ninfer::ops::detail - private launch prototype for argmax.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void argmax_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream);

void shortlist_exact_argmax_launch(const Tensor& hidden, const Tensor& approximate_logits,
                                   std::int32_t valid_rows,
                                   const Weight& exact_head, const std::int32_t* id_map,
                                   Tensor& candidate_ids, Tensor& candidate_scores, Tensor& out,
                                   cudaStream_t stream);

} // namespace ninfer::ops::detail
