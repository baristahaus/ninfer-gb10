#pragma once
#include "core/weight.h"

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Computes one vocabulary argmax per column:
 *
 *   out[t] = min argmax_{0 <= v < valid_rows} float(logits[v,t]).
 *
 * `logits` is contiguous BF16 [physical_rows,T], `out` is contiguous I32 [T], and
 * 1 <= valid_rows <= physical_rows. Physical rows [valid_rows,physical_rows) do not
 * participate. Equal maxima select the lowest row index. `out` must not overlap `logits`.
 * The Op has no workspace and changes no state other than writing all of `out`.
 */
void argmax(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream);

/**
 * Selects the exact-head argmax from a broad approximate-logit shortlist.
 *
 * Each 512-row tile in [0,valid_rows) contributes its eight best approximate rows. The Op
 * remaps those rows through `id_map`, evaluates their complete FP32-accumulated dot products
 * against the exact head (contiguous BF16, or row-scaled FP8 whose row multiplier scales the
 * complete dot product), and returns the best exact token per column.
 * `candidate_ids` and `candidate_scores` are caller-owned scratch tensors of shape
 * [8*ceil(valid_rows/512),T].
 */
void shortlist_exact_argmax(const Tensor& hidden, const Tensor& approximate_logits,
                            std::int32_t valid_rows,
                            const Weight& exact_head, const std::int32_t* id_map,
                            Tensor& candidate_ids, Tensor& candidate_scores, Tensor& out,
                            cudaStream_t stream);

} // namespace ninfer::ops
