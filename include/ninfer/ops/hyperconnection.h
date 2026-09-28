#pragma once
#include "core/weight.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

class Bf16GemmContext;

struct HyperConnectionWeights {
    Tensor norm;
    Weight down;
    Weight up;
    Weight injection;
};

// Initializes the four-stream state by repeating each BF16 [2560,T] input column.
void hyperconnection_repeat(const Tensor& input, Tensor& hyper, cudaStream_t stream);

// Adds one [2560,T] embedding branch to each of the four [2560,T]
// predictor branches in-place.
void hyperconnection_add_repeated(const Tensor& embedding, Tensor& hyper,
                                  cudaStream_t stream);

[[nodiscard]] std::size_t hyperconnection_mix_workspace_capacity_bytes(std::int32_t tokens,
                                                                        bool with_injection);

// The [320,10240] down and [10240,320] up projections are each BF16 or row-scaled FP8
// (fp8_e4m3fn_row_bf16); the [4,10240] injection rows are BF16.
void hyperconnection_mix(const Tensor& hyper, const HyperConnectionWeights& weights,
                         Tensor& block_input, Tensor* injection, WorkspaceArena& workspace,
                         cudaStream_t stream, Bf16GemmContext* bf16_gemm = nullptr);

// Commits one pending branch output and prepares the next branch in one pass,
// preserving the materialized BF16 combine boundary before grouped RMSNorm.
void hyperconnection_combine_mix(Tensor& hyper, const Tensor& previous_block_output,
                                 const Tensor& previous_injection,
                                 const HyperConnectionWeights& weights,
                                 Tensor& block_input, Tensor* injection,
                                 WorkspaceArena& workspace, cudaStream_t stream,
                                 Bf16GemmContext* bf16_gemm = nullptr);

void hyperconnection_combine(Tensor& hyper, const Tensor& block_output, const Tensor& injection,
                             cudaStream_t stream);

} // namespace ninfer::ops
