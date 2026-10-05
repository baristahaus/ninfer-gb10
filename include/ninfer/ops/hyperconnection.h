#pragma once
#include "core/weight.h"
#include "ninfer/ops/activation_steering.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

class Bf16GemmContext;

// Optional per-call activation hooks of an attention-input hyperconnection mix.
struct HyperConnectionActivation {
    // Steering adjusts the selected residual lanes of `hyper` in place, after the represented
    // BF16 combine and before grouped RMSNorm. Strength, rank and lane mask are read from device
    // data per (row, layer, lane); a zero strength or rank leaves the arithmetic unchanged.
    const ActivationDevice* steering = nullptr;
    int layer                        = 0;
    // Tokens per activation row: row = token / width (prefill chunk, 1 for decode, K+1 for MTP).
    int width = 1;
    // Capture of the post-steering stream, normalization, gates and mixed input.
    const ActivationDevice* capture = nullptr;
    const Tensor* positions         = nullptr;
    const Tensor* ids               = nullptr;
    const Tensor* valid             = nullptr;
    bool speculative_columns        = false;
};

// Down [320,10240] and Up [10240,320] are both BF16 or both row-scaled FP8 E4M3 (weight-only;
// activations and the BF16 low-rank boundary are unchanged). Injection is BF16.
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

// `activation->steering`, when set, rewrites the steered lanes of `hyper` in place before the
// mix; `hyper` is otherwise read-only.
void hyperconnection_mix(const Tensor& hyper, const HyperConnectionWeights& weights,
                         Tensor& block_input, Tensor* injection, WorkspaceArena& workspace,
                         cudaStream_t stream, Bf16GemmContext* bf16_gemm = nullptr,
                         const HyperConnectionActivation* activation = nullptr);

// Commits one pending branch output and prepares the next branch in one pass,
// preserving the materialized BF16 combine boundary before grouped RMSNorm.
void hyperconnection_combine_mix(Tensor& hyper, const Tensor& previous_block_output,
                                 const Tensor& previous_injection,
                                 const HyperConnectionWeights& weights,
                                 Tensor& block_input, Tensor* injection,
                                 WorkspaceArena& workspace, cudaStream_t stream,
                                 Bf16GemmContext* bf16_gemm = nullptr,
                                 const HyperConnectionActivation* activation = nullptr);

void hyperconnection_combine(Tensor& hyper, const Tensor& block_output, const Tensor& injection,
                             cudaStream_t stream);

} // namespace ninfer::ops
