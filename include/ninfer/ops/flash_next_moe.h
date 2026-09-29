#pragma once
#include "core/weight.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <variant>

namespace ninfer::ops {

class Bf16GemmContext;

struct FlashNextExpertBank {
    const void* codes = nullptr;
    const void* scales = nullptr;
    const float* weight_scale_divisors = nullptr;
    const float* input_scale_divisors = nullptr;
    QType qtype = QType::NVFP4;
    std::int32_t experts = 0;
    std::int32_t rows = 0;
    std::int32_t columns = 0;
};

// The shared expert's gate and up projections: two BF16 [640,2560] matrices, or one row-scaled
// FP8 parent [1280,2560] whose gate rows [0,640) precede their up rows [640,1280).
struct FlashNextSharedGateUpPair {
    Weight gate;
    Weight up;
};

using FlashNextSharedGateUp = std::variant<FlashNextSharedGateUpPair, Weight>;

struct FlashNextMoeWeights {
    Weight router;
    FlashNextSharedGateUp shared_gate_up;
    Weight shared_down; // BF16 or row-scaled FP8 [2560,640]
    Weight shared_scale;
    FlashNextExpertBank routed_gate_up;
    FlashNextExpertBank routed_down;
};

[[nodiscard]] std::size_t flash_next_moe_workspace_capacity_bytes(std::int32_t tokens);

// Exact Qwen3.8 Flash-Next 512-way, normalized top-10 routed MoE plus sigmoid-gated shared
// expert. Banks are expert-major NVFP4 (main layers, and the MTP layer of the FP8 profile) or
// expert-major BF16 (the checkpoint's MTP banks); the bank's format selects the route. The
// shared expert is BF16, or row-scaled FP8 with a packed gate/up parent (LinearSwiGLU's
// Flash-Next profile). Destination is overwritten with the BF16 result.
void flash_next_moe(const Tensor& input, const FlashNextMoeWeights& weights, Tensor& destination,
                    WorkspaceArena& workspace, cudaStream_t stream,
                    Bf16GemmContext* bf16_gemm = nullptr,
                    bool wide_decode_gate = true);

} // namespace ninfer::ops
