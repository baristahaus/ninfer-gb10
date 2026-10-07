#pragma once
#include "core/weight.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

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

struct FlashNextMoeWeights {
    Weight router;
    Weight shared_gate;
    Weight shared_up;
    Weight shared_down;
    Weight shared_scale;
    FlashNextExpertBank routed_gate_up;
    FlashNextExpertBank routed_down;
};

[[nodiscard]] std::size_t flash_next_moe_workspace_capacity_bytes(std::int32_t tokens);

// Exact Qwen3.8 Flash-Next 512-way, normalized top-10 routed MoE plus sigmoid-gated shared
// expert. Main-model banks are expert-major NVFP4; the MTP bank is expert-major BF16 or NVFP4.
// Destination is overwritten with the BF16 result.
void flash_next_moe(const Tensor& input, const FlashNextMoeWeights& weights, Tensor& destination,
                    WorkspaceArena& workspace, cudaStream_t stream,
                    Bf16GemmContext* bf16_gemm = nullptr);

} // namespace ninfer::ops
