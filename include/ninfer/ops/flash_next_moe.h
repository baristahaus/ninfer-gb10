#pragma once
#include "core/weight.h"

#include "core/device.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>
#include <variant>

namespace ninfer::ops {



struct FlashNextExpertBank {
    const void* codes = nullptr;
    const void* scales = nullptr;
    const float* weight_scale_divisors = nullptr;
    const float* input_scale_divisors = nullptr;
    QType qtype = QType::NVFP4;
    std::int32_t experts = 0;
    std::int32_t rows = 0;
    std::int32_t columns = 0;
    // SP-4 tiered banks: the pinned-host warm mini-bank (the same per-expert block layout as
    // `codes`/`scales`), the always-armed per-expert activation counters, and the residency
    // table (bit 31 marks the warm tier, the low bits the expert's compacted rank). Null
    // pointers keep the untiered direct path where every expert reads the device bank at its
    // own rank. `tiered_hot_experts` is the device mini-bank's hot prefix; the cache tail
    // slots behind it are addressed by the residency's compacted rank.
    const void* warm_codes = nullptr;
    const void* warm_scales = nullptr;
    const float* warm_weight_scale_divisors = nullptr;
    std::uint32_t* expert_activations = nullptr;
    const std::uint32_t* expert_residency = nullptr;
    std::uint32_t tiered_hot_experts = 0;
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
                    MemoryArena& workspace, CudaExecutionView execution);

// SP-4 tiered banks: reads and zeroes the per-expert activation counters. Returns the host
// counts (index = expert id). Synchronous diagnostic; call between rounds. Throws when the
// bank is untiered (null counters).
[[nodiscard]] std::vector<std::uint32_t> read_and_reset_expert_activations(
    const FlashNextExpertBank& bank, CudaExecutionView execution);

// SP-4 adaptive tiers: exchanges one warm expert's mini-bank block with one hot expert's
// device block, capacity-neutrally. The promoted expert's pinned-host block moves into the
// demoted expert's device slot (promote first, so the demoted data lands in the vacated
// warm slot), then the tier-state residency is rewritten stream-ordered: the promoted
// expert gains the device rank, the demoted expert the warm rank. Both banks of a layer
// must be swapped with the same expert pair to keep their residencies consistent. The
// caller owns the policy; the copies and the residency writes are stream-ordered on the
// execution stream.
void swap_expert_tier(const FlashNextExpertBank& bank, int promote_expert, int demote_expert,
                      CudaExecutionView execution);

} // namespace ninfer::ops
