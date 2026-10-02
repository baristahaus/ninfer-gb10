#pragma once
#include "core/weight.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops {

class Bf16GemmContext;

struct FlashNextPleWeights {
    Weight key_projection;
    Weight value_projection;
    Tensor key_norm;
    Tensor query_norm;
    Tensor convolution_norm;
    Tensor convolution;
    Tensor embedding_scale;
};

[[nodiscard]] std::size_t flash_next_ple_workspace_capacity_bytes(std::int32_t tokens);

// Exact Qwen3.8 Flash-Next PLE transform. `gathered_fp8` contains the 16 selected
// 160-element table rows flattened to [2560,T]. The table itself remains file-backed.
// The convolution history is BF16 [10240,9], oldest to newest: the transform reads
// `source_state` and writes the history after the last token to `destination_state`. The two
// may be the same tensor (in-place continuation); distinct tensors must not overlap, and
// `source_state` is then left unchanged (a StateImage fork continuing from a checkpoint).
void flash_next_ple(const Tensor& hyper, const Tensor& gathered_fp8,
                    const FlashNextPleWeights& weights, const Tensor& source_state,
                    Tensor& destination_state, Tensor& destination, WorkspaceArena& workspace,
                    cudaStream_t stream, Bf16GemmContext* bf16_gemm = nullptr);

// Exact-B selected-slot transition. Inputs and output are flattened [rows,W,B]; state is
// [10240,9,Slots]. Each row reads source_slots[b] and publishes its final valid state to
// destination_slots[b].
void flash_next_ple_batch_update(const Tensor& hyper, const Tensor& gathered_fp8,
                                 const FlashNextPleWeights& weights, Tensor& states,
                                 const Tensor& valid_columns, const Tensor& source_slots,
                                 const Tensor& destination_slots, std::int32_t width,
                                 std::int32_t batch, Tensor& destination,
                                 WorkspaceArena& workspace, cudaStream_t stream,
                                 Bf16GemmContext* bf16_gemm = nullptr);

void flash_next_ple_replay_record(const Tensor& hyper, const Tensor& gathered_fp8,
                                  const FlashNextPleWeights& weights, const Tensor& states,
                                  const Tensor& valid_columns, const Tensor& source_slots,
                                  std::int32_t width, std::int32_t batch, Tensor& records,
                                  Tensor& destination, WorkspaceArena& workspace,
                                  cudaStream_t stream,
                                  Bf16GemmContext* bf16_gemm = nullptr);

struct FlashNextPleFoldRow {
    std::int32_t source_state_slot;
    std::int32_t destination_state_slot;
    std::int32_t commit_columns;
};

void flash_next_ple_replay_fold(const Tensor& records, Tensor& states,
                                std::span<const FlashNextPleFoldRow> rows,
                                cudaStream_t stream);

} // namespace ninfer::ops
