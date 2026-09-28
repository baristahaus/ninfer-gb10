#pragma once

// Identity-free Qwen3.8 family runtime helper.

#include "core/arena.h"
#include "core/tensor.h"
#include <ninfer/models/qwen3_8_flash_next/mtp_alignment.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace ninfer::models::qwen3_8_flash_next::detail {

// Composes the generic scatter Op from the family-provided shifted-window interpretation.
void scatter_shifted_visual_embeddings(Tensor& input_embeddings, const Tensor& visual_embeddings,
                                       const qwen3_8_flash_next::MtpVisualOverlap& overlap,
                                       Tensor& destination_indices, cudaStream_t stream);

} // namespace ninfer::models::qwen3_8_flash_next::detail
