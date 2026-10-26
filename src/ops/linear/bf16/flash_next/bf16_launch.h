#pragma once
#include "core/weight.h"

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::flash_next {

// Keep the candidate domain separate from the measured production crossover so the two kernel
// families remain directly comparable in the benchmark overlap.
inline constexpr std::int32_t kBf16SmallTMinTokens         = 2;
inline constexpr std::int32_t kBf16SmallTMaxTokens         = 32;
inline constexpr std::int32_t kBf16LinearSmallTDispatchEnd = 27;

using Bf16Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_bf16_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
// Every column of x [K,T] bitwise as launch_bf16_decode would compute it alone, reading the weight
// once per eight columns (PLE key [10240,2560] and value [2560,2560] projections).
void launch_bf16_decode_columns(const Tensor& x, const Weight& weight, Tensor& out,
                                cudaStream_t stream);
void launch_bf16_hc_down_silu_decode(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream);
void launch_bf16_query_gate_decode(const Tensor& x, const Weight& weight, Tensor& query,
                                   Tensor& gate, cudaStream_t stream);
void launch_bf16_shared_swiglu_decode(const Tensor& x, const Weight& gate_weight,
                                      const Weight& up_weight, Tensor& out,
                                      cudaStream_t stream);
void launch_bf16_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
// Three [512,2560] BF16 projections of one input at T = 2..16 in one launch, each output bitwise
// as launch_bf16_small_t computes it alone. Returns false, launching nothing, outside that domain.
bool launch_bf16_narrow_triple_small_t(const Tensor& x, const Weight& first, const Weight& second,
                                       const Weight& third, Tensor& first_out, Tensor& second_out,
                                       Tensor& third_out, cudaStream_t stream);
void launch_bf16_hc_down_silu_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream);
void launch_bf16_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail::flash_next
