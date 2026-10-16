#pragma once

// The dense EXL3 route of the Linear Op (SP-7 contract sections 4 and 7): admission, workspace
// capacity and launch for `exl3_dense_v1` weights.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// The registered dense problems (SP-7 dense step): the 27B dense-route projection geometries
// that the converter's dense layout emits, at the decode token shapes T in 1..8. Prefill token
// counts are unregistered until their route exists and are refused at admission.
[[nodiscard]] bool exl3_dense_registered(std::int32_t n, std::int32_t k);

[[nodiscard]] std::size_t exl3_dense_workspace_bytes(std::int32_t n, std::int32_t k,
                                                     std::int32_t max_tokens);

// Admits the weight (Exl3Dense layout, mcg or mul1 codebook, a registered problem) and launches
// the fused decode route. `workspace` is the caller's transient arena; the route allocates its
// staging scratch inside a scope of it, sized by exl3_dense_workspace_bytes(n, k, T).
void exl3_dense_dispatch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena* workspace,
                         cudaStream_t stream);

// The kernel launcher (exl3_dense.cu). The dispatch has already validated every argument.
void launch_exl3_dense(const Tensor& x, const Weight& w, Tensor& out, void* workspace,
                       cudaStream_t stream);

} // namespace ninfer::ops::detail
