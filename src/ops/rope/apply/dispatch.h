#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "ops/rope/kernels/yarn_device.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void rope_dispatch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                   int multiprocessor_count, cudaStream_t stream, RopeYarnDevice yarn = {});

void rope_dispatch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                   int multiprocessor_count, cudaStream_t stream, RopeYarnDevice yarn = {});

} // namespace ninfer::ops::detail
