#pragma once

#include "ops/linear/common/output.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_operands.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// One owner per shared NVFP4 A4 TMA instantiation.
//
// The TMA kernels are non-RDC: a kernel must be emitted in the translation unit that launches it,
// and the device linker does not merge two copies. Two Ops that select the same schedule and the
// same segmented output therefore each emitted the same `__global__` symbol, and the module
// registered it twice. compute-sanitizer reports "Duplicate entry kernels named ..." for exactly
// that, and cuda-gdb then reports "No CUDA kernels. No CUDA devices." on a live process.
//
// These two entries hold the only instantiations of the shared shapes. Every Op that serves one of
// these geometries calls the entry instead of launching the template itself.

using Nvfp4Segmented6144x1024  = LinearBf16SegmentedOutput<6144, 1024, 6144, 1024>;
using Nvfp4Segmented10240x6144 = LinearBf16SegmentedOutput<10240, 6144>;

void launch_nvfp4_a4_tma_segmented_6144_1024(const Nvfp4A4Operands& operands,
                                             Nvfp4Segmented6144x1024 output, cudaStream_t stream);

void launch_nvfp4_a4_tma_segmented_10240_6144(const Nvfp4A4Operands& operands,
                                              Nvfp4Segmented10240x6144 output, cudaStream_t stream);

} // namespace ninfer::ops::detail
