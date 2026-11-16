// Implements: include/ninfer/ops/speculative_round.h
//
// Owns the accepted-hidden gather of the speculative round: for each request it copies the hidden
// row block of the selected column. Moved here from ops/launcher/speculative_round.cu and
// ops/kernel/speculative_round.cuh.

#include "ninfer/ops/speculative_round.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/sampling/acceptance/host_checks.h"

#include <algorithm>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ninfer::ops {

namespace {

using acceptance_detail::require_dtype;
using acceptance_detail::require_vector;

__global__ void speculative_select_accepted_hidden_kernel(const __nv_bfloat16* hidden,
                                                          const std::int32_t* selectors,
                                                          __nv_bfloat16* out, std::int32_t rows,
                                                          std::int32_t cols) {
    const int batch = static_cast<int>(blockIdx.y);
    const int row   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) { return; }
    const int col = selectors[batch];
    if (col < 0 || col >= cols) { return; }
    out[static_cast<std::int64_t>(batch) * rows + row] =
        hidden[(static_cast<std::int64_t>(batch) * cols + col) * rows + row];
}

} // namespace

void speculative_select_accepted_hidden(const Tensor& hidden, const Tensor& selectors, Tensor& out,
                                        cudaStream_t stream) {
    constexpr const char* op = "speculative_select_accepted_hidden";
    require_dtype(hidden, DType::BF16, op, "hidden");
    if (hidden.ne[0] <= 0 || hidden.ne[1] <= 0 || hidden.ne[2] <= 0 || hidden.ne[3] != 1) {
        throw std::invalid_argument("speculative_select_accepted_hidden: invalid hidden shape");
    }
    require_vector(selectors, DType::I32, hidden.ne[2], op, "selectors");
    require_dtype(out, DType::BF16, op, "out");
    if (out.ne[0] != hidden.ne[0] || out.ne[1] != hidden.ne[2] || out.ne[2] != 1 ||
        out.ne[3] != 1) {
        throw std::invalid_argument("speculative_select_accepted_hidden: invalid output shape");
    }
    constexpr int kBlock = 256;
    const int rows       = hidden.ne[0];
    const int batch      = hidden.ne[2];
    const dim3 grid(static_cast<unsigned int>(std::max(1, div_up(rows, kBlock))),
                    static_cast<unsigned int>(batch));
    speculative_select_accepted_hidden_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data),
        static_cast<const std::int32_t*>(selectors.data), static_cast<__nv_bfloat16*>(out.data),
        rows, hidden.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
