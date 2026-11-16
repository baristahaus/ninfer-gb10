// Implements: include/ninfer/ops/speculative_round.h
//
// Owns the verify-input preparation of the speculative round: the ids a verification pass consumes
// and, in the longer form, the positions they occupy. Moved here from
// ops/launcher/speculative_round.cu and ops/kernel/speculative_round.cuh, whose remaining accept
// launchers had no callers and were deleted.

#include "ninfer/ops/speculative_round.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/sampling/acceptance/host_checks.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops {

namespace {

using acceptance_detail::require_matrix;
using acceptance_detail::require_vector;

__global__ void speculative_prepare_verify_inputs_kernel(const std::int32_t* anchors,
                                                         const std::int32_t* drafts,
                                                         const std::int32_t* base_positions,
                                                         const std::int32_t* current_extents,
                                                         std::int32_t* verify_ids,
                                                         std::int32_t* positions, std::int32_t k) {
    const int row = static_cast<int>(blockIdx.y);
    const int T   = k + 1;
    int extent    = current_extents[row];
    extent        = extent < 0 ? 0 : (extent > k ? k : extent);
    for (int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x; j < T;
         j += blockDim.x * gridDim.x) {
        const int off = row * T + j;
        verify_ids[off] =
            j == 0 ? anchors[row] : (j <= extent ? drafts[row * k + j - 1] : anchors[row]);
        if (positions != nullptr) {
            positions[off] = base_positions[row] + (j <= extent ? j : extent);
        }
    }
}

void launch(const Tensor& anchors, const Tensor& drafts, const Tensor* base_positions,
            const Tensor& current_extents, Tensor& verify_ids, Tensor* positions,
            cudaStream_t stream) {
    constexpr int kBlock = 32;
    const int k          = drafts.ne[0];
    const int batch      = drafts.ne[1];
    const dim3 grid(static_cast<unsigned int>(div_up(k + 1, kBlock)),
                    static_cast<unsigned int>(batch));
    speculative_prepare_verify_inputs_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const std::int32_t*>(drafts.data),
        base_positions != nullptr ? static_cast<const std::int32_t*>(base_positions->data)
                                  : nullptr,
        static_cast<const std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(verify_ids.data),
        positions != nullptr ? static_cast<std::int32_t*>(positions->data) : nullptr, k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void speculative_prepare_verify_inputs(const Tensor& anchors, const Tensor& drafts,
                                       const Tensor& base_positions, const Tensor& current_extents,
                                       Tensor& verify_ids, Tensor& positions, cudaStream_t stream) {
    constexpr const char* op = "speculative_prepare_verify_inputs";
    const std::int32_t k     = drafts.ne[0];
    const std::int32_t batch = drafts.ne[1];
    if (k < 1) { throw std::invalid_argument("speculative_prepare_verify_inputs: K must be >=1"); }
    if (batch < 1) {
        throw std::invalid_argument("speculative_prepare_verify_inputs: B must be >=1");
    }
    require_vector(anchors, DType::I32, batch, op, "anchors");
    require_matrix(drafts, DType::I32, k, batch, op, "drafts");
    require_vector(base_positions, DType::I32, batch, op, "base_positions");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_matrix(verify_ids, DType::I32, k + 1, batch, op, "verify_ids");
    require_matrix(positions, DType::I32, k + 1, batch, op, "positions");
    launch(anchors, drafts, &base_positions, current_extents, verify_ids, &positions, stream);
}

void speculative_prepare_verify_ids(const Tensor& anchors, const Tensor& drafts,
                                    const Tensor& current_extents, Tensor& verify_ids,
                                    cudaStream_t stream) {
    constexpr const char* op = "speculative_prepare_verify_ids";
    const std::int32_t k     = drafts.ne[0];
    const std::int32_t batch = drafts.ne[1];
    if (k < 1 || batch < 1) {
        throw std::invalid_argument("speculative_prepare_verify_ids: K and B must be positive");
    }
    require_vector(anchors, DType::I32, batch, op, "anchors");
    require_matrix(drafts, DType::I32, k, batch, op, "drafts");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_matrix(verify_ids, DType::I32, k + 1, batch, op, "verify_ids");
    launch(anchors, drafts, nullptr, current_extents, verify_ids, nullptr, stream);
}

} // namespace ninfer::ops
