// Implements: include/ninfer/ops/speculative_round.h
//
// Owns the proposal id remap: every proposed token id is replaced by id_map[id] when the id is in
// range, and left alone otherwise. Moved here from ops/launcher/speculative_round.cu and
// ops/kernel/speculative_round.cuh.

#include "ninfer/ops/speculative_round.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/sampling/acceptance/host_checks.h"

#include <algorithm>
#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops {

namespace {

using acceptance_detail::require_dtype;

__global__ void proposal_remap_token_ids_kernel(std::int32_t* proposal_tokens,
                                                std::int32_t proposal_count,
                                                const std::int32_t* id_map, std::int32_t n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= proposal_count) { return; }
    const int idx = proposal_tokens[i];
    if (idx >= 0 && idx < n) { proposal_tokens[i] = id_map[idx]; }
}

} // namespace

void proposal_remap_token_ids(Tensor& proposal_tokens, const std::int32_t* id_map, std::int32_t n,
                              cudaStream_t stream) {
    constexpr const char* op = "proposal_remap_token_ids";
    require_dtype(proposal_tokens, DType::I32, op, "proposal_tokens");
    if (proposal_tokens.ne[0] <= 0 || proposal_tokens.ne[1] != 1 || proposal_tokens.ne[2] != 1 ||
        proposal_tokens.ne[3] != 1) {
        throw std::invalid_argument(
            "proposal_remap_token_ids: proposal_tokens must be a non-empty vector");
    }
    if (id_map == nullptr || n <= 0) {
        throw std::invalid_argument("proposal_remap_token_ids: id_map must be non-null and n>0");
    }
    constexpr int kBlock = 256;
    const int count      = proposal_tokens.ne[0];
    const int grid       = std::max(1, div_up(count, kBlock));
    proposal_remap_token_ids_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<std::int32_t*>(proposal_tokens.data), count, id_map, n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
