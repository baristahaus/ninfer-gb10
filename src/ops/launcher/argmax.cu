// Implements: include/ninfer/ops/argmax.h
// Match: validated contiguous BF16 logits and I32 output.
// Algorithm assumptions: one tile uses a direct reduction; larger domains use
// zero-initialized atomic winners across route-selected row tiles.
#include "ops/launcher/argmax.h"

#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/kernel/argmax.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kFullPhysicalRows      = 248320;
constexpr std::int32_t kFullValidRows         = 248077;
constexpr std::int32_t kShortlistRows         = 131072;
constexpr std::int32_t kSmallAggregateColumns = 8;
constexpr int kFullAggregateBlock             = 128;
constexpr int kShortlistAggregateBlock        = 256;

int tiled_block_for(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t t_count) {
    if (t_count <= kSmallAggregateColumns) { return kArgmaxBlock; }
    if (physical_rows == kFullPhysicalRows && valid_rows == kFullValidRows) {
        return kFullAggregateBlock;
    }
    if (physical_rows == kShortlistRows && valid_rows == kShortlistRows) {
        return kShortlistAggregateBlock;
    }
    return kArgmaxBlock;
}

void argmax_tiled_atomic_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                                int block, cudaStream_t stream);

} // namespace

void argmax_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                   cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t t_count       = logits.ne[1];
    if (t_count == 0) { return; }

    constexpr int kTileElems = kArgmaxBlock * kArgmaxItemsPerThread;
    const int tiled_blocks   = div_up(valid_rows, kTileElems);
    if (tiled_blocks < 2) {
        argmax_kernel<<<static_cast<unsigned int>(t_count), kArgmaxBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::int32_t*>(out.data),
            valid_rows, physical_rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    argmax_tiled_atomic_launch(logits, out, valid_rows,
                               tiled_block_for(physical_rows, valid_rows, t_count), stream);
}

void shortlist_exact_argmax_launch(const Tensor& hidden, const Tensor& approximate_logits,
                                   std::int32_t valid_rows,
                                   const Weight& exact_head, const std::int32_t* id_map,
                                   Tensor& candidate_ids, Tensor& candidate_scores, Tensor& out,
                                   cudaStream_t stream) {
    const int shortlist_rows = valid_rows;
    const int physical_rows = approximate_logits.ne[0];
    const int tokens = approximate_logits.ne[1];
    const int tiles = div_up(shortlist_rows, kShortlistRerankTile);
    const int candidate_rows = tiles * kShortlistCandidatesPerTile;
    shortlist_tile_candidates_kernel<<<dim3(static_cast<unsigned>(tiles),
                                              static_cast<unsigned>(tokens)),
                                       kShortlistRerankTile, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(approximate_logits.data), id_map,
        static_cast<std::int32_t*>(candidate_ids.data), shortlist_rows, physical_rows,
        candidate_rows);
    CUDA_CHECK(cudaGetLastError());
    const dim3 score_grid(static_cast<unsigned>(candidate_rows), static_cast<unsigned>(tokens));
    if (exact_head.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        shortlist_exact_scores_kernel<true><<<score_grid, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(hidden.data), exact_head.qdata,
            static_cast<const __nv_bfloat16*>(exact_head.scales),
            static_cast<const std::int32_t*>(candidate_ids.data),
            static_cast<float*>(candidate_scores.data), hidden.ne[0], candidate_rows);
    } else {
        shortlist_exact_scores_kernel<false><<<score_grid, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(hidden.data), exact_head.qdata, nullptr,
            static_cast<const std::int32_t*>(candidate_ids.data),
            static_cast<float*>(candidate_scores.data), hidden.ne[0], candidate_rows);
    }
    CUDA_CHECK(cudaGetLastError());
    shortlist_exact_select_kernel<<<static_cast<unsigned>(tokens), kShortlistRerankTile, 0,
                                    stream>>>(
        static_cast<const float*>(candidate_scores.data),
        static_cast<const std::int32_t*>(candidate_ids.data),
        static_cast<std::int32_t*>(out.data), candidate_rows);
    CUDA_CHECK(cudaGetLastError());
}

namespace {

void argmax_tiled_atomic_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                                int block, cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t t_count       = logits.ne[1];
    const int tiled_blocks           = div_up(valid_rows, block * kArgmaxItemsPerThread);
    for_each_token_slice(t_count, 1, [&](int token_offset, int token_count) {
        const Tensor logits_slice = logits.slice(1, token_offset, token_count);
        Tensor out_slice          = out.slice(0, token_offset, token_count);
        CUDA_CHECK(cudaMemsetAsync(out_slice.data, 0,
                                   static_cast<std::size_t>(token_count) * sizeof(std::int32_t),
                                   stream));
        const dim3 grid(static_cast<unsigned int>(tiled_blocks),
                        static_cast<unsigned int>(token_count));
        argmax_tiled_atomic_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits_slice.data),
            static_cast<std::int32_t*>(out_slice.data), valid_rows, physical_rows);
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace
} // namespace ninfer::ops::detail
