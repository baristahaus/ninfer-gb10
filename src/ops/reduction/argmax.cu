#include "ops/common/host/operands.h"
#include "ops/common/cuda/reduce.cuh"
#include "ninfer/ops/argmax.h"
#include <cstdint>
#include <stdexcept>
#include "ops/common/numeric/math.h"
#include "ops/common/host/launch.h"
#include "core/device.h"
#include "core/platform.h"
#include "ops/common/numeric/convert.cuh"
#include "core/weight.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <climits>
#include <math_constants.h>

namespace ninfer::ops {
using namespace common;

namespace {

// Small-column execution uses 512 threads to reduce atomic contenders. Aggregate
// execution dispatches registered vocab profiles to a smaller block so the much
// larger 2-D grid exposes enough resident CTAs without oversized reductions.
inline constexpr int kArgmaxBlock          = 512;
inline constexpr int kArgmaxItemsPerThread = 1;

__launch_bounds__(kArgmaxBlock) __global__
    void argmax_kernel(const __nv_bfloat16* logits, std::int32_t* out, std::int32_t valid_rows,
                       std::int32_t physical_rows) {
    const std::int32_t t    = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(t) * physical_rows;

    float best_value        = __bfloat162float(logits[base]);
    std::int32_t best_index = 0;
    for (std::int32_t v = static_cast<std::int32_t>(threadIdx.x); v < valid_rows; v += blockDim.x) {
        const float value = __bfloat162float(logits[base + v]);
        if (score_id_better(value, v, best_value, best_index)) {
            best_value = value;
            best_index = v;
        }
    }

    __shared__ float values[kArgmaxBlock];
    __shared__ std::int32_t indices[kArgmaxBlock];
    values[threadIdx.x]  = best_value;
    indices[threadIdx.x] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other_value        = values[threadIdx.x + stride];
            const std::int32_t other_index = indices[threadIdx.x + stride];
            if (score_id_better(other_value, other_index, values[threadIdx.x],
                                indices[threadIdx.x])) {
                values[threadIdx.x]  = other_value;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) { out[t] = indices[0]; }
}

template <int MaxBlock, int Items>
__launch_bounds__(MaxBlock) __global__
    void argmax_tiled_atomic_kernel(const __nv_bfloat16* logits, std::int32_t* out,
                                    std::int32_t valid_rows, std::int32_t physical_rows) {
    const std::int32_t t          = static_cast<std::int32_t>(blockIdx.y);
    const std::int64_t base       = static_cast<std::int64_t>(t) * physical_rows;
    const std::int32_t tile_start = static_cast<std::int32_t>(blockIdx.x) * blockDim.x * Items;

    float best_value        = -CUDART_INF_F;
    std::int32_t best_index = INT32_MAX;
    if constexpr (Items == 1) {
        const auto v = tile_start + threadIdx.x;
        if (v < valid_rows) {
            const float value = __bfloat162float(logits[base + v]);
            if (score_id_better(value, v, best_value, best_index)) {
                best_value = value;
                best_index = v;
            }
        }
    } else {
        const auto start = tile_start + threadIdx.x * Items;
        if (start + Items <= valid_rows) {
            const auto values = Bf16Vector<Items>::load(logits + base + start);
#pragma unroll
            for (int item = 0; item < Items; ++item) {
                const float value = __bfloat162float(values.word[item]);
                if (score_id_better(value, start + item, best_value, best_index)) {
                    best_value = value;
                    best_index = start + item;
                }
            }
        } else {
#pragma unroll
            for (int item = 0; item < Items; ++item) {
                const auto v = start + item;
                if (v < valid_rows) {
                    const float value = __bfloat162float(logits[base + v]);
                    if (score_id_better(value, v, best_value, best_index)) {
                        best_value = value;
                        best_index = v;
                    }
                }
            }
        }
    }

    block_reduce_argmax(best_value, best_index);
    if (threadIdx.x != 0 || best_index == INT32_MAX) { return; }

    int current = out[t];
    while (true) {
        const float current_value = __bfloat162float(logits[base + current]);
        if (!score_id_better(best_value, best_index, current_value, current)) { break; }
        const int observed = atomicCAS(reinterpret_cast<int*>(out + t), current, best_index);
        if (observed == current) { break; }
        current = observed;
    }
}

constexpr std::int32_t kFullPhysicalRows      = 248320;
constexpr std::int32_t kFullValidRows         = 248077;
constexpr std::int32_t kShortlistRows         = 131072;
constexpr std::int32_t kSmallAggregateColumns = 8;
constexpr int kFullAggregateBlock             = 128;
constexpr int kShortlistAggregateBlock        = 256;

int select_tiled_block(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t t_count) {
    if (t_count <= kSmallAggregateColumns) { return kArgmaxBlock; }
    if (physical_rows == kFullPhysicalRows && valid_rows == kFullValidRows) {
        return kFullAggregateBlock;
    }
    if (physical_rows == kShortlistRows && valid_rows == kShortlistRows) {
        return kShortlistAggregateBlock;
    }
    return kArgmaxBlock;
}

template <int MaxBlock, int Items>
void argmax_tiled_atomic_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                                int block, cudaStream_t stream);

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

#if NINFER_TARGET_SPARK
    if (physical_rows == kFullPhysicalRows && valid_rows == kFullValidRows &&
        reinterpret_cast<std::uintptr_t>(logits.data) % 16 == 0) {
        argmax_tiled_atomic_launch<256, 8>(logits, out, valid_rows, 256, stream);
        return;
    }
#endif
    argmax_tiled_atomic_launch<kArgmaxBlock, 1>(
        logits, out, valid_rows, select_tiled_block(physical_rows, valid_rows, t_count), stream);
}

template <int MaxBlock, int Items>
void argmax_tiled_atomic_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                                int block, cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t t_count       = logits.ne[1];
    const int tiled_blocks           = div_up(valid_rows, block * Items);
    for_each_token_slice(t_count, 1, [&](int token_offset, int token_count) {
        const Tensor logits_slice = logits.slice(1, token_offset, token_count);
        Tensor out_slice          = out.slice(0, token_offset, token_count);
        CUDA_CHECK(cudaMemsetAsync(out_slice.data, 0,
                                   static_cast<std::size_t>(token_count) * sizeof(std::int32_t),
                                   stream));
        const dim3 grid(static_cast<unsigned int>(tiled_blocks),
                        static_cast<unsigned int>(token_count));
        argmax_tiled_atomic_kernel<MaxBlock, Items><<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits_slice.data),
            static_cast<std::int32_t*>(out_slice.data), valid_rows, physical_rows);
        CUDA_CHECK(cudaGetLastError());
    });
}

// ---------------------------------------------------------------------------------------------
// Exact shortlist reranking. Moved here from ops/launcher/argmax.cu and ops/kernel/argmax.cuh
// together with the public entry from ops/wrapper/argmax.cpp, so argmax has one owner.
//
// The approximate head proposes two candidates per tile of the vocabulary; an exact BF16 or
// row-scaled FP8 head then scores those candidates and the best (value, lower id) wins. The tie
// rule and the block reduction are the shared ones in ops/common/cuda/reduce.cuh, so this route
// cannot drift from the argmax route beside it.
// ---------------------------------------------------------------------------------------------
inline constexpr int kShortlistRerankTile        = 512;
inline constexpr int kShortlistCandidatesPerTile = 2;

__launch_bounds__(kShortlistRerankTile) __global__
    void shortlist_tile_candidates_kernel(const __nv_bfloat16* logits, const std::int32_t* id_map,
                                          std::int32_t* candidate_ids, std::int32_t shortlist_rows,
                                          std::int32_t physical_rows, std::int32_t candidate_rows) {
    const int tile          = static_cast<int>(blockIdx.x);
    const int token         = static_cast<int>(blockIdx.y);
    const int row           = tile * kShortlistRerankTile + static_cast<int>(threadIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(token) * physical_rows;
    const float original_value =
        row < shortlist_rows ? __bfloat162float(logits[base + row]) : -CUDART_INF_F;
    const std::int32_t original_row = row < shortlist_rows ? row : INT32_MAX;
    float best_value                = original_value;
    std::int32_t best_row           = original_row;
    block_reduce_argmax(best_value, best_row);
    __shared__ std::int32_t first_row;
    if (threadIdx.x == 0) { first_row = best_row; }
    __syncthreads();

    best_value = original_row == first_row ? -CUDART_INF_F : original_value;
    best_row   = original_row == first_row ? INT32_MAX : original_row;
    block_reduce_argmax(best_value, best_row);
    const std::int64_t output_base =
        static_cast<std::int64_t>(token) * candidate_rows + tile * kShortlistCandidatesPerTile;
    if (threadIdx.x == 0) {
        candidate_ids[output_base]     = id_map[first_row];
        candidate_ids[output_base + 1] = id_map[best_row];
    }
}

__launch_bounds__(256) __global__
    void shortlist_exact_scores_kernel(const __nv_bfloat16* hidden, const __nv_bfloat16* exact_head,
                                       const std::int32_t* candidate_ids, float* candidate_scores,
                                       std::int32_t hidden_rows, std::int32_t candidate_rows) {
    const int candidate = static_cast<int>(blockIdx.x);
    const int token     = static_cast<int>(blockIdx.y);
    const std::int64_t candidate_index =
        static_cast<std::int64_t>(token) * candidate_rows + candidate;
    const int token_id          = candidate_ids[candidate_index];
    const __nv_bfloat16* x      = hidden + static_cast<std::int64_t>(token) * hidden_rows;
    const __nv_bfloat16* weight = exact_head + static_cast<std::int64_t>(token_id) * hidden_rows;
    float sum                   = 0.0F;
    for (int k = static_cast<int>(threadIdx.x); k < hidden_rows; k += blockDim.x) {
        sum = fmaf(__bfloat162float(x[k]), __bfloat162float(weight[k]), sum);
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { partial[threadIdx.x] += partial[threadIdx.x + stride]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { candidate_scores[candidate_index] = partial[0]; }
}

// Row-scaled FP8 exact head: each E4M3 code decodes exactly, the FP32 dot product accumulates
// against the BF16 hidden state, and the row's BF16 multiplier scales the complete sum once.
__launch_bounds__(256) __global__ void shortlist_exact_scores_fp8_kernel(
    const __nv_bfloat16* hidden, const std::uint8_t* exact_codes,
    const __nv_bfloat16* exact_row_scales, const std::int32_t* candidate_ids,
    float* candidate_scores, std::int32_t hidden_rows, std::int32_t candidate_rows) {
    const int candidate = static_cast<int>(blockIdx.x);
    const int token     = static_cast<int>(blockIdx.y);
    const std::int64_t candidate_index =
        static_cast<std::int64_t>(token) * candidate_rows + candidate;
    const int token_id        = candidate_ids[candidate_index];
    const __nv_bfloat16* x    = hidden + static_cast<std::int64_t>(token) * hidden_rows;
    const std::uint8_t* codes = exact_codes + static_cast<std::int64_t>(token_id) * hidden_rows;
    float sum                 = 0.0F;
    for (int k = static_cast<int>(threadIdx.x); k < hidden_rows; k += blockDim.x) {
        __nv_fp8_e4m3 code;
        code.__x = codes[k];
        sum      = fmaf(__bfloat162float(x[k]), static_cast<float>(code), sum);
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { partial[threadIdx.x] += partial[threadIdx.x + stride]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        candidate_scores[candidate_index] =
            partial[0] * __bfloat162float(exact_row_scales[token_id]);
    }
}

__launch_bounds__(kShortlistRerankTile) __global__
    void shortlist_exact_select_kernel(const float* candidate_scores,
                                       const std::int32_t* candidate_ids, std::int32_t* out,
                                       std::int32_t candidate_rows) {
    const int token         = static_cast<int>(blockIdx.x);
    const int candidate     = static_cast<int>(threadIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(token) * candidate_rows;
    float best_value        = -CUDART_INF_F;
    std::int32_t best_index = INT32_MAX;
    for (int row = candidate; row < candidate_rows; row += blockDim.x) {
        const float value        = candidate_scores[base + row];
        const std::int32_t index = candidate_ids[base + row];
        if (score_id_better(value, index, best_value, best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    block_reduce_argmax(best_value, best_index);
    if (threadIdx.x == 0) { out[token] = best_index; }
}

void shortlist_exact_argmax_launch(const Tensor& hidden, const Tensor& approximate_logits,
                                   std::int32_t valid_rows, const Weight& exact_head,
                                   const std::int32_t* id_map, Tensor& candidate_ids,
                                   Tensor& candidate_scores, Tensor& out, cudaStream_t stream) {
    const int shortlist_rows = valid_rows;
    const int physical_rows  = approximate_logits.ne[0];
    const int tokens         = approximate_logits.ne[1];
    const int tiles          = div_up(shortlist_rows, kShortlistRerankTile);
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
        shortlist_exact_scores_fp8_kernel<<<score_grid, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(hidden.data),
            static_cast<const std::uint8_t*>(exact_head.qdata),
            static_cast<const __nv_bfloat16*>(exact_head.scales),
            static_cast<const std::int32_t*>(candidate_ids.data),
            static_cast<float*>(candidate_scores.data), hidden.ne[0], candidate_rows);
    } else {
        shortlist_exact_scores_kernel<<<score_grid, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(hidden.data),
            static_cast<const __nv_bfloat16*>(exact_head.qdata),
            static_cast<const std::int32_t*>(candidate_ids.data),
            static_cast<float*>(candidate_scores.data), hidden.ne[0], candidate_rows);
    }
    CUDA_CHECK(cudaGetLastError());
    shortlist_exact_select_kernel<<<static_cast<unsigned>(tokens), kShortlistRerankTile, 0,
                                    stream>>>(static_cast<const float*>(candidate_scores.data),
                                              static_cast<const std::int32_t*>(candidate_ids.data),
                                              static_cast<std::int32_t*>(out.data), candidate_rows);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void argmax(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
            CudaExecutionView execution) {
    const auto stream = execution.stream();
    if (logits.dtype != DType::BF16) { throw std::invalid_argument("argmax: logits must be BF16"); }
    if (out.dtype != DType::I32) { throw std::invalid_argument("argmax: out must be I32"); }

    const std::int64_t logits_n = common::operand::numel(logits, "argmax");
    (void)common::operand::numel(out, "argmax");

    if (logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument("argmax: logits must be rank-2 [vocab,T]");
    }
    if (out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("argmax: out must be rank-1 [T]");
    }
    if (logits.ne[0] <= 0) {
        throw std::invalid_argument("argmax: physical rows must be positive");
    }
    if (valid_rows <= 0 || valid_rows > logits.ne[0]) {
        throw std::invalid_argument("argmax: valid_rows must be in [1, logits.ne[0]]");
    }
    if (out.ne[0] != logits.ne[1]) {
        throw std::invalid_argument("argmax: out shape must be [logits.ne[1]]");
    }
    if (logits_n == 0) { return; }
    if (logits.device != execution.device() || out.device != execution.device()) {
        throw std::invalid_argument("argmax: operands must belong to the execution device");
    }

    if (!logits.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("argmax: logits/out must be contiguous");
    }
    if (logits.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("argmax: logits/out data must be non-null");
    }

    argmax_launch(logits, out, valid_rows, stream);
}

// The stream form of the public argmax. It validates the master-generation shape terms the serving
// and speculative callers use and dispatches to the same argmax_launch the execution-view form
// uses, so the two forms cannot diverge.
std::int64_t argmax_numel_allow_zero(const Tensor& t, const char* label) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("argmax: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("argmax: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

void argmax(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream) {
    if (logits.dtype != DType::BF16) { throw std::invalid_argument("argmax: logits must be BF16"); }
    if (out.dtype != DType::I32) { throw std::invalid_argument("argmax: out must be I32"); }

    const std::int64_t logits_n = argmax_numel_allow_zero(logits, "logits");
    (void)argmax_numel_allow_zero(out, "out");

    if (logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument("argmax: logits must be rank-2 [vocab,T]");
    }
    if (out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("argmax: out must be rank-1 [T]");
    }
    if (logits.ne[0] <= 0) {
        throw std::invalid_argument("argmax: physical rows must be positive");
    }
    if (valid_rows <= 0 || valid_rows > logits.ne[0]) {
        throw std::invalid_argument("argmax: valid_rows must be in [1, logits.ne[0]]");
    }
    if (out.ne[0] != logits.ne[1]) {
        throw std::invalid_argument("argmax: out shape must be [logits.ne[1]]");
    }
    if (logits_n == 0) { return; }

    if (!logits.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("argmax: logits/out must be contiguous");
    }
    if (logits.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("argmax: logits/out data must be non-null");
    }

    argmax_launch(logits, out, valid_rows, stream);
}

void shortlist_exact_argmax(const Tensor& hidden, const Tensor& approximate_logits,
                            std::int32_t valid_rows, const Weight& exact_head,
                            const std::int32_t* id_map, Tensor& candidate_ids,
                            Tensor& candidate_scores, Tensor& out, cudaStream_t stream) {
    constexpr int kTile    = kShortlistRerankTile;
    constexpr int kPerTile = kShortlistCandidatesPerTile;
    if (hidden.dtype != DType::BF16 || approximate_logits.dtype != DType::BF16 ||
        candidate_ids.dtype != DType::I32 || candidate_scores.dtype != DType::FP32 ||
        out.dtype != DType::I32) {
        throw std::invalid_argument("shortlist_exact_argmax: invalid tensor dtype");
    }
    if (!hidden.is_contiguous() || !approximate_logits.is_contiguous() ||
        !candidate_ids.is_contiguous() || !candidate_scores.is_contiguous() ||
        !out.is_contiguous()) {
        throw std::invalid_argument("shortlist_exact_argmax: tensors must be contiguous");
    }
    const int tokens         = hidden.ne[1];
    const int shortlist_rows = valid_rows;
    const int candidate_rows = ((shortlist_rows + kTile - 1) / kTile) * kPerTile;
    if (tokens <= 0 || hidden.ne[0] <= 0 || hidden.ne[2] != 1 || hidden.ne[3] != 1 ||
        shortlist_rows < 2 || shortlist_rows > approximate_logits.ne[0] ||
        approximate_logits.ne[1] != tokens || approximate_logits.ne[2] != 1 ||
        approximate_logits.ne[3] != 1 || candidate_ids.ne[0] != candidate_rows ||
        candidate_ids.ne[1] != tokens || candidate_ids.ne[2] != 1 || candidate_ids.ne[3] != 1 ||
        candidate_scores.ne[0] != candidate_rows || candidate_scores.ne[1] != tokens ||
        candidate_scores.ne[2] != 1 || candidate_scores.ne[3] != 1 || out.ne[0] != tokens ||
        out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("shortlist_exact_argmax: invalid tensor shape");
    }
    const bool bf16_head =
        exact_head.qtype == QType::BF16 && exact_head.layout == QuantLayout::Contiguous;
    const bool fp8_head = exact_head.qtype == QType::FP8_E4M3FN_ROW_BF16 &&
                          exact_head.layout == QuantLayout::RowScale &&
                          exact_head.scales != nullptr;
    if ((!bf16_head && !fp8_head) || exact_head.k != hidden.ne[0] || exact_head.n <= 0 ||
        exact_head.qdata == nullptr || id_map == nullptr || hidden.data == nullptr ||
        approximate_logits.data == nullptr || candidate_ids.data == nullptr ||
        candidate_scores.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("shortlist_exact_argmax: invalid exact head or storage");
    }
    shortlist_exact_argmax_launch(hidden, approximate_logits, valid_rows, exact_head, id_map,
                                  candidate_ids, candidate_scores, out, stream);
}

} // namespace ninfer::ops
