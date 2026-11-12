#include "ops/common/host/operands.h"
#include "ninfer/ops/indexing.h"
#include <stdexcept>
#include <string>
#include "ops/common/numeric/math.h"
#include "ops/common/host/launch.h"
#include "core/device.h"
#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>

namespace ninfer::ops {
using namespace common;

namespace {

__global__ void concat_bf16_features_kernel(const __nv_bfloat16* first, const __nv_bfloat16* second,
                                            __nv_bfloat16* out, std::int32_t rows) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) { return; }

    const int token             = static_cast<int>(blockIdx.y);
    const std::int64_t in_idx   = static_cast<std::int64_t>(token) * rows + row;
    const std::int64_t out_base = static_cast<std::int64_t>(token) * (2 * rows);
    out[out_base + row]         = first[in_idx];
    out[out_base + rows + row]  = second[in_idx];
}

// The packed source is q|k|g|v: two wide ranges and two narrow ones, in that order. The extents are
// operands, not constants: at tensor-parallel width 2 the Rows shard halves every head count, so the
// same op presents 7168 packed rows and 12/2/12/2 heads where width 1 presents 14336 and 24/4/24/4.
__global__ void split_bf16_features_kernel(const __nv_bfloat16* source, __nv_bfloat16* first,
                                           __nv_bfloat16* second, __nv_bfloat16* third,
                                           __nv_bfloat16* fourth, std::int32_t tokens,
                                           std::int32_t total_rows, std::int32_t wide_rows,
                                           std::int32_t narrow_rows) {
    const std::int64_t idx = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t n   = static_cast<std::int64_t>(total_rows) * tokens;
    if (idx >= n) { return; }

    const int row             = static_cast<int>(idx % total_rows);
    const int token           = static_cast<int>(idx / total_rows);
    const __nv_bfloat16 value = source[idx];

    if (row < wide_rows) {
        first[static_cast<std::int64_t>(token) * wide_rows + row] = value;
        return;
    }
    if (row < wide_rows + narrow_rows) {
        const int local = row - wide_rows;
        second[static_cast<std::int64_t>(token) * narrow_rows + local] = value;
        return;
    }
    if (row < 2 * wide_rows + narrow_rows) {
        const int local = row - wide_rows - narrow_rows;
        third[static_cast<std::int64_t>(token) * wide_rows + local] = value;
        return;
    }

    const int local = row - 2 * wide_rows - narrow_rows;
    fourth[static_cast<std::int64_t>(token) * narrow_rows + local] = value;
}

void concat_bf16_features_launch(const Tensor& first, const Tensor& second, Tensor& out,
                                 cudaStream_t stream) {
    constexpr int kBlock = 256;
    for_each_token_slice(first.ne[1], 1, [&](int token_offset, int token_count) {
        const Tensor first_slice  = first.slice(1, token_offset, token_count);
        const Tensor second_slice = second.slice(1, token_offset, token_count);
        Tensor out_slice          = out.slice(1, token_offset, token_count);
        const dim3 grid(static_cast<unsigned int>(div_up(first.ne[0], kBlock)),
                        static_cast<unsigned int>(token_count));
        concat_bf16_features_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(first_slice.data),
            static_cast<const __nv_bfloat16*>(second_slice.data),
            static_cast<__nv_bfloat16*>(out_slice.data), first.ne[0]);
        CUDA_CHECK(cudaGetLastError());
    });
}

void split_bf16_features_launch(const Tensor& source, Tensor& first, Tensor& second, Tensor& third,
                                Tensor& fourth, std::int32_t wide_rows, std::int32_t narrow_rows,
                                cudaStream_t stream) {
    constexpr int kBlock = 256;
    const std::int64_t n = static_cast<std::int64_t>(source.ne[0]) * source.ne[1];
    const int grid =
        static_cast<int>(std::max<std::int64_t>(1, div_up(n, static_cast<std::int64_t>(kBlock))));
    split_bf16_features_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(source.data), static_cast<__nv_bfloat16*>(first.data),
        static_cast<__nv_bfloat16*>(second.data), static_cast<__nv_bfloat16*>(third.data),
        static_cast<__nv_bfloat16*>(fourth.data), source.ne[1], source.ne[0], wide_rows,
        narrow_rows);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void concat_bf16_features(const Tensor& first, const Tensor& second, Tensor& out,
                          CudaExecutionView execution) {
    const auto stream        = execution.stream();
    constexpr const char* op = "concat_bf16_features";
    common::operand::dtype(first, DType::BF16, op, "first");
    common::operand::contiguous_nonnull(first, op, "first", execution.device());
    common::operand::dtype(second, DType::BF16, op, "second");
    common::operand::contiguous_nonnull(second, op, "second", execution.device());
    common::operand::dtype(out, DType::BF16, op, "out");
    common::operand::contiguous_nonnull(out, op, "out", execution.device());
    const std::int32_t rows   = first.ne[0];
    const std::int32_t tokens = first.ne[1];
    if (rows <= 0) { throw std::invalid_argument("concat_bf16_features: D must be positive"); }
    if (tokens <= 0) { throw std::invalid_argument("concat_bf16_features: T must be positive"); }
    common::operand::shape(first, {rows, tokens, 1, 1}, op, "first");
    common::operand::shape(second, {rows, tokens, 1, 1}, op, "second");
    common::operand::shape(out, {2 * rows, tokens, 1, 1}, op, "out");

    concat_bf16_features_launch(first, second, out, stream);
}

void split_bf16_features(const Tensor& source, Tensor& first, Tensor& second, Tensor& third,
                         Tensor& fourth, CudaExecutionView execution) {
    const auto stream        = execution.stream();
    constexpr const char* op = "split_bf16_features";
    common::operand::dtype(source, DType::BF16, op, "source");
    common::operand::contiguous_nonnull(source, op, "source", execution.device());
    common::operand::dtype(first, DType::BF16, op, "first");
    common::operand::contiguous_nonnull(first, op, "first", execution.device());
    common::operand::dtype(second, DType::BF16, op, "second");
    common::operand::contiguous_nonnull(second, op, "second", execution.device());
    common::operand::dtype(third, DType::BF16, op, "third");
    common::operand::contiguous_nonnull(third, op, "third", execution.device());
    common::operand::dtype(fourth, DType::BF16, op, "fourth");
    common::operand::contiguous_nonnull(fourth, op, "fourth", execution.device());
    const std::int32_t tokens = source.ne[1];
    if (tokens <= 0) { throw std::invalid_argument("split_bf16_features: T must be positive"); }
    // DERIVED, NOT ASSERTED AGAINST A LITERAL. The four destinations describe the split, so the
    // packed extents come from them: the head extent is shared, the two wide ranges (query, gate)
    // carry one head count and the two narrow ones (key, value) the other, and the source is their
    // concatenation in q|k|g|v order. Asserting those RELATIONS is what keeps a wrong derivation
    // from moving the failure into the kernel, and it holds at whatever width the caller presents --
    // 14336 packed rows with 24/4/24/4 heads at width 1, 7168 with 12/2/12/2 at width 2.
    const std::int32_t head_dim = first.ne[0];
    if (head_dim <= 0) { throw std::invalid_argument("split_bf16_features: head extent must be positive"); }
    const auto expect_operand = [&](const Tensor& operand, const char* name) {
        if (operand.ne[0] != head_dim || operand.ne[2] != tokens || operand.ne[3] != 1) {
            throw std::invalid_argument(std::string("split_bf16_features: ") + name +
                                        " is not a {head, heads, tokens} range of the source");
        }
    };
    expect_operand(first, "first");
    expect_operand(second, "second");
    expect_operand(third, "third");
    expect_operand(fourth, "fourth");
    if (first.ne[1] != third.ne[1]) {
        throw std::invalid_argument(
            "split_bf16_features: the two wide ranges must carry the same head count");
    }
    if (second.ne[1] != fourth.ne[1]) {
        throw std::invalid_argument(
            "split_bf16_features: the two narrow ranges must carry the same head count");
    }
    const std::int32_t wide_rows   = head_dim * first.ne[1];
    const std::int32_t narrow_rows = head_dim * second.ne[1];
    if (source.ne[0] != 2 * wide_rows + 2 * narrow_rows) {
        throw std::invalid_argument(
            "split_bf16_features: the source's rows are not the concatenation of the four ranges");
    }
    if (source.ne[2] != 1 || source.ne[3] != 1) {
        throw std::invalid_argument("split_bf16_features: the source must be a {rows, tokens} plane");
    }

    split_bf16_features_launch(source, first, second, third, fourth, wide_rows, narrow_rows, stream);
}

} // namespace ninfer::ops
