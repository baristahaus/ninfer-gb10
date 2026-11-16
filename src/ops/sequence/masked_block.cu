#include "ninfer/ops/sequence.h"
#include "ops/common/host/operands.h"
#include <cstdint>
#include <stdexcept>
#include <string>
#include "core/device.h"

namespace ninfer::ops {

namespace {

__global__ void prepare_masked_block_kernel(const std::int32_t* anchors,
                                            const std::int32_t* lengths,
                                            const std::int32_t* valid_columns, std::int32_t mask_id,
                                            std::int32_t* ids, std::int32_t* positions,
                                            std::int32_t block_size) {
    const int i = static_cast<int>(threadIdx.x);
    const int b = static_cast<int>(blockIdx.x);
    if (i >= block_size) return;
    const int offset  = i + block_size * b;
    const int valid   = valid_columns[b];
    ids[offset]       = i == 0 ? anchors[b] : mask_id;
    positions[offset] = lengths[b] + min(i, valid - 1);
}

void prepare_masked_block_launch(const Tensor& anchors, const Tensor& lengths,
                                 const Tensor& valid_columns, std::int32_t mask_id, Tensor& ids,
                                 Tensor& positions, std::int32_t block_size, cudaStream_t stream) {
    prepare_masked_block_kernel<<<ids.ne[1], 32, 0, stream>>>(
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const std::int32_t*>(lengths.data),
        static_cast<const std::int32_t*>(valid_columns.data), mask_id,
        static_cast<std::int32_t*>(ids.data), static_cast<std::int32_t*>(positions.data),
        block_size);
    CUDA_CHECK(cudaGetLastError());
}

void require_i32_vector(const Tensor& tensor, std::int32_t size, const char* name) {
    if (tensor.dtype != DType::I32 || tensor.ne[0] != size || tensor.ne[1] != 1 ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr) {
        throw std::invalid_argument("prepare_masked_block: " + std::string(name) +
                                    " must be a contiguous I32 vector");
    }
}

void require_i32_matrix(const Tensor& tensor, std::int32_t width, std::int32_t batch,
                        const char* name) {
    if (tensor.dtype != DType::I32 || tensor.ne[0] != width || tensor.ne[1] != batch ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr) {
        throw std::invalid_argument("prepare_masked_block: " + std::string(name) +
                                    " must be a contiguous I32 matrix");
    }
}

bool tensor_ranges_overlap(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}
} // namespace

void prepare_masked_block(const Tensor& anchors, const Tensor& lengths, const Tensor& valid_columns,
                          std::int32_t mask_id, Tensor& ids, Tensor& positions,
                          CudaExecutionView execution) {
    const auto stream             = execution.stream();
    const std::int32_t block_size = ids.ne[0];
    const std::int32_t batch_size = ids.ne[1];
    if (block_size < 1 || block_size > 16) {
        throw std::invalid_argument("prepare_masked_block: W must be 1..16");
    }
    if (batch_size < 1 || batch_size > 8) {
        throw std::invalid_argument("prepare_masked_block: B must be 1..8");
    }
    if (mask_id < 0) {
        throw std::invalid_argument("prepare_masked_block: mask_id must be nonnegative");
    }
    common::operand::tensor(anchors, DType::I32, {batch_size}, 1, "prepare_masked_block", "anchors",
                            execution.device());
    common::operand::tensor(lengths, DType::I32, {batch_size}, 1, "prepare_masked_block", "lengths",
                            execution.device());
    common::operand::tensor(valid_columns, DType::I32, {batch_size}, 1, "prepare_masked_block",
                            "valid_columns", execution.device());
    common::operand::tensor(ids, DType::I32, {block_size, batch_size}, 1, "prepare_masked_block",
                            "ids", execution.device());
    common::operand::tensor(positions, DType::I32, {block_size, batch_size}, 1,
                            "prepare_masked_block", "positions", execution.device());
    const auto output          = common::operand::range(ids, "prepare_masked_block");
    const auto position_output = common::operand::range(positions, "prepare_masked_block");
    common::operand::disjoint(output, position_output, "prepare_masked_block");
    for (const auto* input : {&anchors, &lengths, &valid_columns}) {
        const auto range = common::operand::range(*input, "prepare_masked_block");
        common::operand::disjoint(output, range, "prepare_masked_block");
        common::operand::disjoint(position_output, range, "prepare_masked_block");
    }
    prepare_masked_block_launch(anchors, lengths, valid_columns, mask_id, ids, positions,
                                block_size, stream);
}

void prepare_masked_block(const Tensor& anchors, const Tensor& lengths, const Tensor& valid_columns,
                          std::int32_t mask_id, Tensor& ids, Tensor& positions,
                          cudaStream_t stream) {
    const std::int32_t block_size = ids.ne[0];
    const std::int32_t batch_size = ids.ne[1];
    if (block_size < 1 || block_size > 16) {
        throw std::invalid_argument("prepare_masked_block: W must be 1..16");
    }
    if (batch_size < 1 || batch_size > 8) {
        throw std::invalid_argument("prepare_masked_block: B must be 1..8");
    }
    if (mask_id < 0) {
        throw std::invalid_argument("prepare_masked_block: mask_id must be nonnegative");
    }
    require_i32_vector(anchors, batch_size, "anchors");
    require_i32_vector(lengths, batch_size, "lengths");
    require_i32_vector(valid_columns, batch_size, "valid_columns");
    require_i32_matrix(ids, block_size, batch_size, "ids");
    require_i32_matrix(positions, block_size, batch_size, "positions");
    if (tensor_ranges_overlap(ids, positions) || tensor_ranges_overlap(anchors, ids) ||
        tensor_ranges_overlap(anchors, positions) || tensor_ranges_overlap(lengths, ids) ||
        tensor_ranges_overlap(lengths, positions) || tensor_ranges_overlap(valid_columns, ids) ||
        tensor_ranges_overlap(valid_columns, positions)) {
        throw std::invalid_argument("prepare_masked_block: inputs and outputs must not overlap");
    }
    prepare_masked_block_launch(anchors, lengths, valid_columns, mask_id, ids, positions,
                                block_size, stream);
}

} // namespace ninfer::ops
