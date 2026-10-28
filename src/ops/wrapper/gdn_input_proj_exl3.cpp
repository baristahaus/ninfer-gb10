#include "ninfer/ops/gdn_input_proj.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/exl3/exl3_dense_dispatch.h"
#include "ops/linear/exl3/exl3_prefill.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t cols, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != cols ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("gdn_input_proj: invalid ") + label);
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

void require_nonoverlap(const Tensor& x, const Tensor& qkv, const Tensor& z) {
    if (overlaps(x, qkv) || overlaps(x, z) || overlaps(qkv, z)) {
        throw std::invalid_argument("gdn_input_proj: x, qkv, and z must not overlap");
    }
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("gdn_input_proj: invalid compute policy");
}

} // namespace

void gdn_input_proj_exl3(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                         LinearPolicy policy, MemoryArena& workspace, cudaStream_t stream) {
    validate_policy(policy);
    const std::int32_t cols = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("gdn_input_proj: T must be positive"); }
    // The dense EXL3 route (SP-7): the fused [16384, 5120] parent's row groups
    // (query, key, value, z) dispatch as row-sliced dense linears at decode token
    // shapes; larger token counts compose the decoded weight with BF16 linears.
    constexpr std::int32_t kHidden  = 5120;
    constexpr std::int32_t kQkvRows = 10240;
    constexpr std::int32_t kZRows   = 6144;
    constexpr std::int32_t kRows    = kQkvRows + kZRows;
    require_matrix(x, kHidden, cols, "x");
    require_matrix(qkv, kQkvRows, cols, "qkv");
    require_matrix(z, kZRows, cols, "z");
    require_nonoverlap(x, qkv, z);
    if (weight.n != kRows || weight.k != kHidden || weight.layout != QuantLayout::Exl3Dense) {
        throw std::invalid_argument("exl3 gdn_input_proj: unsupported weight shape");
    }
    constexpr std::int32_t kQRows = 2048;
    constexpr std::int32_t kKRows = 2048;
    constexpr std::int32_t kVRows = 6144;
    const Weight query_w = detail::exl3_dense_slice(weight, 0, kQRows);
    const Weight key_w   = detail::exl3_dense_slice(weight, kQRows, kKRows);
    const Weight value_w = detail::exl3_dense_slice(weight, kQRows + kKRows, kVRows);
    const Weight z_w     = detail::exl3_dense_slice(weight, kQkvRows, kZRows);
    // The linear routes write token-major [T, rows] outputs, while the caller's qkv
    // is the token-major [T, 10240] fused buffer: each row group projects into its
    // own contiguous staging plane and a 2D copy places it at the group's column
    // range of every token.
    auto scope = workspace.scope();
    const auto staging = [&](std::int32_t rows) {
        const auto span = workspace.alloc_bytes(
            static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * 2, 256);
        return Tensor(span.data, workspace.device(), DType::BF16, {rows, cols});
    };
    Tensor q_stage = staging(kQRows);
    Tensor k_stage = staging(kKRows);
    Tensor v_stage = staging(kVRows);
    const auto place = [&](const Tensor& stage, std::int32_t row_begin, std::int32_t rows) {
        CUDA_CHECK(cudaMemcpy2DAsync(
            static_cast<std::byte*>(qkv.data) + row_begin * 2, kQkvRows * 2, stage.data, rows * 2,
            rows * 2, cols, cudaMemcpyDeviceToDevice, stream));
    };
    if (cols <= 8) {
        detail::exl3_dense_dispatch(x, query_w, q_stage, &workspace, stream);
        detail::exl3_dense_dispatch(x, key_w, k_stage, &workspace, stream);
        detail::exl3_dense_dispatch(x, value_w, v_stage, &workspace, stream);
        detail::exl3_dense_dispatch(x, z_w, z, &workspace, stream);
    } else {
        const auto prime_span = workspace.alloc_bytes(
            static_cast<std::size_t>(kRows) * static_cast<std::size_t>(kHidden) * 2, 256);
        const auto u_span = workspace.alloc_bytes(
            static_cast<std::size_t>(kHidden) * static_cast<std::size_t>(cols) * 2, 256);
        Tensor prime(prime_span.data, workspace.device(), DType::BF16, {kRows, kHidden});
        Tensor u(u_span.data, workspace.device(), DType::BF16, {kHidden, cols});
        detail::exl3_prefill_dequant(weight, prime, stream);
        detail::exl3_prefill_transform(x, weight, u, stream);
        const auto linear_part = [&](std::int32_t row_begin, std::int32_t rows, Tensor& out) {
            Weight dense;
            dense.qtype  = QType::BF16;
            dense.layout = QuantLayout::Contiguous;
            dense.ndim   = 2;
            dense.n = dense.shape[0] = dense.padded_shape[0] = rows;
            dense.k = dense.shape[1] = dense.padded_shape[1] = kHidden;
            dense.qdata = static_cast<const std::byte*>(prime.data) +
                          static_cast<std::ptrdiff_t>(row_begin) * kHidden * 2;
            detail::bf16_dispatch(u, dense, out, policy, stream);
        };
        linear_part(0, kQRows, q_stage);
        linear_part(kQRows, kKRows, k_stage);
        linear_part(kQRows + kKRows, kVRows, v_stage);
        linear_part(kQkvRows, kZRows, z);
    }
    place(q_stage, 0, kQRows);
    place(k_stage, kQRows, kKRows);
    place(v_stage, kQRows + kKRows, kVRows);
}

std::size_t gdn_input_proj_exl3_workspace_capacity_bytes(std::int32_t parent_rows,
                                                         std::int32_t input_rows,
                                                         LinearPolicy policy,
                                                         std::int32_t min_tokens,
                                                         std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("gdn_input_proj workspace: invalid token interval");
    }
    if (parent_rows != 16384 || input_rows != 5120) {
        throw std::invalid_argument("gdn_input_proj workspace: unsupported EXL3 profile");
    }
    // The dense EXL3 route (SP-7): decode dispatches use the row slices' capacity;
    // the composed prefill route adds one decoded [16384, 5120] plane plus the
    // transformed activation. Both routes stage the three qkv row groups (the
    // token-major fused buffer receives 2D copies), adding one qkv-sized plane.
    const std::size_t staging =
        static_cast<std::size_t>(10240) * static_cast<std::size_t>(max_tokens) * 2;
    return std::max(detail::exl3_dense_workspace_bytes(6144, 5120, std::min(max_tokens, 8)) +
                        staging,
                    max_tokens > 8 ? detail::exl3_prefill_workspace_bytes(parent_rows, input_rows,
                                                                          max_tokens) +
                                         staging
                                   : std::size_t{0});
}

} // namespace ninfer::ops
