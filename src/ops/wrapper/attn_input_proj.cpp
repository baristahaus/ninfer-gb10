#include "core/weight.h"
#include "ninfer/ops/attn_input_proj.h"

#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"
#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"
#include "ops/attn_input_proj/q8/q8_attn_input_plan.h"
#include "ops/linear/exl3/exl3_dense_dispatch.h"
#include "ops/linear/exl3/exl3_prefill.h"
#include "ops/linear/kernels/fp8/fp8_geometry.h"
#include "ops/linear/kernels/fp8/fp8_format.h"
#include "ops/linear/kernels/nvfp4/nvfp4_layout.h"
#include "ops/linear/nvfp4/nvfp4_format.h"

#include <cstddef>
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
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_rowsplit(const Weight& weight, QType qtype, std::int32_t rows, const char* label) {
    const bool q4_planes =
        qtype != QType::Q4_G64_FP16 || (weight.qhigh == nullptr && weight.high_plane_bytes == 0);
    const bool q5_planes =
        qtype != QType::Q5_G64_FP16 || (weight.qhigh != nullptr && weight.high_plane_bytes != 0);
    if (weight.qtype != qtype || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 64 || weight.group != 64 ||
        weight.ndim != 2 || weight.n != rows || weight.k != 5120 || weight.shape[0] != rows ||
        weight.shape[1] != 5120 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 5120 || !q4_planes || !q5_planes ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 4) ||
        (qtype == QType::Q5_G64_FP16 && !aligned_to(weight.qhigh, 16))) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_q8_rowsplit(const Weight& weight, std::int32_t rows, std::int32_t hidden,
                         const char* label) {
    if (weight.qtype != QType::Q8_G32_FP16 || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        weight.ndim != 2 || weight.n != rows || weight.k != hidden || weight.shape[0] != rows ||
        weight.shape[1] != hidden || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != hidden || weight.qhigh != nullptr ||
        weight.high_plane_bytes != 0 || !aligned_to(weight.qdata, 16) ||
        !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_bf16_contiguous(const Weight& weight, std::int32_t rows, std::int32_t hidden,
                             const char* label) {
    const std::uint64_t payload_bytes = static_cast<std::uint64_t>(rows) *
                                        static_cast<std::uint64_t>(hidden) * sizeof(std::uint16_t);
    if (weight.qtype != QType::BF16 || weight.layout != QuantLayout::Contiguous ||
        weight.payload_bytes < payload_bytes || weight.high_plane_bytes != 0 || weight.ndim != 2 ||
        weight.n != rows || weight.k != hidden || weight.shape[0] != rows ||
        weight.shape[1] != hidden || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != hidden || weight.qhigh != nullptr || weight.scales != nullptr ||
        weight.group_size != 0 || weight.group != 0 || !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("attn_input_proj: invalid compute policy");
}

void dispatch_single_parent(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, LinearPolicy policy, MemoryArena* workspace,
                            cudaStream_t stream) {
    validate_policy(policy);
    if (weight.qtype == QType::BF16) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        require_bf16_contiguous(weight, kRows, kHidden, "query/key/gate/value weight");
        detail::bf16_attn_input_dispatch(x, weight, q, gate, k, v, stream);
        return;
    }

    if (weight.qtype == QType::NVFP4) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        detail::validate_nvfp4_weight(weight, "nvfp4 attn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("nvfp4 attn_input_proj: unsupported weight shape");
        }
        detail::nvfp4_attn_input_dispatch(x, weight, q, gate, k, v, policy, workspace, stream);
        return;
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        detail::validate_fp8_weight(weight, "fp8 attn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("fp8 attn_input_proj: unsupported weight shape");
        }
        detail::fp8_attn_input_dispatch(x, weight, q, gate, k, v, policy, workspace, stream);
        return;
    }

    if (weight.qtype == QType::EXL3) {
        // The dense EXL3 route (SP-7): the fused [14336, 5120] parent's row groups
        // (query, key, gate, value) dispatch as row-sliced dense linears at decode token
        // shapes; larger token counts compose the decoded weight with BF16 linears.
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        if (weight.n != kRows || weight.k != kHidden || weight.layout != QuantLayout::Exl3Dense) {
            throw std::invalid_argument("exl3 attn_input_proj: unsupported weight shape");
        }
        if (workspace == nullptr) {
            throw std::invalid_argument("exl3 attn_input_proj requires caller workspace");
        }
        const Weight query_w = detail::exl3_dense_slice(weight, 0, kQRows);
        const Weight key_w = detail::exl3_dense_slice(weight, kQRows, kKvRows);
        const Weight gate_w = detail::exl3_dense_slice(weight, kQRows + kKvRows, kQRows);
        const Weight value_w =
            detail::exl3_dense_slice(weight, kQRows + kKvRows + kQRows, kKvRows);
        if (cols <= 8) {
            auto scope = workspace->scope();
            detail::exl3_dense_dispatch(x, query_w, q, workspace, stream);
            detail::exl3_dense_dispatch(x, key_w, k, workspace, stream);
            detail::exl3_dense_dispatch(x, gate_w, gate, workspace, stream);
            detail::exl3_dense_dispatch(x, value_w, v, workspace, stream);
            return;
        }
        auto scope = workspace->scope();
        const auto prime_span = workspace->alloc_bytes(
            static_cast<std::size_t>(kRows) * static_cast<std::size_t>(kHidden) * 2, 256);
        const auto u_span = workspace->alloc_bytes(
            static_cast<std::size_t>(kHidden) * static_cast<std::size_t>(cols) * 2, 256);
        Tensor prime(prime_span.data, DType::BF16, {kRows, kHidden});
        Tensor u(u_span.data, DType::BF16, {kHidden, cols});
        detail::exl3_prefill_dequant(weight, prime, stream);
        detail::exl3_prefill_transform(x, weight, u, stream);
        const auto linear_part = [&](std::int32_t row_begin, std::int32_t rows, Tensor& out) {
            Weight dense;
            dense.qtype   = QType::BF16;
            dense.layout  = QuantLayout::Contiguous;
            dense.ndim    = 2;
            dense.n = dense.shape[0] = dense.padded_shape[0] = rows;
            dense.k = dense.shape[1] = dense.padded_shape[1] = kHidden;
            dense.qdata = static_cast<const std::byte*>(prime.data) +
                          static_cast<std::ptrdiff_t>(row_begin) * kHidden * 2;
            detail::bf16_dispatch(u, dense, out, policy, stream);
        };
        linear_part(0, kQRows, q);
        linear_part(kQRows, kKvRows, k);
        linear_part(kQRows + kKvRows, kQRows, gate);
        linear_part(2 * kQRows + kKvRows, kKvRows, v);
        return;
    }

    constexpr std::int32_t kHidden = 2048;
    constexpr std::int32_t kQRows  = 4096;
    constexpr std::int32_t kKvRows = 512;
    constexpr std::int32_t kRows   = 9216;
    const std::int32_t cols        = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(gate, kQRows, cols, "gate");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_q8_rowsplit(weight, kRows, kHidden, "query/key/gate/value weight");
    detail::q8_attn_input_dispatch(x, weight, q, gate, k, v, stream);
}

} // namespace

std::size_t attn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                                     std::int32_t input_rows, LinearPolicy policy,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("attn_input_proj workspace: invalid token interval");
    }

    switch (parent_qtype) {
    case QType::BF16:
        if (parent_rows != 14336 || input_rows != 5120) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported BF16 profile");
        }
        return 0;
    case QType::NVFP4:
        if (parent_rows != detail::Nvfp4N14336K5120::kOutputRows ||
            input_rows != detail::Nvfp4N14336K5120::kInputRows) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_attn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    case QType::FP8_E4M3FN_ROW_BF16:
        if (parent_rows != detail::Fp8N14336K5120::kOutputRows ||
            input_rows != detail::Fp8N14336K5120::kInputRows) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported FP8 profile");
        }
        return detail::fp8_attn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    case QType::Q8_G32_FP16:
        if (parent_rows != 9216 || input_rows != 2048) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported Q8 profile");
        }
        (void)detail::q8_attn_input_resolve_plan(
            {input_rows, 4096, 512, parent_rows, input_rows, min_tokens});
        (void)detail::q8_attn_input_resolve_plan(
            {input_rows, 4096, 512, parent_rows, input_rows, max_tokens});
        return 0;
    case QType::EXL3:
        if (parent_rows != 14336 || input_rows != 5120) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported EXL3 profile");
        }
        // The dense EXL3 route (SP-7): decode-shape dispatches use the row slices'
        // capacity; the composed prefill route adds one decoded [14336, 5120] plane and
        // the transformed activation.
        return std::max(detail::exl3_dense_workspace_bytes(6144, 5120, std::min(max_tokens, 8)),
                        max_tokens > 8 ? detail::exl3_prefill_workspace_bytes(
                                             parent_rows, input_rows, max_tokens)
                                       : std::size_t{0});
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::FP32:
    case QType::INT32:
        break;
    }
    throw std::invalid_argument("attn_input_proj workspace: unsupported parent qtype");
}

void attn_input_proj(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     cudaStream_t stream) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    const std::int32_t cols        = x.ne[1];
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(gate, kQRows, cols, "gate");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_rowsplit(query_key_weight, QType::Q4_G64_FP16, kQRows + kKvRows, "query/key weight");
    require_rowsplit(gate_value_weight, QType::Q5_G64_FP16, kQRows + kKvRows, "gate/value weight");

    detail::q4_q5_attn_input_dispatch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                      stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                     MemoryArena& workspace, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_gate_value_weight, q, gate, k, v, policy, &workspace,
                           stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_gate_value_weight, q, gate, k, v, LinearPolicy::A16Only,
                           nullptr, stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_value_weight, Tensor& q, Tensor& k,
                     Tensor& v, cudaStream_t stream) {
    constexpr std::int32_t kQRows  = 4096;
    constexpr std::int32_t kKvRows = 1024;
    constexpr std::int32_t kRows   = 6144;
    const std::int32_t hidden      = x.ne[0];
    const std::int32_t cols        = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
    if (hidden != 2048 && hidden != 5120) {
        throw std::invalid_argument("attn_input_proj: unsupported Q8 Q/K/V profile");
    }
    require_matrix(x, hidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_q8_rowsplit(query_key_value_weight, kRows, hidden, "query/key/value weight");

    detail::q8_attn_input_dispatch(x, query_key_value_weight, q, k, v, stream);
}

} // namespace ninfer::ops
