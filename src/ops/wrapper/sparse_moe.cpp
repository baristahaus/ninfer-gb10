#include "core/weight.h"
#include "ninfer/ops/sparse_moe.h"

#include "core/nvtx.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/sparse_moe/decode/sparse_moe_decode.h"
#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHidden         = 2048;
constexpr std::int32_t kExperts        = 256;
constexpr std::int32_t kExpertRows     = 1024;
constexpr std::int32_t kIntermediate   = 512;
constexpr std::int32_t kRoutedGateRows = kExperts * kExpertRows;
constexpr std::int32_t kRoutedDownRows = kExperts * kHidden;

struct AddressRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end   = 0;
    std::string name;
};

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

AddressRange address_range(const void* pointer, std::size_t bytes, std::string name) {
    if (pointer == nullptr || bytes == 0) {
        throw std::invalid_argument("sparse_moe: " + name + " storage must be non-empty");
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::overflow_error("sparse_moe: " + name + " address range overflows");
    }
    return {begin, begin + bytes, std::move(name)};
}

void require_disjoint(const std::vector<AddressRange>& ranges) {
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        for (std::size_t j = i + 1; j < ranges.size(); ++j) {
            if (ranges[i].begin < ranges[j].end && ranges[j].begin < ranges[i].end) {
                throw std::invalid_argument("sparse_moe: " + ranges[i].name + " overlaps " +
                                            ranges[j].name);
            }
        }
    }
}

std::int32_t require_tensor(const Tensor& tensor, const char* name, std::int32_t hidden) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != hidden || tensor.ne[1] < 1 ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr || !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("sparse_moe: invalid ") + name);
    }
    return tensor.ne[1];
}

void require_matrix_metadata(const Weight& weight, std::int32_t n, std::int32_t k,
                             const char* name) {
    if (weight.ndim != 2 || weight.n != n || weight.k != k || weight.shape[0] != n ||
        weight.shape[1] != k || weight.shape[2] != 1 || weight.shape[3] != 1 ||
        weight.padded_shape[0] != n || weight.padded_shape[1] != k || weight.padded_shape[2] != 1 ||
        weight.padded_shape[3] != 1) {
        throw std::invalid_argument(std::string("sparse_moe: invalid shape for ") + name);
    }
}

void require_router(const Weight& weight, std::int32_t rows, std::int32_t hidden,
                    std::vector<AddressRange>& ranges) {
    require_matrix_metadata(weight, rows, hidden, "router_shared_gate");
    const std::size_t bytes = static_cast<std::size_t>(rows) * hidden * 2;
    if (weight.qtype != QType::BF16 || weight.layout != QuantLayout::Contiguous ||
        weight.qdata == nullptr || weight.qhigh != nullptr || weight.scales != nullptr ||
        weight.payload_bytes < bytes || !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument(
            "sparse_moe: router_shared_gate must be aligned contiguous BF16");
    }
    ranges.push_back(address_range(weight.qdata, bytes, "router_shared_gate"));
}

struct QuantGeometry {
    std::int32_t group_size;
    std::size_t code_bytes_per_group;
    std::size_t high_bytes_per_group;
};

QuantGeometry quant_geometry(QType qtype) {
    switch (qtype) {
    case QType::Q4_G64_FP16:
        return {64, 32, 0};
    case QType::Q5_G64_FP16:
        return {64, 32, 8};
    case QType::Q6_G64_FP16:
        return {64, 32, 16};
    case QType::Q8_G32_FP16:
        return {32, 32, 0};
    default:
        throw std::invalid_argument("sparse_moe: unsupported quantized weight format");
    }
}

// SP-7 EXL3 bank admission (contract section 7): the complete bank payload with its
// directory-resolved geometry, the resolved sparse-MoE shape, and the codebook admission —
// MUL1 and MCG first; 3INST banks are refused until a qualified route exists.
void require_exl3_bank(const Weight& weight, std::int32_t experts, std::int32_t rows_per_expert,
                       std::int32_t k, const char* name, std::vector<AddressRange>& ranges) {
    if (weight.qtype != QType::EXL3 || weight.layout != QuantLayout::Exl3ExpertBank ||
        weight.ndim != 2 || weight.payload == nullptr || weight.exl3_experts != experts ||
        weight.exl3_rows_per_expert != rows_per_expert || weight.n != experts * rows_per_expert ||
        weight.k != k || weight.exl3_codebook > 2 ||
        weight.exl3_trellis_offset % 256 != 0 || !aligned_to(weight.payload, 256)) {
        throw std::invalid_argument(std::string("sparse_moe: invalid EXL3 bank ") + name);
    }
    ranges.push_back(address_range(weight.payload, weight.payload_bytes, name));
}

void require_quantized(const Weight& weight, std::int32_t n, std::int32_t k, const char* name,
                       std::vector<AddressRange>& ranges) {
    require_matrix_metadata(weight, n, k, name);
    const QuantGeometry geometry       = quant_geometry(weight.qtype);
    const std::size_t groups           = static_cast<std::size_t>(n) * k / geometry.group_size;
    const std::size_t code_bytes       = groups * geometry.code_bytes_per_group;
    const std::size_t high_bytes       = groups * geometry.high_bytes_per_group;
    const std::size_t scale_bytes      = groups * 2;
    const std::size_t required_payload = code_bytes + high_bytes + scale_bytes;
    if (weight.layout != QuantLayout::RowSplit || weight.scale_dtype != DType::FP16 ||
        weight.group_size != static_cast<std::uint32_t>(geometry.group_size) ||
        weight.group != geometry.group_size || weight.qdata == nullptr ||
        weight.scales == nullptr || weight.payload_bytes < required_payload ||
        weight.high_plane_bytes < high_bytes || !aligned_to(weight.qdata, 16) ||
        !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument(std::string("sparse_moe: invalid row-split ") + name);
    }
    if ((high_bytes == 0 && weight.qhigh != nullptr) ||
        (high_bytes != 0 && (weight.qhigh == nullptr || !aligned_to(weight.qhigh, 16)))) {
        throw std::invalid_argument(std::string("sparse_moe: invalid high plane for ") + name);
    }
    ranges.push_back(address_range(weight.qdata, code_bytes, std::string(name) + " code"));
    if (high_bytes != 0) {
        ranges.push_back(address_range(weight.qhigh, high_bytes, std::string(name) + " high"));
    }
    ranges.push_back(address_range(weight.scales, scale_bytes, std::string(name) + " scales"));
}

// SP-7 dual-geometry admission: the EXL3 route resolves its geometry from the banks'
// directory; exactly the two contract geometries are admitted.
SparseMoeShape resolve_exl3_shape(const SparseMoeWeights& weights) {
    const int experts = weights.exl3_routed_gate.exl3_experts;
    if (experts == kSparseMoe35B.experts) {
        return kSparseMoe35B;
    }
    if (experts == kSparseMoe125B.experts) {
        return kSparseMoe125B;
    }
    throw std::invalid_argument("sparse_moe: unsupported EXL3 expert count " +
                                std::to_string(experts) + " (admitted: 256, 512)");
}

void require_fp8_shared(const Weight& weight, std::int32_t n, std::int32_t k, const char* name,
                        std::vector<AddressRange>& ranges) {
    require_matrix_metadata(weight, n, k, name);
    const detail::Fp8WeightGeometry geometry = detail::validate_fp8_weight(weight, name);
    ranges.push_back(
        address_range(weight.qdata, geometry.code_plane_bytes, std::string(name) + " code"));
    ranges.push_back(address_range(weight.scales, geometry.scale_plane_bytes,
                                   std::string(name) + " scales"));
}

void validate_weights(const SparseMoeWeights& weights, std::vector<AddressRange>& ranges,
                      SparseMoeShape& shape) {
    const bool exl3_route = weights.exl3_routed_gate.payload != nullptr;
    shape                 = exl3_route ? resolve_exl3_shape(weights) : kSparseMoe35B;
    require_router(weights.router_shared_gate, shape.experts + 1, shape.hidden, ranges);
    // SP-7 EXL3 route: a non-null gate bank payload marks the route; the fused and warm
    // banks stay null, and the route is untiered until the SP-4 variable-extent change.
    if (exl3_route) {
        if (weights.routed_gate_up.qdata != nullptr || weights.warm_routed_gate_up.qdata != nullptr ||
            weights.warm_routed_down.qdata != nullptr || weights.tiered_hot_experts != 0) {
            throw std::invalid_argument(
                "sparse_moe: the EXL3 route keeps the fused and warm banks null and untiered");
        }
        if (weights.exl3_routed_gate.exl3_codebook == 0 ||
            weights.exl3_routed_up.exl3_codebook == 0 || weights.routed_down.exl3_codebook == 0) {
            throw std::invalid_argument(
                "sparse_moe: 3INST EXL3 banks are refused until a qualified route exists");
        }
        require_exl3_bank(weights.exl3_routed_gate, shape.experts, shape.intermediate,
                          shape.hidden, "exl3_routed_gate", ranges);
        require_exl3_bank(weights.exl3_routed_up, shape.experts, shape.intermediate, shape.hidden,
                          "exl3_routed_up", ranges);
        require_exl3_bank(weights.routed_down, shape.experts, shape.hidden, shape.intermediate,
                          "exl3_routed_down", ranges);
        const bool fp8_shared = detail::fp8_row_weight(weights.shared_gate_up);
        if (fp8_shared) {
            if (!detail::fp8_row_weight(weights.shared_down)) {
                throw std::invalid_argument(
                    "sparse_moe: shared_down must be FP8_E4M3FN_ROW_BF16 RowScale to match the "
                    "FP8 shared_gate_up");
            }
            require_fp8_shared(weights.shared_gate_up, 2 * shape.intermediate, shape.hidden,
                               "shared_gate_up", ranges);
            require_fp8_shared(weights.shared_down, shape.hidden, shape.intermediate, "shared_down",
                               ranges);
        } else {
            if (weights.shared_gate_up.qtype != QType::Q8_G32_FP16 ||
                weights.shared_down.qtype != QType::Q8_G32_FP16) {
                throw std::invalid_argument(
                    "sparse_moe: shared weights must be Q8_G32_FP16 row-split or FP8 row-scale");
            }
            require_quantized(weights.shared_gate_up, 2 * shape.intermediate, shape.hidden,
                              "shared_gate_up", ranges);
            require_quantized(weights.shared_down, shape.hidden, shape.intermediate, "shared_down",
                              ranges);
        }
        return;
    }
    if (weights.routed_gate_up.qtype != QType::Q4_G64_FP16 &&
        weights.routed_gate_up.qtype != QType::Q8_G32_FP16) {
        throw std::invalid_argument("sparse_moe: routed_gate_up must be Q4 or Q8");
    }
    if (weights.routed_down.qtype != QType::Q5_G64_FP16 &&
        weights.routed_down.qtype != QType::Q6_G64_FP16 &&
        weights.routed_down.qtype != QType::Q8_G32_FP16) {
        throw std::invalid_argument("sparse_moe: routed_down must be Q5, Q6, or Q8");
    }
    if (weights.shared_gate_up.qtype != QType::Q8_G32_FP16 ||
        weights.shared_down.qtype != QType::Q8_G32_FP16) {
        throw std::invalid_argument("sparse_moe: shared weights must be Q8");
    }
    require_quantized(weights.shared_gate_up, 2 * kIntermediate, kHidden, "shared_gate_up", ranges);
    require_quantized(weights.shared_down, kHidden, kIntermediate, "shared_down", ranges);
    // SP-4 phase 2a: a tiered bank splits its routed experts at `tiered_hot_experts`. The
    // device banks carry the hot expert prefix; the pinned-host warm twins carry the suffix
    // in expert order. An untiered bank keeps full-bank device shapes and null warm twins.
    const auto warm_gate_up_null = weights.warm_routed_gate_up.qdata == nullptr &&
                                   weights.warm_routed_gate_up.scales == nullptr;
    const auto warm_down_null    = weights.warm_routed_down.qdata == nullptr &&
                                  weights.warm_routed_down.scales == nullptr;
    if (weights.tiered_hot_experts == 0) {
        if (!warm_gate_up_null || !warm_down_null) {
            throw std::invalid_argument("sparse_moe: warm twins require a tiered boundary");
        }
        require_quantized(weights.routed_gate_up, kRoutedGateRows, kHidden, "routed_gate_up",
                          ranges);
        require_quantized(weights.routed_down, kRoutedDownRows, kIntermediate, "routed_down",
                          ranges);
        return;
    }
    const auto hot  = static_cast<std::int32_t>(weights.tiered_hot_experts);
    const auto warm = kExperts - hot;
    if (hot <= 0 || hot >= kExperts) {
        throw std::invalid_argument("sparse_moe: tiered boundary must be a strict expert prefix");
    }
    if (warm_gate_up_null || warm_down_null) {
        throw std::invalid_argument("sparse_moe: tiered banks require warm twins");
    }
    require_quantized(weights.routed_gate_up, hot * kExpertRows, kHidden, "routed_gate_up",
                      ranges);
    require_quantized(weights.routed_down, hot * kHidden, kIntermediate, "routed_down", ranges);
    require_quantized(weights.warm_routed_gate_up, warm * kExpertRows, kHidden,
                      "warm_routed_gate_up", ranges);
    require_quantized(weights.warm_routed_down, warm * kHidden, kIntermediate, "warm_routed_down",
                      ranges);
}

} // namespace

std::size_t sparse_moe_workspace_capacity_bytes(QType routed_gate_up, QType routed_down,
                                                std::int32_t min_tokens, std::int32_t max_tokens,
                                                const SparseMoeShape& shape) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("sparse_moe workspace: invalid token interval");
    }
    if (routed_gate_up == QType::EXL3 || routed_down == QType::EXL3) {
        if (routed_gate_up != QType::EXL3 || routed_down != QType::EXL3) {
            throw std::invalid_argument("sparse_moe workspace: EXL3 banks must pair");
        }
        // The wrapper loops the single-token decode per prefill column; the decode
        // workspace is reused per token, so the capacity is interval-independent.
        return detail::resolve_sparse_moe_decode_plan(routed_gate_up, routed_down, {}, shape)
            .workspace_bytes;
    }
    (void)detail::resolve_sparse_moe_decode_plan(routed_gate_up, routed_down);

    const bool q8_profile =
        routed_gate_up == QType::Q8_G32_FP16 && routed_down == QType::Q8_G32_FP16;
    const std::int32_t prefill_first =
        q8_profile ? detail::kSparseMoePrefillQ8Q8Min
                   : (routed_down == QType::Q5_G64_FP16 ? detail::kSparseMoePrefillQ4Q5Min
                                                        : detail::kSparseMoePrefillQ4Q6Min);

    std::size_t required = 0;
    if (min_tokens == 1) { required = detail::sparse_moe_decode_workspace_bytes(); }

    const std::int32_t small_first = std::max(min_tokens, detail::kSparseMoeSmallTMin);
    const std::int32_t small_last =
        std::min({max_tokens, detail::kSparseMoeSmallTMax, prefill_first - 1});
    if (small_first <= small_last) {
        required = std::max(required, detail::sparse_moe_small_t_workspace_bytes(small_last));
    }

    const std::int32_t prefill_interval_first = std::max(min_tokens, prefill_first);
    if (prefill_interval_first <= max_tokens) {
        required = std::max(required, detail::sparse_moe_prefill_workspace_bytes(max_tokens));
    }
    return required;
}

void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, MemoryArena& workspace, cudaStream_t stream) {
    sparse_moe(x, weights, epilogue, destination, SparseMoeHints{}, workspace, stream);
}

void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, const SparseMoeHints& hints, MemoryArena& workspace,
                cudaStream_t stream) {
    if (epilogue != SparseMoeEpilogue::AddResidual) {
        throw std::invalid_argument("sparse_moe: unsupported epilogue");
    }
    const bool exl3_route = weights.exl3_routed_gate.payload != nullptr;
    SparseMoeShape shape = exl3_route ? resolve_exl3_shape(weights) : kSparseMoe35B;
    const std::int32_t tokens = require_tensor(x, "x", shape.hidden);
    if (require_tensor(destination, "destination", shape.hidden) != tokens) {
        throw std::invalid_argument("sparse_moe: x and destination token counts must match");
    }

    std::vector<AddressRange> ranges;
    ranges.reserve(16);
    ranges.push_back(address_range(x.data, x.bytes(), "x"));
    ranges.push_back(address_range(destination.data, destination.bytes(), "destination"));
    validate_weights(weights, ranges, shape);

    // The EXL3 route has no batched kernels yet; the shared token loop below runs the
    // single-token decode per column, which is numerically identical (routing is
    // per-token independent).
    const bool use_small_t = !exl3_route && detail::sparse_moe_uses_small_t(tokens);
    const bool use_prefill =
        !exl3_route && detail::sparse_moe_uses_prefill(tokens, weights.routed_gate_up.qtype,
                                                       weights.routed_down.qtype);
    nvtx::ScopedRange moe_range(use_prefill   ? nvtx::Name::SparseMoePrefill
                                : use_small_t ? nvtx::Name::SparseMoeSmallT
                                              : nvtx::Name::SparseMoeDecode,
                                nvtx::Category::Moe, static_cast<std::uint64_t>(tokens));
    std::size_t required = 0;
    if (use_prefill) {
        required = detail::resolve_sparse_moe_prefill_plan(tokens, weights.routed_gate_up.qtype,
                                                           weights.routed_down.qtype)
                       .workspace_bytes;
    } else if (use_small_t) {
        required = detail::resolve_sparse_moe_small_t_plan(tokens, weights.routed_gate_up.qtype,
                                                           weights.routed_down.qtype)
                       .workspace_bytes;
    } else {
        required = detail::resolve_sparse_moe_decode_plan(
                       exl3_route ? QType::EXL3 : weights.routed_gate_up.qtype,
                       exl3_route ? QType::EXL3 : weights.routed_down.qtype, SparseMoeHints{}, shape)
                       .workspace_bytes;
    }
    if (workspace.storage().data == nullptr || workspace.capacity() < required ||
        workspace.used() > workspace.capacity() - required) {
        throw std::invalid_argument("sparse_moe: insufficient workspace capacity");
    }
    ranges.push_back(address_range(workspace.storage().data, workspace.capacity(), "workspace"));
    require_disjoint(ranges);

    auto scope = workspace.scope();
    if (use_prefill) {
        const detail::SparseMoePrefillPlan plan = detail::resolve_sparse_moe_prefill_plan(
            tokens, weights.routed_gate_up.qtype, weights.routed_down.qtype);
        const detail::SparseMoePrefillWorkspace views =
            detail::allocate_sparse_moe_prefill_workspace(workspace, plan.slice_tokens);
        detail::sparse_moe_prefill_launch(x, weights, destination, plan, views, stream);
        return;
    }
    if (use_small_t) {
        const detail::SparseMoeSmallTPlan plan = detail::resolve_sparse_moe_small_t_plan(
            tokens, weights.routed_gate_up.qtype, weights.routed_down.qtype);
        const detail::SparseMoeSmallTWorkspace views =
            detail::allocate_sparse_moe_small_t_workspace(workspace, tokens);
        detail::sparse_moe_small_t_launch(x, weights, destination, plan, views, stream);
        return;
    }

    const detail::SparseMoeDecodePlan plan = detail::resolve_sparse_moe_decode_plan(
        exl3_route ? QType::EXL3 : weights.routed_gate_up.qtype,
        exl3_route ? QType::EXL3 : weights.routed_down.qtype, hints, shape);
    const detail::SparseMoeDecodeWorkspace views =
        detail::allocate_sparse_moe_decode_workspace(workspace, exl3_route, shape);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const Tensor x_column     = x.slice(1, token, 1);
        Tensor destination_column = destination.slice(1, token, 1);
        detail::sparse_moe_decode_launch(x_column, weights, destination_column, plan, views, shape,
                                         stream);
    }
}

} // namespace ninfer::ops
