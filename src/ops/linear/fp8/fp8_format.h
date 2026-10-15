#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

struct Fp8WeightGeometry {
    std::uint64_t code_plane_bytes;
    std::uint64_t scale_plane_offset;
    std::uint64_t scale_plane_bytes;
    std::uint64_t required_payload_bytes;
};

Fp8WeightGeometry validate_fp8_weight(const Weight& weight, const char* operation);

// Linear weights that may be either BF16 or row-scale FP8 (the Flash-Next projections).
enum class WeightFormats { Bf16, Bf16OrFp8, Bf16Fp8OrMxfp8 };

inline bool fp8_row_weight(const Weight& weight) {
    return weight.qtype == QType::FP8_E4M3FN_ROW_BF16 && weight.layout == QuantLayout::RowScale;
}

inline bool mxfp8_block_weight(const Weight& weight) {
    return weight.qtype == QType::MXFP8 && weight.layout == QuantLayout::DenseBlockScaleK32Mxfp8;
}

inline void require_weight(const Weight& weight, int rows, int columns, const char* label,
                           WeightFormats formats = WeightFormats::Bf16) {
    const bool bf16 = weight.qtype == QType::BF16 && weight.layout == QuantLayout::Contiguous;
    const bool allowed =
        bf16 || (formats == WeightFormats::Bf16OrFp8 && fp8_row_weight(weight)) ||
        (formats == WeightFormats::Bf16Fp8OrMxfp8 &&
         (fp8_row_weight(weight) || mxfp8_block_weight(weight)));
    if (!allowed || weight.qdata == nullptr || weight.n != rows || weight.k != columns) {
        throw std::invalid_argument(label);
    }
}

} // namespace ninfer::ops::detail
