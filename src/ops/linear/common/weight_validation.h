#pragma once

// Weight-geometry validation helpers shared by the Flash-Next op routes. These live here, not
// in a per-format kernel header, because several routes (GDN, QSA, embedding, KV
// materialization) validate the same mixed BF16/row-scale-FP8 projection contract.
#include "core/tensor.h"
#include "core/weight.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// Linear weights that may be BF16, row-scale FP8, or MXFP8 block-scale (the Flash-Next
// projections).
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
