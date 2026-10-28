#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "ops/linear/common/weight_validation.h"

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

inline bool mxfp8_block_weight(const Weight& weight) {
    return weight.qtype == QType::MXFP8 && weight.layout == QuantLayout::DenseBlockScaleK32Mxfp8;
}

} // namespace ninfer::ops::detail
