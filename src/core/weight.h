#pragma once

#include "core/dtype.h"

#include <cstdint>

namespace ninfer {

enum class QType : std::uint16_t {
    Q4_G64_FP16         = 0,
    Q5_G64_FP16         = 1,
    Q6_G64_FP16         = 2,
    Q8_G32_FP16         = 3,
    BF16                = 4,
    FP32                = 5,
    INT32               = 6,
    NVFP4               = 7,
    FP8_E4M3FN_ROW_BF16 = 8,
    FP8_E4M3FN = 9,
    EXL3                = 10,
};

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
    ExpertBlockScaleK16M128x4 = 4,
    Exl3ExpertBank      = 5,
};

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4_G64_FP16;
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata          = nullptr;
    const void* qhigh          = nullptr;
    const void* scales         = nullptr;
    std::int32_t n             = 0;
    std::int32_t k             = 0;
    std::int32_t group         = 0;
    QuantLayout layout         = QuantLayout::RowSplit;
    DType scale_dtype          = DType::FP32;
    std::int32_t scale_ne[4]   = {1, 1, 1, 1};
    std::int64_t scale_nb[4]   = {0, 0, 0, 0};
    float weight_scale_divisor = 0.0F;
    float input_scale_divisor  = 0.0F;

    // EXL3 expert bank (SP-7; QType::EXL3, layout Exl3ExpertBank). `payload` addresses the
    // complete bank; the per-expert directory at its start resolves each expert's trellis
    // extent and bitrate at runtime, so mixed-width banks need no host-side tables. Plane
    // offsets are payload-relative; the directory entry is {u64 offset, u32 extent, u8 2K,
    // three zero bytes} per expert.
    std::uint64_t exl3_trellis_offset = 0;
    std::uint64_t exl3_suh_offset      = 0;
    std::uint64_t exl3_svh_offset      = 0;
    std::uint32_t exl3_codebook        = 0;  // 0 = 3inst, 1 = mcg, 2 = mul1
    std::uint32_t exl3_experts         = 0;
    std::uint32_t exl3_rows_per_expert = 0;
};

} // namespace ninfer
