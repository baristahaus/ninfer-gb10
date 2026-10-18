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
    NVFP4                = 7,
    FP8_E4M3FN_ROW_BF16 = 8,
    FP8_E4M3FN = 9,
    EXL3                = 10,
    MXFP8                = 11,
};

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
    ExpertBlockScaleK16M128x4 = 4,
    Exl3ExpertBank      = 5,
    Exl3Dense           = 6,
    DenseBlockScaleK32Mxfp8 = 7,
    BlockScaleK16       = 8,
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

    // EXL3 expert bank and dense linear (SP-7; QType::EXL3, layouts Exl3ExpertBank and
    // Exl3Dense). `payload` addresses the complete encoded object; for a bank the per-expert
    // directory at its start resolves each expert's trellis extent and bitrate at runtime, so
    // mixed-width banks need no host-side tables; a dense linear is the one-expert case
    // without a directory (trellis at offset 0, `exl3_experts` = 1). Plane offsets are
    // payload-relative; the directory entry is {u64 offset, u32 extent, u8 2K, three zero
    // bytes} per expert.
    std::uint64_t exl3_trellis_offset = 0;
    std::uint64_t exl3_suh_offset      = 0;
    std::uint64_t exl3_svh_offset      = 0;
    std::uint32_t exl3_codebook        = 0;  // 0 = 3inst, 1 = mcg, 2 = mul1
    std::uint32_t exl3_experts         = 0;
    std::uint32_t exl3_rows_per_expert = 0;
    // Row-sliced dense views (SP-7 fused mixers): `exl3_row_stride` is the parent's n
    // (0 = this weight's own n, an unsliced complete object) and `exl3_k2` is the
    // explicit bitrate (0 = resolve from the dense extent at launch; a slice's byte
    // extent does not determine it). svh addresses shift by the slice's row begin.
    std::uint32_t exl3_k2         = 0;
    std::uint32_t exl3_row_stride = 0;
};

} // namespace ninfer
