#pragma once

#include "artifact/reader.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::models::qwen3_8_flash_next {

inline constexpr std::int32_t kPleHeads        = 16;
inline constexpr std::int32_t kPleHeadWidth    = 160;
inline constexpr std::int32_t kPleEmbeddingDim = kPleHeads * kPleHeadWidth;
inline constexpr std::int32_t kPleEosToken     = 248044;
inline constexpr std::uint64_t kPleRows        = 320001536;

// block_scale_k16_v1 plane geometry of the NVFP4 PLE table: 80 packed E2M1 code
// bytes per row, then the natural [rows, 10] E4M3FN group-scale plane (the code
// plane size is 256-aligned), then one trailing FP32 scale_2 word.
inline constexpr std::uint64_t kPleNvfp4CodeBytesPerRow  = 80;
inline constexpr std::uint64_t kPleNvfp4CodePlaneBytes   = kPleRows * kPleNvfp4CodeBytesPerRow;
inline constexpr std::uint64_t kPleNvfp4ScalePlaneOffset = kPleNvfp4CodePlaneBytes;
inline constexpr std::uint64_t kPleNvfp4ScaleBytesPerRow = 10;
inline constexpr std::uint64_t kPleNvfp4Scale2Offset =
    kPleNvfp4ScalePlaneOffset + kPleRows * kPleNvfp4ScaleBytesPerRow;
inline constexpr std::uint64_t kPleNvfp4TableBytes = kPleNvfp4Scale2Offset + 4;

using PleIds = std::array<std::uint32_t, kPleHeads>;

/** Compute the exact bigram/trigram row IDs for one complete token sequence. */
void compute_ple_ids(std::span<const std::int32_t> tokens, std::span<PleIds> output);

/** Gather the 16 head rows for each token into token-major contiguous FP8
 * storage (kPleEmbeddingDim bytes per token). The vendor FP8 table's E4M3
 * codes are staged unchanged; the NVFP4 group-16 table's rows are decoded to
 * FP8 with the table's scale_2 folded into the staged values (the gather
 * consumer route for block_scale_k16_v1). */
void gather_ple_fp8(const artifact::MappedRange& table, std::span<const PleIds> ids,
                    std::span<std::byte> output);

/** Gather the 16 NVFP4 head rows for each token, decode them to FP8 with the
 * table's scale_2 folded in, and store them token-major (2560 bytes/token). */
void gather_ple_nvfp4(const artifact::MappedRange& table, std::span<const PleIds> ids,
                      std::span<std::byte> output);

/** Read the table's trailing FP32 scale_2 word. */
float ple_nvfp4_scale2(const artifact::MappedRange& table);

/** Decode one NVFP4 head row (80 packed E2M1 code bytes low-nibble-first, 10
 * E4M3FN group-scale bytes) into 160 BF16 values, exact on the BF16 grid and
 * without the table's scale_2 multiplier. A NaN group scale throws. */
void decode_ple_nvfp4_row_bf16(std::span<const std::byte> codes, std::span<const std::byte> scales,
                               std::span<std::byte> output_bf16);

} // namespace ninfer::models::qwen3_8_flash_next
