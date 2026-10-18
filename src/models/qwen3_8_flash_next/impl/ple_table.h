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
// Staged PLE values are token-major BF16, so one token occupies 5120 bytes.
inline constexpr std::int64_t kPleStageBytesPerToken = 2 * kPleEmbeddingDim;
inline constexpr std::int32_t kPleEosToken     = 248044;
inline constexpr std::uint64_t kPleRows        = 320001536;

// block_scale_k16_v1 plane geometry of the NVFP4 PLE table: 80 packed E2M1 code
// bytes per row, then the natural [rows, 10] E4M3FN scale plane (the code plane
// size is 256-aligned), then one FP32 scale_2 word.
inline constexpr std::uint64_t kPleNvfp4CodeBytesPerRow    = 80;
inline constexpr std::uint64_t kPleNvfp4CodePlaneBytes     = kPleRows * kPleNvfp4CodeBytesPerRow;
inline constexpr std::uint64_t kPleNvfp4ScalePlaneOffset   = kPleNvfp4CodePlaneBytes;
inline constexpr std::uint64_t kPleNvfp4ScaleBytesPerRow   = 10;
inline constexpr std::uint64_t kPleNvfp4Scale2Offset       =
    kPleNvfp4ScalePlaneOffset + kPleRows * kPleNvfp4ScaleBytesPerRow;
inline constexpr std::uint64_t kPleNvfp4TableBytes         = kPleNvfp4Scale2Offset + 4;

using PleIds = std::array<std::uint32_t, kPleHeads>;

/** Compute the exact bigram/trigram row IDs for one complete token sequence. */
void compute_ple_ids(std::span<const std::int32_t> tokens, std::span<PleIds> output);

/** Gather the 16 vendor-table FP8 head rows for each token and store their
 * values as exact BF16, token-major (5120 bytes/token). The table's own
 * embedding-scale multiplier stays out of the stored values. */
void gather_ple_fp8_as_bf16(const artifact::MappedRange& table, std::span<const PleIds> ids,
                            std::span<std::byte> output);

/** Decode one NVFP4 head row (80 packed E2M1 code bytes low-nibble-first, 10
 * E4M3FN group-scale bytes) into 160 BF16 values, exact on the BF16 grid. */
void decode_ple_nvfp4_row_bf16(std::span<const std::byte> codes,
                               std::span<const std::byte> scales,
                               std::span<std::byte> output_bf16);

/** Gather the 16 NVFP4 head rows for each token, decode them to BF16, and
 * store them token-major (5120 bytes/token). The table's scale_2 multiplier
 * stays out of the stored values; the consumer applies it as its embedding
 * scale, exactly like the vendor FP8 table's own scale object. */
void gather_ple_nvfp4_as_bf16(const artifact::MappedRange& table, std::span<const PleIds> ids,
                              std::span<std::byte> output);

/** Gather PLE rows for the given table format into token-major BF16 values
 * (5120 bytes/token). The vendor FP8 table's E4M3 codes convert exactly; the
 * NVFP4 table's E2M1 x E4M3-group-scale products are exact on the BF16 grid. */
inline void gather_ple_rows(const artifact::MappedRange& table, bool nvfp4,
                            std::span<const PleIds> ids, std::span<std::byte> output) {
    if (nvfp4) {
        gather_ple_nvfp4_as_bf16(table, ids, output);
    } else {
        gather_ple_fp8_as_bf16(table, ids, output);
    }
}

/** Read the table's trailing FP32 scale_2 word. */
float ple_nvfp4_scale2(const artifact::MappedRange& table);

} // namespace ninfer::models::qwen3_8_flash_next
