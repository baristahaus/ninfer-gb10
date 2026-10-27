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

using PleIds = std::array<std::uint32_t, kPleHeads>;

/** Compute the exact bigram/trigram row IDs for one complete token sequence. */
void compute_ple_ids(std::span<const std::int32_t> tokens, std::span<PleIds> output);

/** Gather the 16 FP8 head rows for each token into token-major contiguous storage. */
void gather_ple_fp8(const artifact::MappedRange& table, std::span<const PleIds> ids,
                    std::span<std::byte> output);

} // namespace ninfer::models::qwen3_8_flash_next
