#include "models/qwen3_8_flash_next/impl/ple_table.h"

#include "core/performance.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_8_flash_next {
namespace {

constexpr std::array<std::uint64_t, 3> kMultipliers = {
    23703573157769ULL,
    20109073645365ULL,
    8052911324071ULL,
};
constexpr std::array<std::uint32_t, kPleHeads> kSizes = {
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
    20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
};
constexpr std::array<std::uint32_t, kPleHeads> kOffsets = {
    0,         20000003,  40000026,  60000059,  80000106,  100000165, 120000228, 140000297,
    160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275,
};

std::uint32_t positive_remainder(std::uint64_t bits, std::uint32_t modulus) noexcept {
    const auto signed_value   = std::bit_cast<std::int64_t>(bits);
    const auto signed_modulus = static_cast<std::int64_t>(modulus);
    std::int64_t result       = signed_value % signed_modulus;
    if (result < 0) { result += signed_modulus; }
    return static_cast<std::uint32_t>(result);
}

std::uint64_t product(std::int32_t token, std::uint64_t multiplier) noexcept {
    return static_cast<std::uint64_t>(static_cast<std::int64_t>(token)) * multiplier;
}

float decode_e4m3_host(unsigned word) noexcept {
    const unsigned exponent = (word >> 3) & 0xF;
    const unsigned fraction = word & 7;
    if (exponent == 15 && fraction == 7) { return std::numeric_limits<float>::quiet_NaN(); }
    const std::uint32_t sign = static_cast<std::uint32_t>(word & 0x80) << 24;
    if (exponent == 0) {
        // Subnormal: fraction * 2^-9.
        return std::bit_cast<float>(
            sign | std::bit_cast<std::uint32_t>(static_cast<float>(fraction) * 0x1p-9F));
    }
    return std::bit_cast<float>(sign | ((exponent + 120U) << 23) | (fraction << 20));
}

float decode_e2m1_host(unsigned nibble) noexcept {
    static constexpr float kMagnitudes[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    const float magnitude                 = kMagnitudes[nibble & 7];
    return (nibble & 8) != 0 ? -magnitude : magnitude;
}

std::uint16_t bf16_bits_rn(float value) noexcept {
    const std::uint32_t bits    = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t lsb     = (bits >> 16) & 1;
    const std::uint32_t rounded = (bits + 0x7FFFU + lsb) & 0xFFFF0000U;
    return static_cast<std::uint16_t>(rounded >> 16);
}

constexpr std::uint32_t round_rne(std::uint32_t value, unsigned shift) noexcept {
    // value >> shift with round-to-nearest-even on the dropped bits; shift in [1, 24].
    const std::uint32_t half      = 1U << (shift - 1);
    std::uint32_t quotient        = value >> shift;
    const std::uint32_t remainder = value & ((1U << shift) - 1U);
    if (remainder > half || (remainder == half && (quotient & 1)) != 0U) { ++quotient; }
    return quotient;
}

// Round-to-nearest-even E4M3FN encode of a finite value, saturating past 448
// exactly like the hardware convert (the staged FP8 rows must never produce
// the reserved 0x7F/0xFF words).
std::uint8_t encode_e4m3_rn_satfinite(float value) noexcept {
    const auto bits               = std::bit_cast<std::uint32_t>(value);
    const bool negative           = (bits >> 31) != 0;
    const std::uint32_t magnitude = bits & 0x7fff'ffff;
    const std::uint32_t sign      = static_cast<std::uint32_t>(negative) << 7;
    if (magnitude < 0x3a80'0000) { return static_cast<std::uint8_t>(sign); } // |value| < 2^-10
    if (magnitude >= 0x7f80'0000 || magnitude >= 0x43e8'0000) {              // inf/NaN or >= 464
        return static_cast<std::uint8_t>(sign | 0x7E);
    }
    const int exponent           = static_cast<int>(magnitude >> 23) - 127;
    const std::uint32_t mantissa = magnitude & 0x007f'ffff;
    if (exponent < -6) {
        // Subnormal target: quantize |value| / 2^-9 with round-to-nearest-even; the exponent
        // floor above keeps the shift in [21, 24].
        const std::uint32_t quantum = round_rne(0x0080'0000U | mantissa, 14 - exponent);
        if (quantum == 8) { return static_cast<std::uint8_t>(sign | 0x08); } // the minimum normal
        return static_cast<std::uint8_t>(sign | quantum);
    }
    std::uint32_t top             = mantissa >> 20;
    const std::uint32_t remainder = mantissa & 0x000f'ffff;
    if (remainder > 0x0008'0000U || (remainder == 0x0008'0000U && (top & 1)) != 0U) { ++top; }
    int biased = exponent + 7;
    if (top == 8) { // the rounding carried into the next exponent
        top = 0;
        ++biased;
    }
    if (biased > 15 || (biased == 15 && top == 7)) {
        return static_cast<std::uint8_t>(sign | 0x7E);
    }
    return static_cast<std::uint8_t>(sign | (static_cast<std::uint32_t>(biased) << 3) | top);
}

} // namespace

void compute_ple_ids(std::span<const std::int32_t> tokens, std::span<PleIds> output) {
    NINFER_PERF_SCOPE("ninfer.host/1|ple.hash");

    if (tokens.size() != output.size()) {
        throw std::invalid_argument("PLE token and output lengths must match");
    }
    std::int32_t previous          = kPleEosToken;
    std::int32_t previous2         = kPleEosToken;
    std::uint32_t segment_position = 0;
    for (std::size_t position = 0; position < tokens.size(); ++position) {
        const std::int32_t current  = tokens[position];
        const std::int32_t shifted1 = segment_position >= 1 ? previous : kPleEosToken;
        const std::int32_t shifted2 = segment_position >= 2 ? previous2 : kPleEosToken;
        const std::uint64_t first   = product(current, kMultipliers[0]);
        const std::uint64_t bigram  = first ^ product(shifted1, kMultipliers[1]);
        const std::uint64_t trigram = bigram ^ product(shifted2, kMultipliers[2]);
        for (std::size_t head = 0; head < 8; ++head) {
            output[position][head] = kOffsets[head] + positive_remainder(bigram, kSizes[head]);
        }
        for (std::size_t head = 8; head < kPleHeads; ++head) {
            output[position][head] = kOffsets[head] + positive_remainder(trigram, kSizes[head]);
        }

        previous2 = previous;
        previous  = current;
        if (current == kPleEosToken) {
            segment_position = 0;
        } else if (segment_position != std::numeric_limits<std::uint32_t>::max()) {
            ++segment_position;
        }
    }
}

void gather_ple_fp8(const artifact::MappedRange& table, std::span<const PleIds> ids,
                    std::span<std::byte> output) {
    NINFER_PERF_SCOPE("ninfer.host/1|ple.gather");

    if (table.size() == kPleNvfp4TableBytes) { return gather_ple_nvfp4(table, ids, output); }
    constexpr std::uint64_t table_bytes = kPleRows * kPleHeadWidth;
    if (table.size() != table_bytes) {
        throw std::invalid_argument("PLE table has the wrong byte length");
    }
    const std::uint64_t required = static_cast<std::uint64_t>(ids.size()) * kPleEmbeddingDim;
    if (output.size() != required) {
        throw std::invalid_argument("PLE gathered output has the wrong byte length");
    }
    // The table does not fit in page cache next to the weights on a unified-memory device, and
    // hashed rows are scattered, so an uncached row is one page fault. Copying row by row would
    // serve those faults one at a time. Instead the reads of the next kPrefetchRows rows are
    // started ahead of the copy, so their I/O overlaps; the window bounds the advice calls in
    // flight for a long prefill chunk.
    constexpr std::size_t kPrefetchRows = 512;
    const std::size_t rows              = ids.size() * kPleHeads;
    const auto row_id                   = [&ids](std::size_t index) -> std::uint64_t {
        const std::uint64_t row = ids[index / kPleHeads][index % kPleHeads];
        if (row >= kPleRows) { throw std::out_of_range("PLE row ID is outside the table"); }
        return row;
    };
    std::size_t prefetched = 0;
    for (std::size_t index = 0; index < rows; ++index) {
        for (const std::size_t end = std::min(rows, index + kPrefetchRows); prefetched < end;
             ++prefetched) {
            table.prefetch(row_id(prefetched) * kPleHeadWidth, kPleHeadWidth);
        }
        const std::uint64_t destination = static_cast<std::uint64_t>(index) * kPleHeadWidth;
        table.copy(row_id(index) * kPleHeadWidth, output.subspan(destination, kPleHeadWidth));
    }
}

float ple_nvfp4_scale2(const artifact::MappedRange& table) {
    if (table.size() != kPleNvfp4TableBytes) {
        throw std::invalid_argument("PLE NVFP4 table has the wrong byte length");
    }
    std::array<std::byte, 4> word{};
    table.copy(kPleNvfp4Scale2Offset, word);
    std::uint32_t bits = 0;
    for (int byte = 0; byte < 4; ++byte) {
        bits |= static_cast<std::uint32_t>(word[byte]) << (8 * byte);
    }
    return std::bit_cast<float>(bits);
}

void decode_ple_nvfp4_row_bf16(std::span<const std::byte> codes, std::span<const std::byte> scales,
                               std::span<std::byte> output_bf16) {
    if (codes.size() != kPleNvfp4CodeBytesPerRow || scales.size() != kPleNvfp4ScaleBytesPerRow ||
        output_bf16.size() != 2 * kPleHeadWidth) {
        throw std::invalid_argument("PLE NVFP4 row spans have the wrong size");
    }
    auto out = std::span(reinterpret_cast<std::uint16_t*>(output_bf16.data()), kPleHeadWidth);
    for (int group = 0; group < kPleNvfp4ScaleBytesPerRow; ++group) {
        const float scale = decode_e4m3_host(static_cast<unsigned>(scales[group]));
        if (std::isnan(scale)) {
            throw std::range_error("PLE NVFP4 group scale is not a legal E4M3 value");
        }
        for (int half = 0; half < 8; ++half) {
            const unsigned packed = static_cast<unsigned>(codes[group * 8 + half]);
            // The E2M1 x E4M3-scale product has at most six significand bits, so the BF16
            // rounding below is exact.
            out[group * 16 + 2 * half]     = bf16_bits_rn(decode_e2m1_host(packed & 0xF) * scale);
            out[group * 16 + 2 * half + 1] = bf16_bits_rn(decode_e2m1_host(packed >> 4) * scale);
        }
    }
}

void gather_ple_nvfp4(const artifact::MappedRange& table, std::span<const PleIds> ids,
                      std::span<std::byte> output) {
    NINFER_PERF_SCOPE("ninfer.host/1|ple.gather.nvfp4");
    if (table.size() != kPleNvfp4TableBytes) {
        throw std::invalid_argument("PLE NVFP4 table has the wrong byte length");
    }
    const std::uint64_t required = static_cast<std::uint64_t>(ids.size()) * kPleEmbeddingDim;
    if (output.size() != required) {
        throw std::invalid_argument("PLE gathered output has the wrong byte length");
    }
    // The scale_2 word folds into the staged FP8 values, so the PLE consumer's embedding
    // scale is exactly one; the staged rows carry the same values the vendor FP8 table
    // would stage for the same source weights.
    const float scale_2 = ple_nvfp4_scale2(table);
    if (!std::isfinite(scale_2) || scale_2 <= 0.0F) {
        throw std::range_error("PLE NVFP4 scale_2 is not finite and positive");
    }
    const std::size_t rows = ids.size() * kPleHeads;
    const auto row_id      = [&ids](std::size_t index) -> std::uint64_t {
        const std::uint64_t row = ids[index / kPleHeads][index % kPleHeads];
        if (row >= kPleRows) { throw std::out_of_range("PLE row ID is outside the table"); }
        return row;
    };
    // The same page-fault windowing as the FP8 gather, over both planes: an uncached row is
    // one fault in the code plane and one in the scale plane.
    constexpr std::size_t kPrefetchRows = 512;
    std::array<std::byte, kPleNvfp4CodeBytesPerRow> codes{};
    std::array<std::byte, kPleNvfp4ScaleBytesPerRow> scales{};
    std::array<std::byte, 2 * kPleHeadWidth> decoded{};
    std::size_t prefetched = 0;
    for (std::size_t index = 0; index < rows; ++index) {
        for (const std::size_t end = std::min(rows, index + kPrefetchRows); prefetched < end;
             ++prefetched) {
            const std::uint64_t row = row_id(prefetched);
            table.prefetch(row * kPleNvfp4CodeBytesPerRow, kPleNvfp4CodeBytesPerRow);
            table.prefetch(kPleNvfp4ScalePlaneOffset + row * kPleNvfp4ScaleBytesPerRow,
                           kPleNvfp4ScaleBytesPerRow);
        }
        const std::uint64_t row = row_id(index);
        table.copy(row * kPleNvfp4CodeBytesPerRow, codes);
        table.copy(kPleNvfp4ScalePlaneOffset + row * kPleNvfp4ScaleBytesPerRow, scales);
        decode_ple_nvfp4_row_bf16(codes, scales, decoded);
        const auto* words = reinterpret_cast<const std::uint16_t*>(decoded.data());
        auto* staged      = output.data() + static_cast<std::uint64_t>(index) * kPleHeadWidth;
        for (int element = 0; element < kPleHeadWidth; ++element) {
            const float value =
                std::bit_cast<float>(static_cast<std::uint32_t>(words[element]) << 16) * scale_2;
            staged[element] = static_cast<std::byte>(encode_e4m3_rn_satfinite(value));
        }
    }
}

} // namespace ninfer::models::qwen3_8_flash_next
