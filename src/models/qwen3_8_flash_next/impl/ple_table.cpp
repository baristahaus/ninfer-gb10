#include "models/qwen3_8_flash_next/impl/ple_table.h"

#include "core/performance.h"

#include <algorithm>
#include <array>
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

} // namespace ninfer::models::qwen3_8_flash_next

namespace ninfer::models::qwen3_8_flash_next {
namespace {

// Exact host-side E4M3FN/E2M1 word decoders for the staging decode. Every
// staged value (an E4M3 code, or an E2M1 magnitude times an E4M3 group scale)
// has at most six significant mantissa bits, so BF16 represents it exactly.

float decode_e2m1_host(unsigned code) {
    static constexpr float kMagnitudes[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    const float magnitude = kMagnitudes[code & 7];
    return (code & 8) ? -magnitude : magnitude;
}

float decode_e4m3_host(unsigned word) {
    const float sign = (word & 0x80) ? -1.0F : 1.0F;
    const unsigned exponent = (word >> 3) & 0xF;
    const unsigned fraction = word & 7;
    if (exponent == 0) { return sign * static_cast<float>(fraction) * 0.001953125F; }
    if (exponent == 15 && fraction == 7) { return std::numeric_limits<float>::quiet_NaN(); }
    return sign * std::ldexp(1.0F + static_cast<float>(fraction) / 8.0F,
                             static_cast<int>(exponent) - 7);
}

/** Round a host float to nearest-even BF16 (bit pattern). */
std::uint16_t bf16_bits_rn(float value) {
    const std::uint32_t bits      = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t lsb       = (bits >> 16) & 1;
    const std::uint32_t rounded   = (bits + 0x7FFFU + lsb) & 0xFFFF0000U;
    return static_cast<std::uint16_t>(rounded >> 16);
}

} // namespace

void decode_ple_nvfp4_row_bf16(std::span<const std::byte> codes, std::span<const std::byte> scales,
                               std::span<std::byte> output_bf16) {
    if (codes.size() != kPleNvfp4CodeBytesPerRow || scales.size() != kPleNvfp4ScaleBytesPerRow ||
        output_bf16.size() != 2 * kPleHeadWidth) {
        throw std::invalid_argument("PLE NVFP4 row spans have the wrong size");
    }
    auto out = std::span(reinterpret_cast<std::uint16_t*>(output_bf16.data()), kPleHeadWidth);
    for (int group = 0; group < 10; ++group) {
        const float scale = decode_e4m3_host(static_cast<unsigned>(scales[group]));
        if (std::isnan(scale)) {
            throw std::range_error("PLE NVFP4 group scale is not a legal E4M3 value");
        }
        for (int half = 0; half < 8; ++half) {
            const unsigned packed = static_cast<unsigned>(codes[group * 8 + half]);
            out[group * 16 + 2 * half] =
                bf16_bits_rn(decode_e2m1_host(packed & 0xF) * scale);
            out[group * 16 + 2 * half + 1] =
                bf16_bits_rn(decode_e2m1_host(packed >> 4) * scale);
        }
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

void gather_ple_nvfp4_as_bf16(const artifact::MappedRange& table, std::span<const PleIds> ids,
                              std::span<std::byte> output) {
    NINFER_PERF_SCOPE("ninfer.host/1|ple.gather.nvfp4");
    if (table.size() != kPleNvfp4TableBytes) {
        throw std::invalid_argument("PLE NVFP4 table has the wrong byte length");
    }
    const std::uint64_t required = static_cast<std::uint64_t>(ids.size()) * kPleStageBytesPerToken;
    if (output.size() != required) {
        throw std::invalid_argument("PLE gathered output has the wrong byte length");
    }
    const auto row_id = [&ids](std::size_t index) -> std::uint64_t {
        const std::uint64_t row = ids[index / kPleHeads][index % kPleHeads];
        if (row >= kPleRows) { throw std::out_of_range("PLE row ID is outside the table"); }
        return row;
    };
    std::array<std::byte, kPleNvfp4CodeBytesPerRow> codes{};
    std::array<std::byte, kPleNvfp4ScaleBytesPerRow> scales{};
    std::size_t prefetched = 0;
    constexpr std::size_t kPrefetchRows = 512;
    const std::size_t rows = ids.size() * kPleHeads;
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
        decode_ple_nvfp4_row_bf16(codes, scales,
                                  output.subspan(static_cast<std::uint64_t>(index) *
                                                     2 * kPleHeadWidth,
                                                 2 * kPleHeadWidth));
    }
}
void gather_ple_fp8_as_bf16(const artifact::MappedRange& table, std::span<const PleIds> ids,
                            std::span<std::byte> output) {
    NINFER_PERF_SCOPE("ninfer.host/1|ple.gather");

    constexpr std::uint64_t table_bytes = kPleRows * kPleHeadWidth;
    if (table.size() != table_bytes) {
        throw std::invalid_argument("PLE table has the wrong byte length");
    }
    const std::uint64_t required = static_cast<std::uint64_t>(ids.size()) * kPleStageBytesPerToken;
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
    // code = exact BF16 value of the stored E4M3 code (all E4M3FN values are
    // exactly representable in BF16); built once, indexed by code word.
    static const auto code_lut = [] {
        std::array<std::uint16_t, 256> result{};
        for (unsigned word = 0; word < 256; ++word) {
            result[word] = bf16_bits_rn(decode_e4m3_host(word));
        }
        return result;
    }();
    std::array<std::byte, kPleHeadWidth> row{};
    std::size_t prefetched = 0;
    for (std::size_t index = 0; index < rows; ++index) {
        for (const std::size_t end = std::min(rows, index + kPrefetchRows); prefetched < end;
             ++prefetched) {
            table.prefetch(row_id(prefetched) * kPleHeadWidth, kPleHeadWidth);
        }
        table.copy(row_id(index) * kPleHeadWidth, row);
        const std::uint64_t destination =
            static_cast<std::uint64_t>(index) * 2 * kPleHeadWidth;
        auto out = std::span(reinterpret_cast<std::uint16_t*>(output.data() + destination),
                             kPleHeadWidth);
        for (int element = 0; element < kPleHeadWidth; ++element) {
            out[element] = code_lut[static_cast<unsigned>(row[element])];
        }
    }
}

} // namespace ninfer::models::qwen3_8_flash_next
