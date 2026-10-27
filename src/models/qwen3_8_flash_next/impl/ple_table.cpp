#include "models/qwen3_8_flash_next/impl/ple_table.h"

#include "core/performance.h"

#include <algorithm>
#include <bit>
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

void gather_ple_fp8(const artifact::MappedRange& table, std::span<const PleIds> ids,
                    std::span<std::byte> output) {
    NINFER_PERF_SCOPE("ninfer.host/1|ple.gather");

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

} // namespace ninfer::models::qwen3_8_flash_next
