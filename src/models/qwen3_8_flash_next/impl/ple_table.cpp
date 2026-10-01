#include "models/qwen3_8_flash_next/impl/ple_table.h"

#include "core/performance.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <future>
#include <stdexcept>
#include <vector>

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

void gather_ple(const artifact::MappedRange& table, DType dtype, std::span<const PleIds> ids,
                std::span<std::byte> output, HostWorkerPool* workers) {
    NINFER_PERF_SCOPE("ninfer.host/1|ple.gather");

    if (dtype != DType::FP8_E4M3FN && dtype != DType::BF16) {
        throw std::invalid_argument("PLE table must be FP8 or BF16");
    }
    const std::uint64_t element_bytes = dtype == DType::BF16 ? 2U : 1U;
    const std::uint64_t table_bytes   = kPleRows * kPleHeadWidth * element_bytes;
    if (table.size() != table_bytes) {
        throw std::invalid_argument("PLE table has the wrong byte length");
    }
    const std::uint64_t required =
        static_cast<std::uint64_t>(ids.size()) * kPleEmbeddingDim * element_bytes;
    if (output.size() != required) {
        throw std::invalid_argument("PLE gathered output has the wrong byte length");
    }
    const auto gather = [&](std::size_t begin, std::size_t end) {
        for (std::size_t token = begin; token < end; ++token) {
            for (std::size_t head = 0; head < kPleHeads; ++head) {
                const std::uint64_t row = ids[token][head];
                if (row >= kPleRows) { throw std::out_of_range("PLE row ID is outside the table"); }
                const std::uint64_t source = row * kPleHeadWidth * element_bytes;
                const std::uint64_t destination =
                    (static_cast<std::uint64_t>(token) * kPleEmbeddingDim + head * kPleHeadWidth) *
                    element_bytes;
                table.copy(source, output.subspan(destination, kPleHeadWidth * element_bytes));
            }
        }
    };
    if (workers == nullptr || ids.size() < 128) {
        gather(0, ids.size());
        return;
    }
    const std::size_t tasks = std::min<std::size_t>(workers->snapshot().threads, ids.size() / 64);
    std::vector<std::future<void>> pending;
    pending.reserve(tasks);
    std::exception_ptr error;
    try {
        for (std::size_t task = 0; task < tasks; ++task) {
            pending.push_back(workers->submit(
                [&, task] { gather(ids.size() * task / tasks, ids.size() * (task + 1) / tasks); }));
        }
    } catch (...) { error = std::current_exception(); }
    // Drain every task before propagating an error: workers borrow the caller's buffers.
    for (auto& task : pending) {
        try {
            task.get();
        } catch (...) {
            if (!error) { error = std::current_exception(); }
        }
    }
    if (error) { std::rethrow_exception(error); }
}

} // namespace ninfer::models::qwen3_8_flash_next
