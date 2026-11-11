// The row-parallel head's gather: rank r's block of the vocabulary is zero-padded into the full
// buffer, and the cross-rank sum turns the two padded blocks into the concatenation of the halves.
//
// The arithmetic is exact by construction — the exchange stores RNE_bf16(float(x) + 0.0f), and a BF16
// value round-trips through float and back unchanged — so the host cases below check the LAYOUT:
// which rows each rank writes, which rows it must leave zero, and how the gather splits into
// exchanges that fit the mailbox slot. They use an independent BF16 round-to-nearest-even conversion
// rather than the exchange's own, because a layout check that borrows the implementation's rounding
// can agree with a layout that is wrong.
//
// The device cases run the real gather between two devices and are skipped when the machine has
// fewer. The refusal cases are host-only: they are the plan's, and the plan is pure.

#include "models/common/sharded_group.h"
#include "models/qwen3_5/execution/sharded.h"

#include "core/buffer.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::CudaDeviceScope;
using ninfer::Device;
using ninfer::DType;
using ninfer::Tensor;
using ninfer::models::ShardedGroupOwner;
using ninfer::models::qwen3_5::execution::LogitsGatherPlan;
using ninfer::models::qwen3_5::execution::plan_logits_gather;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

template <typename Exception, typename Fn>
int expect_throws(Fn&& fn, const std::string& label) {
    try {
        fn();
    } catch (const Exception&) { return 0; }
    std::cerr << label << " did not throw the expected exception\n";
    return 1;
}

// An independent BF16 round-to-nearest-even, so the layout checks do not borrow the exchange's own
// conversion. The values used here are small integers and their exact sums, so this only has to be
// right for finite normals.
std::uint16_t bf16_rne(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(rounded >> 16);
}

float bf16_to_float(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result             = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// The exchange's arithmetic: float(mine) + float(peer), one round-to-nearest-even on store.
std::uint16_t bf16_sum(std::uint16_t mine, std::uint16_t peer) {
    return bf16_rne(bf16_to_float(mine) + bf16_to_float(peer));
}

// The layout the gather and this simulation share: element (row, column) of the full buffer, with
// rank `rank`'s block occupying rows [rank * rank_rows, (rank + 1) * rank_rows).
std::size_t offset_of(const LogitsGatherPlan& plan, int rank, int row_in_block, int column) {
    return static_cast<std::size_t>(rank) * static_cast<std::size_t>(plan.rank_rows) +
           static_cast<std::size_t>(column) * static_cast<std::size_t>(plan.vocabulary) +
           static_cast<std::size_t>(row_in_block);
}

// One rank's zero-fill-then-copy into its own full buffer: the buffer starts zeroed, the rank writes
// only its block, so every other row is an exact BF16 zero for the sum that follows.
std::vector<std::uint16_t> padded_full(const LogitsGatherPlan& plan, int rank,
                                       const std::vector<std::uint16_t>& local) {
    std::vector<std::uint16_t> full(
        static_cast<std::size_t>(plan.vocabulary) * static_cast<std::size_t>(plan.columns), 0);
    for (int column = 0; column < plan.columns; ++column) {
        for (int row = 0; row < plan.rank_rows; ++row) {
            full[offset_of(plan, rank, row, column)] =
                local[static_cast<std::size_t>(column) * plan.rank_rows + row];
        }
    }
    return full;
}

// The concatenation the gather is supposed to produce: rank 0's block, then rank 1's.
std::vector<std::uint16_t> concatenation(const LogitsGatherPlan& plan,
                                         const std::array<std::vector<std::uint16_t>, 2>& local) {
    std::vector<std::uint16_t> expected(
        static_cast<std::size_t>(plan.vocabulary) * static_cast<std::size_t>(plan.columns), 0);
    for (int rank = 0; rank < 2; ++rank) {
        for (int column = 0; column < plan.columns; ++column) {
            for (int row = 0; row < plan.rank_rows; ++row) {
                expected[offset_of(plan, rank, row, column)] =
                    local[static_cast<std::size_t>(rank)][static_cast<std::size_t>(column) *
                                                              plan.rank_rows +
                                                          row];
            }
        }
    }
    return expected;
}

// Two halves whose values differ, and differ from each other's row range, so a transposed block, a
// swapped rank or a missed row all show up as a bit that disagrees.
std::array<std::vector<std::uint16_t>, 2> distinct_halves(const LogitsGatherPlan& plan) {
    std::array<std::vector<std::uint16_t>, 2> local;
    for (int rank = 0; rank < 2; ++rank) {
        local[static_cast<std::size_t>(rank)].resize(
            static_cast<std::size_t>(plan.rank_rows) * static_cast<std::size_t>(plan.columns));
        for (int column = 0; column < plan.columns; ++column) {
            for (int row = 0; row < plan.rank_rows; ++row) {
                // Small integers: exact in BF16, and distinct across rank, row and column.
                const int value = 1 + rank * 100 + column * 10 + row;
                local[static_cast<std::size_t>(rank)][static_cast<std::size_t>(column) *
                                                           plan.rank_rows +
                                                       row] =
                    bf16_rne(static_cast<float>(value));
            }
        }
    }
    return local;
}

// --- host-only: the plan ----------------------------------------------------------------

int exercise_plan() {
    int failures = 0;
    // 8 rows of vocabulary, 4 rows per rank, 3 columns, a slot that carries all 3 columns at once.
    const auto plan = plan_logits_gather(8, 4, 0, 3, 1024);
    failures += check(plan.vocabulary == 8 && plan.rank_rows == 4 && plan.columns == 3,
                      "the plan does not carry the extents it was given");
    failures += check(plan.bytes_per_column == 16, "a column of 8 BF16 rows is not 16 bytes");
    failures += check(plan.chunk_columns == 3, "the plan splits a gather the slot can carry whole");
    failures += check(plan.exchanges == 1, "a gather that fits one slot needs more than one exchange");
    failures += check(plan.chunk_bytes == 48, "one exchange's payload is not 3 columns of 16 bytes");

    // The same gather through a 32-byte slot: two whole columns per exchange, so 3 columns take 2.
    const auto split = plan_logits_gather(8, 4, 0, 3, 32);
    failures += check(split.chunk_columns == 2, "a 32-byte slot carries two 16-byte columns");
    failures += check(split.exchanges == 2, "3 columns through a 2-column chunk is not 2 exchanges");
    failures += check(split.chunk_bytes == 32, "the widest split exchange is not 32 bytes");

    // A slot that is not a whole multiple of a column still carries whole columns only.
    const auto ragged = plan_logits_gather(8, 4, 1, 5, 40);
    failures += check(ragged.chunk_columns == 2, "a 40-byte slot carries more than two 16-byte columns");
    failures += check(ragged.exchanges == 3, "5 columns through a 2-column chunk is not 3 exchanges");
    return failures;
}

// --- host-only: the padded sum is the concatenation --------------------------------------

int exercise_padded_sum(const std::size_t slot_bytes, const char* label) {
    int failures = 0;
    const auto plan = plan_logits_gather(8, 4, 0, 3, slot_bytes);
    const auto local = distinct_halves(plan);
    const auto expected = concatenation(plan, local);

    // Simulate the gather exchange by exchange, exactly as the helper walks it: zero the slice, copy
    // this rank's block of the slice, sum the two ranks' buffers.
    std::array<std::vector<std::uint16_t>, 2> full{padded_full(plan, 0, local[0]),
                                                   padded_full(plan, 1, local[1])};
    for (int chunk = 0; chunk < plan.exchanges; ++chunk) {
        const int first = chunk * plan.chunk_columns;
        const int count = std::min(plan.chunk_columns, plan.columns - first);
        for (int rank = 0; rank < 2; ++rank) {
            for (int column = first; column < first + count; ++column) {
                for (int row = 0; row < plan.rank_rows; ++row) {
                    const auto index = offset_of(plan, rank, row, column);
                    full[static_cast<std::size_t>(rank)][index] = local[static_cast<std::size_t>(rank)]
                        [static_cast<std::size_t>(column) * plan.rank_rows + row];
                }
            }
        }
        // The exchange sums EVERY element of the slice, not only the rows this rank wrote: the
        // peer's rows hold this rank's zeros, and this rank's rows hold the peer's zeros.
        for (int column = first; column < first + count; ++column) {
            for (int row = 0; row < plan.vocabulary; ++row) {
                const auto index = static_cast<std::size_t>(column) *
                                       static_cast<std::size_t>(plan.vocabulary) +
                                   static_cast<std::size_t>(row);
                const auto sum = bf16_sum(full[0][index], full[1][index]);
                full[0][index] = sum;
                full[1][index] = sum;
            }
        }
    }

    // Bit for bit, on both ranks: the sum of the two zero-padded blocks IS the concatenation.
    for (int rank = 0; rank < 2; ++rank) {
        std::size_t wrong = 0;
        for (std::size_t index = 0; index < expected.size(); ++index) {
            if (full[static_cast<std::size_t>(rank)][index] != expected[index]) { ++wrong; }
        }
        failures += check(wrong == 0,
                          std::string(label) + ": rank " + std::to_string(rank) + " holds " +
                              std::to_string(wrong) + " of " + std::to_string(expected.size()) +
                              " elements that are not the concatenation");
    }
    return failures;
}

int exercise_chunked_padded_sum() {
    // The same claim through a slot that forces the gather into two exchanges: chunking must not
    // change a single bit.
    return exercise_padded_sum(32, "chunked padded sum");
}

// --- host-only: the zero-fill leaves the owned block alone -------------------------------

int exercise_owned_block_survives() {
    int failures = 0;
    const auto plan  = plan_logits_gather(8, 4, 0, 3, 1024);
    const auto local = distinct_halves(plan);
    for (int rank = 0; rank < 2; ++rank) {
        // Zero-fill the whole buffer first, then copy this rank's block back. The copy has to be the
        // last writer of the owned rows, or the zero-fill would erase them.
        auto full = padded_full(plan, rank, local[static_cast<std::size_t>(rank)]);
        std::size_t owned_wrong = 0;
        std::size_t peer_nonzero = 0;
        for (int column = 0; column < plan.columns; ++column) {
            for (int row = 0; row < plan.rank_rows; ++row) {
                const auto owned = offset_of(plan, rank, row, column);
                const auto peer  = offset_of(plan, 1 - rank, row, column);
                if (full[owned] !=
                    local[static_cast<std::size_t>(rank)][static_cast<std::size_t>(column) *
                                                              plan.rank_rows +
                                                          row]) {
                    ++owned_wrong;
                }
                if (full[peer] != 0) { ++peer_nonzero; }
            }
        }
        failures += check(owned_wrong == 0, "rank " + std::to_string(rank) +
                                                "'s owned block is not its own projection");
        failures += check(peer_nonzero == 0,
                          "rank " + std::to_string(rank) + " did not leave the peer's block zero");
    }
    return failures;
}

// --- host-only: the refusals ------------------------------------------------------------

int exercise_refusals() {
    int failures = 0;

    // One column does not fit the slot. This is the case chunking cannot fix, and the message has to
    // name both sizes or the caller cannot tell what to resize.
    try {
        (void)plan_logits_gather(8, 4, 0, 3, 8);
        failures += check(false, "a slot smaller than one column was accepted");
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        failures += check(message.find("payload=16") != std::string::npos,
                          "the refusal does not name the column payload: " + message);
        failures += check(message.find("slot=8") != std::string::npos,
                          "the refusal does not name the slot: " + message);
    }

    // The vocabulary is not two rank halves: the concatenation would not cover it.
    failures += expect_throws<std::invalid_argument>(
        [] { (void)plan_logits_gather(8, 3, 0, 1, 1024); },
        "a vocabulary that is not two rank halves");

    // A column that is not whole 16-byte exchange units cannot be exchanged at all.
    failures += expect_throws<std::invalid_argument>(
        [] { (void)plan_logits_gather(6, 3, 0, 1, 1024); },
        "a vocabulary whose column is not a multiple of 16 bytes");

    // A rank outside the pair.
    failures += expect_throws<std::invalid_argument>(
        [] { (void)plan_logits_gather(8, 4, 2, 1, 1024); },
        "a rank outside the pair");
    return failures;
}

// --- two devices: the real gather -------------------------------------------------------

int exercise_device_gather(ninfer::ExecutionContext& context) {
    constexpr std::int32_t kVocabulary = 8;
    constexpr std::int32_t kRankRows   = 4;
    constexpr std::int32_t kColumns    = 3;
    int failures                       = 0;

    // The partial extents are irrelevant to the gather; the logits extents are what it uses. The
    // mailbox slot (8 x 4 BF16 = 64 bytes) carries a 3-column gather whole, so this exercises one
    // exchange and the layout, not the chunk loop.
    ShardedGroupOwner owner(context, 8, 4, kVocabulary, kColumns);
    const auto& group = owner.group();
    failures += check(group.logits[0].data != nullptr && group.logits[1].data != nullptr,
                      "the owner built no logits pair");
    failures += check(group.logits[0].ne[0] == kVocabulary && group.logits[0].ne[1] == kColumns,
                      "the logits pair is not [vocabulary, columns]");
    failures += check(group.logits[0].data != group.logits[1].data,
                      "the two ranks share one logits buffer");

    const auto plan = plan_logits_gather(kVocabulary, kRankRows, 0, kColumns,
                                         group.mailbox->max_bytes());
    const auto local = distinct_halves(plan);
    const auto expected = concatenation(plan, local);

    std::array<ninfer::DeviceBuffer, 2> scratch;
    for (int rank = 0; rank < 2; ++rank) {
        const CudaDeviceScope scope(context.device(rank).device());
        const std::size_t bytes =
            static_cast<std::size_t>(kRankRows) * kColumns * sizeof(std::uint16_t);
        scratch[static_cast<std::size_t>(rank)] =
            ninfer::DeviceBuffer(context.device(rank).device(), bytes);
        failures += check(cudaMemcpy(scratch[static_cast<std::size_t>(rank)].span().data,
                                     local[static_cast<std::size_t>(rank)].data(), bytes,
                                     cudaMemcpyHostToDevice) == cudaSuccess,
                          "the rank's projection could not be filled");
    }

    // Both ranks issue the gather, each on its own stream, then both are synchronized: the exchange
    // kernels meet on the mailbox.
    std::size_t slot = 0;
    for (int rank = 0; rank < 2; ++rank) {
        const CudaDeviceScope scope(context.device(rank).device());
        Tensor local_view{scratch[static_cast<std::size_t>(rank)].span(), DType::BF16,
                          {kRankRows, kColumns}};
        ninfer::models::qwen3_5::execution::all_gather_logits(
            group, rank, slot, local_view, group.logits[static_cast<std::size_t>(rank)],
            context.device(rank).execution_view());
    }
    for (int rank = 0; rank < 2; ++rank) {
        const CudaDeviceScope scope(context.device(rank).device());
        failures += check(cudaDeviceSynchronize() == cudaSuccess,
                          "rank " + std::to_string(rank) + " synchronized");
    }
    failures += check(!group.mailbox->hang_reported(),
                      "a poller gave up: the two ranks did not meet on the mailbox");
    failures += check(slot == static_cast<std::size_t>(plan.exchanges),
                      "the gather did not consume one exchange index per exchange");

    // Both ranks must hold the whole vocabulary, bit for bit.
    for (int rank = 0; rank < 2; ++rank) {
        const CudaDeviceScope scope(context.device(rank).device());
        std::vector<std::uint16_t> host(expected.size(), 0);
        failures += check(cudaMemcpy(host.data(), group.logits[static_cast<std::size_t>(rank)].data,
                                     host.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost) ==
                              cudaSuccess,
                          "rank " + std::to_string(rank) + "'s gathered logits could not be read");
        std::size_t wrong = 0;
        for (std::size_t index = 0; index < expected.size(); ++index) {
            if (host[index] != expected[index]) { ++wrong; }
        }
        failures += check(wrong == 0,
                          "rank " + std::to_string(rank) + " gathered " + std::to_string(wrong) +
                              " of " + std::to_string(expected.size()) +
                              " elements that are not the concatenation of the two halves");
    }
    return failures;
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
           error == cudaErrorStubLibrary;
}

} // namespace

int main() {
    int failures = 0;
    failures += exercise_plan();
    failures += exercise_padded_sum(1024, "padded sum");
    failures += exercise_chunked_padded_sum();
    failures += exercise_owned_block_survives();
    failures += exercise_refusals();

    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cerr << "SKIP: no usable CUDA device; the two-device gather did not run; supply one "
                     "with CUDA_VISIBLE_DEVICES (sm_121a on GB10, sm_120a on RTX 5090)\n";
        return failures == 0 ? 77 : 1;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count < 2) {
        std::cerr << "SKIP: the gather needs two devices. Set CUDA_VISIBLE_DEVICES to two devices, "
                     "for example 0,1\n";
        return failures == 0 ? 77 : 1;
    }

    try {
        ninfer::ExecutionContext two(std::vector<int>{0, 1});
        failures += exercise_device_gather(two);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: threw: " << error.what() << '\n';
        return 1;
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
