// The SP-4 tier-exchange oracle: swap_expert_tier must move each expert's planar
// per-plane block exactly, preserve every untouched slot byte-for-byte, and rewrite the
// residency to the exchanged ranks. Pure data movement, so the oracle is an exact
// byte-level comparison against a host-computed permutation. Regression for the two
// defects found by review: the interleaved-block copy over planar banks, and the
// unsynchronized copy pair that destroyed the demoted expert's weights.
#include "ninfer/ops/flash_next_moe.h"
#include "core/buffer.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

namespace {

using namespace ninfer;

constexpr int kExperts = 8;
constexpr int kRows = 16;
constexpr int kColumns = 64;
constexpr int kHotSlots = 4;
constexpr int kCacheSlots = 1;
constexpr int kWarmSlots = kExperts - kHotSlots;
// NVFP4 planar geometry per expert.
constexpr std::size_t kCodeBlock = static_cast<std::size_t>(kRows) * kColumns / 2;
constexpr std::size_t kScaleBlock = static_cast<std::size_t>(kRows) * kColumns / 16;
constexpr std::size_t kHotSlotsTotal = kHotSlots + kCacheSlots;
constexpr std::size_t kHotCodes = kCodeBlock * kHotSlotsTotal;
constexpr std::size_t kHotScales = kScaleBlock * kHotSlotsTotal;
constexpr std::size_t kHotDivisors = kHotSlotsTotal * 4;
constexpr std::size_t kWarmCodes = kCodeBlock * kWarmSlots;
constexpr std::size_t kWarmScales = kScaleBlock * kWarmSlots;
constexpr std::size_t kWarmDivisors = static_cast<std::size_t>(kWarmSlots) * 4;

struct TieredBankFixture {
    DeviceBuffer hot;
    PinnedHostBuffer warm;
    DeviceBuffer staging;
    DeviceBuffer residency;
    std::vector<std::uint8_t> hot_expect;
    std::vector<std::uint8_t> warm_expect;

    // The warm bank is mutable scratch while the fixture fills it; the op reads it const.
    [[nodiscard]] void* warm_data() noexcept { return const_cast<void*>(warm.data()); }

    [[nodiscard]] void* staging_data() noexcept { return staging.data(); }

    TieredBankFixture(Device device)
        : hot(device, kHotCodes + kHotScales + kHotDivisors),
          warm(kWarmCodes + kWarmScales + kWarmDivisors),
          staging(device, kCodeBlock),
          residency(device, static_cast<std::size_t>(kExperts) * 4) {
        // Hot slots 0..3 hold experts 0,2,4,6; warm slots 0..3 hold experts 1,3,5,7.
        // Every (plane, slot) carries a distinct byte tag so any misplacement mismatches.
        std::vector<std::uint8_t> hot_bytes(kHotCodes + kHotScales + kHotDivisors, 0);
        for (std::size_t slot = 0; slot < kHotSlotsTotal; ++slot) {
            std::memset(hot_bytes.data() + slot * kCodeBlock, static_cast<int>(11 + slot * 5),
                        kCodeBlock);
            std::memset(hot_bytes.data() + kHotCodes + slot * kScaleBlock,
                        static_cast<int>(31 + slot * 5), kScaleBlock);
            const float divisor = 1.0F + static_cast<float>(slot) * 0.25F;
            std::memcpy(hot_bytes.data() + kHotCodes + kHotScales + slot * 4, &divisor, 4);
        }
        hot.copy_from_host(hot_bytes.data(), hot_bytes.size(), 0);
        hot_expect = hot_bytes;

        std::vector<std::uint8_t> warm_bytes(kWarmCodes + kWarmScales + kWarmDivisors, 0);
        for (int slot = 0; slot < kWarmSlots; ++slot) {
            std::memset(warm_bytes.data() + static_cast<std::size_t>(slot) * kCodeBlock,
                        static_cast<int>(51 + slot * 5), kCodeBlock);
            std::memset(warm_bytes.data() + kWarmCodes + static_cast<std::size_t>(slot) * kScaleBlock,
                        static_cast<int>(71 + slot * 5), kScaleBlock);
            const float divisor = 2.0F + static_cast<float>(slot) * 0.5F;
            std::memcpy(warm_bytes.data() + kWarmCodes + kWarmScales +
                            static_cast<std::size_t>(slot) * 4,
                        &divisor, 4);
        }
        std::memcpy(warm_data(), warm_bytes.data(), warm_bytes.size());
        warm_expect = warm_bytes;

        // residency[expert] = rank | warm-bit: experts 0,2,4,6 hot (slots 0..3),
        // experts 1,3,5,7 warm (slots 0..3).
        std::vector<std::uint32_t> table(static_cast<std::size_t>(kExperts), 0);
        for (int slot = 0; slot < kHotSlots; ++slot) {
            table[static_cast<std::size_t>(slot * 2)] = static_cast<std::uint32_t>(slot);
        }
        for (int slot = 0; slot < kWarmSlots; ++slot) {
            table[static_cast<std::size_t>(slot * 2 + 1)] =
                static_cast<std::uint32_t>(slot) | 0x80000000u;
        }
        residency.copy_from_host(table.data(), table.size() * sizeof(std::uint32_t), 0);
    }
};

ops::FlashNextExpertBank make_bank(TieredBankFixture& fixture) {
    ops::FlashNextExpertBank bank{};
    bank.codes = fixture.hot.data();
    bank.scales = static_cast<const std::byte*>(fixture.hot.data()) + kHotCodes;
    bank.weight_scale_divisors = reinterpret_cast<const float*>(
        static_cast<const std::byte*>(fixture.hot.data()) + kHotCodes + kHotScales);
    bank.qtype = QType::NVFP4;
    bank.experts = kExperts;
    bank.rows = kRows;
    bank.columns = kColumns;
    bank.warm_codes = fixture.warm.data();
    bank.warm_scales = static_cast<const std::byte*>(fixture.warm.data()) + kWarmCodes;
    bank.warm_weight_scale_divisors = reinterpret_cast<const float*>(
        static_cast<const std::byte*>(fixture.warm.data()) + kWarmCodes + kWarmScales);
    bank.tier_swap_staging = fixture.staging_data();
    bank.expert_residency = static_cast<const std::uint32_t*>(fixture.residency.data());
    return bank;
}

int compare(const char* what, const std::vector<std::uint8_t>& got,
            const std::vector<std::uint8_t>& want) {
    if (got == want) { return 0; }
    for (std::size_t i = 0; i < want.size() && i < got.size(); ++i) {
        if (got[i] != want[i]) {
            std::cerr << what << ": first mismatch at byte " << i << ": got "
                      << static_cast<int>(got[i]) << " want " << static_cast<int>(want[i])
                      << '\n';
            return 1;
        }
    }
    std::cerr << what << ": size mismatch " << got.size() << " vs " << want.size() << '\n';
    return 1;
}

int run() {
    const Device device = ninfer::test::cuda_environment().context.device();
    TieredBankFixture fixture(device);
    const auto bank = make_bank(fixture);
    const auto execution = ninfer::test::cuda_environment().context.execution_view();

    // Promote expert 3 (warm slot 1) over expert 2 (hot slot 1).
    ops::swap_expert_tier(bank, 3, 2, execution);

    // The op enqueues its copies on the execution stream and returns. The readbacks below do not
    // order against that stream — one is a default-stream blocking copy and the other is a bare
    // memcpy of pinned host memory — so without this wait the comparison can run while the copies
    // are still in flight. That made this test fail roughly one run in six.
    if (cudaStreamSynchronize(execution.stream()) != cudaSuccess) {
        std::cerr << "tier swap oracle: execution stream did not drain\n";
        return 1;
    }

    auto hot_got = fixture.hot_expect;
    fixture.hot.copy_to_host(hot_got.data(), hot_got.size(), 0);
    std::vector<std::uint8_t> warm_got(fixture.warm_expect.size());
    std::memcpy(warm_got.data(), fixture.warm.data(), warm_got.size());

    // The expected exchange: hot slot 1 <- warm slot 1 (expert 3's planes and divisor),
    // warm slot 1 <- hot slot 1 (expert 2's), every other slot untouched.
    auto expected_hot = fixture.hot_expect;
    std::memcpy(expected_hot.data() + kCodeBlock,
                fixture.warm_expect.data() + kCodeBlock, kCodeBlock);
    std::memcpy(expected_hot.data() + kHotCodes + kScaleBlock,
                fixture.warm_expect.data() + kWarmCodes + kScaleBlock, kScaleBlock);
    std::memcpy(expected_hot.data() + kHotCodes + kHotScales + 4,
                fixture.warm_expect.data() + kWarmCodes + kWarmScales + 4, 4);
    auto expected_warm = fixture.warm_expect;
    std::memcpy(expected_warm.data() + kCodeBlock,
                fixture.hot_expect.data() + kCodeBlock, kCodeBlock);
    std::memcpy(expected_warm.data() + kWarmCodes + kScaleBlock,
                fixture.hot_expect.data() + kHotCodes + kScaleBlock, kScaleBlock);
    std::memcpy(expected_warm.data() + kWarmCodes + kWarmScales + 4,
                fixture.hot_expect.data() + kHotCodes + kHotScales + 4, 4);
    int failures = 0;
    failures += compare("hot mini-bank", hot_got, expected_hot);
    failures += compare("warm mini-bank", warm_got, expected_warm);

    // Residency: expert 3 now hot at slot 1, expert 2 warm at slot 1, others unchanged.
    std::vector<std::uint32_t> table(static_cast<std::size_t>(kExperts), 0);
    fixture.residency.copy_to_host(table.data(), table.size() * sizeof(std::uint32_t), 0);
    const auto entry = [](std::uint32_t rank, bool is_warm) {
        return rank | (is_warm ? 0x80000000u : 0u);
    };
    if (table[3] != entry(1, false) || table[2] != entry(1, true)) {
        std::cerr << "residency: expert3=" << table[3] << " expert2=" << table[2] << '\n';
        failures += 1;
    }
    for (int e = 0; e < kExperts; ++e) {
        if (e == 2 || e == 3) { continue; }
        const auto want = entry(static_cast<std::uint32_t>(e / 2), (e % 2) != 0);
        if (table[static_cast<std::size_t>(e)] != want) {
            std::cerr << "residency: expert " << e << " became " << table[e] << " want "
                      << want << '\n';
            failures += 1;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::cerr << "SKIP: no usable CUDA device; the test did not run; supply one with CUDA_VISIBLE_DEVICES (sm_121a on GB10, sm_120a on RTX 5090)\n";
        return 77;
    }
    try {
        const int failures = run();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " tier swap oracle\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "tier swap oracle: " << error.what() << '\n';
        return 1;
    }
}
