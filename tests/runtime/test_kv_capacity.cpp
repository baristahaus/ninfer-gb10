#include "runtime/engine/kv_capacity.h"
#include "runtime/engine/startup_memory.h"

#include <iostream>
#include <array>
#include <stdexcept>
#include <span>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// Field-for-field equality of two resolutions; a symmetric resolution must reproduce the
// single-device one exactly.
bool same_resolution(const ninfer::runtime::KvCapacityResolution& a,
                     const ninfer::runtime::KvCapacityResolution& b) {
    return a.mode == b.mode && a.main_page_groups == b.main_page_groups &&
           a.maximum_main_page_groups == b.maximum_main_page_groups &&
           a.resolved_tokens == b.resolved_tokens &&
           a.minimum_runtime_reservation_bytes == b.minimum_runtime_reservation_bytes &&
           a.bytes_per_additional_main_page_group == b.bytes_per_additional_main_page_group &&
           a.runtime_reservation_bytes == b.runtime_reservation_bytes &&
           a.available_after_weights_bytes == b.available_after_weights_bytes &&
           a.available_after_startup_bytes == b.available_after_startup_bytes &&
           a.automatic_headroom_bytes == b.automatic_headroom_bytes &&
           a.planned_slack_bytes == b.planned_slack_bytes;
}

} // namespace

int main() {
    int failures = 0;
    const ninfer::runtime::SequenceCapacityCurve curve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 6,
        .minimum_device_reservation_bytes        = 1000,
        .bytes_per_additional_main_page_group = 128,
    };

    const auto automatic =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 1360);
    failures +=
        check(automatic.main_page_groups == 4 && automatic.resolved_tokens == 256 &&
                  automatic.runtime_reservation_bytes == 1256 &&
                  automatic.automatic_headroom_bytes == 50 && automatic.planned_slack_bytes == 104,
              "automatic KV capacity did not select the largest fitting page count");

    const auto capped =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 10000);
    failures += check(capped.main_page_groups == 6 && capped.resolved_tokens == 384,
                      "automatic KV capacity exceeded or missed the target maximum");

    const std::array offers{ninfer::runtime::RankCapacityOffer{curve, 2000},
                            ninfer::runtime::RankCapacityOffer{{64, 2, 6, 1500, 256}, 2100}};
    const auto common =
        ninfer::runtime::resolve_group_kv_capacity(ninfer::KvCapacityPolicy::automatic(0), offers);
    failures += check(common[0].main_page_groups == 4 && common[1].main_page_groups == 4 &&
                          common[0].runtime_reservation_bytes == 1256 &&
                          common[1].runtime_reservation_bytes == 2012,
                      "group capacity did not preserve common pages and local byte costs");
    bool group_rejected = false;
    try {
        (void)ninfer::runtime::resolve_group_kv_capacity(
            ninfer::KvCapacityPolicy::explicit_capacity(320), offers);
    } catch (const std::invalid_argument&) { group_rejected = true; }
    failures += check(group_rejected, "group capacity silently reduced an explicit request");

    // --- symmetric capacity across an execution group -------------------------------------
    // A wide curve keeps the headroom arithmetic visible: 50 bytes of headroom, 100 bytes per
    // extra page, 1000 bytes for the minimum. A single subtraction at 1380 bytes gives capacity
    // 1330, three extra pages (5 total, 1300 bytes); a second subtraction would give 4 pages.
    const ninfer::runtime::SequenceCapacityCurve wide{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 100,
        .minimum_device_reservation_bytes        = 1000,
        .bytes_per_additional_main_page_group = 100,
    };
    const std::array<std::size_t, 1> single_budget{1380};
    const auto single = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(50), wide, single_budget.front());
    failures += check(single.main_page_groups == 5 && single.runtime_reservation_bytes == 1300 &&
                          single.planned_slack_bytes == 80,
                      "single-device automatic capacity baseline changed");

    // Width 1 is today's behaviour exactly.
    const auto symmetric_one = ninfer::runtime::resolve_symmetric_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(50), wide, single_budget);
    failures += check(same_resolution(symmetric_one, single),
                      "symmetric capacity of one rank differs from the single-device resolution");

    // Two equal ranks: the headroom is subtracted once, not once per rank (5 pages, not 4).
    const std::array<std::size_t, 2> equal_budgets{1380, 1380};
    const auto symmetric_equal = ninfer::runtime::resolve_symmetric_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(50), wide, equal_budgets);
    failures += check(symmetric_equal.main_page_groups == 5 &&
                          symmetric_equal.runtime_reservation_bytes == 1300 &&
                          symmetric_equal.planned_slack_bytes == 80,
                      "symmetric capacity subtracted the headroom twice or selected another count");
    failures += check(same_resolution(symmetric_equal, single),
                      "two equal ranks resolved differently from one rank");

    // Two unequal ranks: the smaller figure is the bottleneck and the resolution reports it.
    const std::array<std::size_t, 2> unequal_budgets{1380, 1500};
    const auto symmetric_unequal = ninfer::runtime::resolve_symmetric_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(50), wide, unequal_budgets);
    failures += check(same_resolution(symmetric_unequal, single) &&
                          symmetric_unequal.available_after_weights_bytes == 1380,
                      "symmetric capacity did not resolve against the smaller rank");

    // The group form credits a rank with the reservation part it does not allocate: 380 free bytes
    // plus a 1000-byte credit is the same budget as 1380 and must not bottleneck the group.
    const std::array group_credit{ninfer::runtime::RankCapacityOffer{wide, 380, 1000},
                                  ninfer::runtime::RankCapacityOffer{wide, 1380, 0}};
    const auto credited = ninfer::runtime::resolve_group_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(50), group_credit);
    failures += check(credited[0].main_page_groups == 5 && credited[1].main_page_groups == 5 &&
                          credited[0].runtime_reservation_bytes == 1300 &&
                          credited[1].runtime_reservation_bytes == 1300 &&
                          credited[0].available_after_weights_bytes == 1380 &&
                          credited[0].planned_slack_bytes == 80,
                      "group capacity did not credit the unallocated reservation");
    bool uncredited_rejected = false;
    try {
        (void)ninfer::runtime::resolve_symmetric_kv_capacity(
            ninfer::KvCapacityPolicy::automatic(50), wide,
            std::array<std::size_t, 1>{380});
    } catch (const std::invalid_argument&) { uncredited_rejected = true; }
    failures += check(uncredited_rejected,
                      "an uncredited 380-byte rank was accepted as the group bottleneck");

    // An explicit capacity that only fits with the credit must be accepted, and rejected without
    // it: 5 pages cost 1300 bytes.
    const auto credited_explicit = ninfer::runtime::resolve_group_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(320), group_credit);
    failures += check(credited_explicit[0].main_page_groups == 5 &&
                          credited_explicit[1].main_page_groups == 5,
                      "explicit group capacity ignored the unallocated-reservation credit");
    const std::array group_uncredited{ninfer::runtime::RankCapacityOffer{wide, 380, 0},
                                      ninfer::runtime::RankCapacityOffer{wide, 1380, 0}};
    bool explicit_uncredited_rejected = false;
    try {
        (void)ninfer::runtime::resolve_group_kv_capacity(
            ninfer::KvCapacityPolicy::explicit_capacity(320), group_uncredited);
    } catch (const std::invalid_argument&) { explicit_uncredited_rejected = true; }
    failures += check(explicit_uncredited_rejected,
                      "explicit group capacity accepted a rank that cannot hold the pages");

    // The symmetric form rejects an empty group rather than returning a default.
    bool empty_group_rejected = false;
    try {
        (void)ninfer::runtime::resolve_symmetric_kv_capacity(
            ninfer::KvCapacityPolicy::automatic(50), wide, std::span<const std::size_t>{});
    } catch (const std::invalid_argument&) { empty_group_rejected = true; }
    failures += check(empty_group_rejected, "symmetric capacity accepted an empty group");

    const auto explicit_capacity = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), curve, 1200);
    failures += check(explicit_capacity.main_page_groups == 3 &&
                          explicit_capacity.resolved_tokens == 192 &&
                          explicit_capacity.runtime_reservation_bytes == 1128,
                      "explicit KV capacity did not use page-aligned token semantics");

    bool insufficient_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve,
                                                   1049);
    } catch (const std::invalid_argument&) { insufficient_rejected = true; }
    failures += check(insufficient_rejected,
                      "automatic KV capacity accepted less than the minimum reservation");

    // The probe left 200 bytes in the driver; the new free snapshot already includes it.
    const ninfer::runtime::StartupMemoryBudget budget({10000, 12000}, 8000, 500, 0);
    failures += check(budget.available({7800, 12000}) == 5300,
                      "startup retained allocation was not charged exactly once");
    failures += check(budget.available({9800, 12000}) == 7300,
                      "destroyed candidate backing was not returned to the budget");
    failures += check(budget.retained({7800, 12000}, 2000) == 200,
                      "owned weights were included in opaque retained storage");
    failures += check(ninfer::device_memory_growth({8000, 12000}, {8100, 12000}) == -100,
                      "negative memory growth wrapped to a large unsigned estimate");
    failures += check(budget.fits({2100, 12000}) && !budget.fits({1900, 12000}),
                      "actual startup footprint did not obey the Device budget");
    const ninfer::runtime::StartupMemoryBudget physical({3000, 12000}, 8000, 500, 0);
    failures += check(physical.available({2000, 12000}) == 1500,
                      "physical availability did not bound the configured budget");

    const ninfer::runtime::StartupMemoryBudget shared({10000, 12000}, 8000, 500, 1000);
    failures += check(shared.available({7800, 12000}) == 4300 && shared.headroom() == 1500,
                      "shared Host payload capacity was not reserved");
    failures += check(shared.fits({3000, 12000}) && !shared.fits({2999, 12000}),
                      "startup consumed the shared Host payload reservation");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
