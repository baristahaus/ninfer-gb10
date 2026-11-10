#include "runtime/engine/kv_capacity.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::runtime {
namespace {

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

void validate_curve(const SequenceCapacityCurve& curve) {
    if (curve.main_page_tokens == 0 || curve.minimum_main_page_groups == 0 ||
        curve.minimum_main_page_groups > curve.maximum_main_page_groups ||
        curve.minimum_device_reservation_bytes == 0) {
        throw std::invalid_argument("sequence capacity curve is invalid");
    }
    if (curve.minimum_main_page_groups < curve.maximum_main_page_groups &&
        curve.bytes_per_additional_main_page_group == 0) {
        throw std::invalid_argument("expandable sequence capacity curve has zero byte stride");
    }
}

std::uint32_t explicit_page_groups(const KvCapacityPolicy& policy,
                                   const SequenceCapacityCurve& curve) {
    if (policy.explicit_tokens == 0) {
        throw std::invalid_argument("explicit KV capacity must be positive");
    }
    const std::uint64_t pages = 1ULL + (static_cast<std::uint64_t>(policy.explicit_tokens) - 1ULL) /
                                           curve.main_page_tokens;
    if (pages > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("explicit KV page count exceeds uint32");
    }
    return static_cast<std::uint32_t>(pages);
}

// Page count the curve affords at `capacity_budget`. The caller has already checked that the
// curve's minimum reservation fits the budget. This is the one place the automatic growth formula
// lives; every entry point prices through it.
std::uint32_t affordable_page_groups(const SequenceCapacityCurve& curve,
                                     std::size_t capacity_budget) {
    if (curve.minimum_main_page_groups == curve.maximum_main_page_groups) {
        return curve.minimum_main_page_groups;
    }
    const std::size_t additional = (capacity_budget - curve.minimum_device_reservation_bytes) /
                                   curve.bytes_per_additional_main_page_group;
    const std::uint64_t candidate =
        static_cast<std::uint64_t>(curve.minimum_main_page_groups) + additional;
    return static_cast<std::uint32_t>(
        std::min<std::uint64_t>(candidate, curve.maximum_main_page_groups));
}

} // namespace

std::vector<KvCapacityResolution>
resolve_group_kv_capacity(const KvCapacityPolicy& policy,
                          std::span<const RankCapacityOffer> offers) {
    if (offers.empty()) throw std::invalid_argument("capacity negotiation requires owners");
    const auto tokens   = offers.front().curve.main_page_tokens;
    for (const auto& offer : offers) {
        validate_curve(offer.curve);
        if (offer.curve.main_page_tokens != tokens)
            throw std::invalid_argument("capacity owners disagree on logical page extent");
    }

    // A rank's budget includes the reservation part it does not allocate: those bytes are free for
    // KV capacity on that rank, so ignoring them would under-size the group.
    std::vector<std::size_t> budgets;
    budgets.reserve(offers.size());
    for (const auto& offer : offers) {
        budgets.push_back(checked_add(offer.available_bytes,
                                      offer.unallocated_reservation_bytes,
                                      "rank capacity budget overflows size_t"));
    }

    std::uint32_t pages = 0;
    if (policy.mode == KvCapacityMode::Explicit) {
        pages = explicit_page_groups(policy, offers.front().curve);
    } else if (policy.mode == KvCapacityMode::Automatic) {
        // The headroom is a per-device reserve, so every rank leaves it free once. The group page
        // count is then the smallest count any rank can hold; subtracting the headroom from the
        // bottleneck a second time would under-size every rank.
        pages = offers.front().curve.maximum_main_page_groups;
        for (std::size_t index = 0; index < offers.size(); ++index) {
            const auto& curve = offers[index].curve;
            const std::size_t budget = budgets[index];
            if (budget < policy.automatic_headroom_bytes) {
                throw std::invalid_argument(
                    "automatic KV headroom requires " +
                    std::to_string(policy.automatic_headroom_bytes) + " bytes, but rank " +
                    std::to_string(index) + " has only " + std::to_string(budget) +
                    " bytes available after weights and its unallocated-reservation credit");
            }
            const std::size_t capacity_budget = budget - policy.automatic_headroom_bytes;
            if (capacity_budget < curve.minimum_device_reservation_bytes) {
                throw std::invalid_argument(
                    "minimum Engine runtime reservation requires " +
                    std::to_string(curve.minimum_device_reservation_bytes) +
                    " bytes in addition to " +
                    std::to_string(policy.automatic_headroom_bytes) +
                    " bytes of automatic headroom, but rank " + std::to_string(index) + " has " +
                    std::to_string(budget) + " bytes available for runtime capacity");
            }
            pages = std::min(pages, affordable_page_groups(curve, capacity_budget));
        }
    } else {
        throw std::invalid_argument("unknown KV capacity policy");
    }

    std::vector<KvCapacityResolution> result;
    result.reserve(offers.size());
    for (std::size_t index = 0; index < offers.size(); ++index) {
        const auto& offer = offers[index];
        const auto resolved = offer.curve.resolved_tokens(pages);
        auto local = resolve_kv_capacity(KvCapacityPolicy::explicit_capacity(resolved), offer.curve,
                                         budgets[index]);
        local.mode = policy.mode;
        local.automatic_headroom_bytes = policy.automatic_headroom_bytes;
        result.push_back(local);
    }
    return result;
}

KvCapacityResolution
resolve_symmetric_kv_capacity(const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve,
                              std::span<const std::size_t> available_runtime_bytes_per_rank) {
    if (available_runtime_bytes_per_rank.empty()) {
        throw std::invalid_argument("KV capacity resolution requires at least one rank budget");
    }
    // One curve, one figure per rank: the bottleneck is the smallest figure, and resolve_kv_capacity
    // subtracts the automatic headroom once from it. A group of one rank is the single-device path.
    return resolve_kv_capacity(policy, curve,
                               *std::min_element(available_runtime_bytes_per_rank.begin(),
                                                 available_runtime_bytes_per_rank.end()));
}


std::size_t SequenceCapacityCurve::reservation_bytes(std::uint32_t main_page_groups) const {
    validate_curve(*this);
    if (main_page_groups < minimum_main_page_groups ||
        main_page_groups > maximum_main_page_groups) {
        throw std::invalid_argument("Main KV page count " + std::to_string(main_page_groups) +
                                    " is outside the target capacity curve [" +
                                    std::to_string(minimum_main_page_groups) + ", " +
                                    std::to_string(maximum_main_page_groups) + "]");
    }
    const std::size_t additional =
        static_cast<std::size_t>(main_page_groups - minimum_main_page_groups);
    return checked_add(minimum_device_reservation_bytes,
                       checked_mul(additional, bytes_per_additional_main_page_group,
                                   "sequence capacity increment overflows size_t"),
                       "sequence capacity reservation overflows size_t");
}

std::uint32_t SequenceCapacityCurve::resolved_tokens(std::uint32_t main_page_groups) const {
    validate_curve(*this);
    if (main_page_groups < minimum_main_page_groups ||
        main_page_groups > maximum_main_page_groups) {
        throw std::invalid_argument("Main KV page count " + std::to_string(main_page_groups) +
                                    " is outside the target capacity curve [" +
                                    std::to_string(minimum_main_page_groups) + ", " +
                                    std::to_string(maximum_main_page_groups) + "]");
    }
    const std::uint64_t tokens =
        static_cast<std::uint64_t>(main_page_groups) * main_page_tokens;
    if (tokens > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("resolved KV token capacity exceeds uint32");
    }
    return static_cast<std::uint32_t>(tokens);
}

KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                         const SequenceCapacityCurve& curve,
                                         std::size_t available_runtime_bytes) {
    validate_curve(curve);

    std::uint32_t pages         = curve.minimum_main_page_groups;
    std::size_t capacity_budget = available_runtime_bytes;
    switch (policy.mode) {
    case KvCapacityMode::Explicit:
        if (policy.automatic_headroom_bytes != 0) {
            throw std::invalid_argument("explicit KV capacity must not carry automatic headroom");
        }
        pages = explicit_page_groups(policy, curve);
        break;
    case KvCapacityMode::Automatic:
        if (available_runtime_bytes < policy.automatic_headroom_bytes) {
            throw std::invalid_argument(
                "automatic KV headroom requires " +
                std::to_string(policy.automatic_headroom_bytes) + " bytes, but only " +
                std::to_string(available_runtime_bytes) + " bytes are available after weights");
        }
        capacity_budget -= policy.automatic_headroom_bytes;
        if (capacity_budget < curve.minimum_device_reservation_bytes) {
            throw std::invalid_argument(
                "minimum Engine runtime reservation requires " +
                std::to_string(curve.minimum_device_reservation_bytes) + " bytes in addition to " +
                std::to_string(policy.automatic_headroom_bytes) +
                " bytes of automatic headroom, but only " +
                std::to_string(available_runtime_bytes) + " bytes are available after weights");
        }
        pages = affordable_page_groups(curve, capacity_budget);
        break;
    default:
        throw std::invalid_argument("unknown KV capacity policy");
    }

    const std::size_t reservation = curve.reservation_bytes(pages);
    if (reservation > capacity_budget) {
        throw std::invalid_argument("requested Engine runtime reservation requires " +
                                    std::to_string(reservation) + " bytes, but only " +
                                    std::to_string(capacity_budget) +
                                    " bytes are available for runtime capacity");
    }

    return KvCapacityResolution{
        .mode                                    = policy.mode,
        .main_page_groups                     = pages,
        .maximum_main_page_groups             = curve.maximum_main_page_groups,
        .resolved_tokens                         = curve.resolved_tokens(pages),
        .minimum_runtime_reservation_bytes       = curve.minimum_device_reservation_bytes,
        .bytes_per_additional_main_page_group = curve.bytes_per_additional_main_page_group,
        .runtime_reservation_bytes               = reservation,
        .available_after_weights_bytes           = available_runtime_bytes,
        .automatic_headroom_bytes                = policy.automatic_headroom_bytes,
        .planned_slack_bytes                     = available_runtime_bytes - reservation,
    };
}

} // namespace ninfer::runtime
