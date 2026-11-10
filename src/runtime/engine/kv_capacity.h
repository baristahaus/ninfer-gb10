#pragma once

#include "runtime/contract/resources.h"

#include <cstddef>
#include <span>
#include <vector>

namespace ninfer::runtime {

[[nodiscard]] KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                                       const SequenceCapacityCurve& curve,
                                                       std::size_t available_runtime_bytes);

// One rank's offer to a group-wide capacity decision.
struct RankCapacityOffer {
    SequenceCapacityCurve curve;
    std::size_t available_bytes;
    // The part of the group's runtime reservation this rank does not allocate (for example the
    // Vision encode workspace on the rank without the tower). It is credited to this rank's budget
    // before the group decision, because the rank can spend those bytes on KV capacity. The
    // distributed preparation report leaves it zero until a rank reports one.
    std::size_t unallocated_reservation_bytes = 0;
};

// One logical page count, with each owner's physical byte cost evaluated separately. A rank's
// budget is its available bytes plus the reservation it does not allocate; each rank pays the
// automatic headroom once out of its own budget (the headroom is a per-device reserve), and the
// common page count is the smallest count every rank can hold. With one rank, or with equal
// curves, this equals resolve_symmetric_kv_capacity.
[[nodiscard]] std::vector<KvCapacityResolution>
resolve_group_kv_capacity(const KvCapacityPolicy&, std::span<const RankCapacityOffer>);

// One capacity for a whole execution group, from one shared curve and one free-byte figure per
// rank (capy-flash `resolve_kv_capacity_symmetric`). The group bottleneck is the smallest figure
// and the automatic headroom is subtracted once from that bottleneck, never once per rank. A
// group of one rank reproduces resolve_kv_capacity exactly. A caller credits a rank with the
// reservation part it does not allocate before passing its figure.
[[nodiscard]] KvCapacityResolution
resolve_symmetric_kv_capacity(const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve,
                              std::span<const std::size_t> available_runtime_bytes_per_rank);

} // namespace ninfer::runtime
