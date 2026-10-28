#pragma once

#include "artifact/placement.h"
#include "artifact/reader.h"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::artifact {

// Where a parameter's payload belongs at load time. Device is the default; Host marks the
// file-mapped host tier (the Flash-Next PLE table); Values marks an inline host value.
enum class Residency { Device, Host, Values };

struct ParameterReference {
    std::string name;
    Shape shape;
    Binding binding;
    Residency residency = Residency::Device;
};

struct HostValues {
    QType format           = QType::FP32;
    std::uint64_t elements = 0;
    std::vector<std::byte> data;

    [[nodiscard]] float scalar_f32() const;
    [[nodiscard]] std::vector<std::int32_t> integers() const;
};

// Collects selected logical demands; neither physical object IDs nor whole-artifact profiles
// determine parameter shapes or native operation support.
class Binder {
public:
    explicit Binder(const Reader& reader);

    [[nodiscard]] ParameterReference parameter(std::string_view name, Shape shape,
                                               std::optional<QType> exact_format = {});
    [[nodiscard]] ParameterReference parameter(std::string_view name, Shape shape,
                                               Residency residency,
                                               std::optional<QType> exact_format = {});
    [[nodiscard]] ParameterReference binding(std::string name, const Binding& binding, Shape shape,
                                             std::optional<QType> exact_format = {});
    [[nodiscard]] ParameterReference binding(std::string name, const Binding& binding, Shape shape,
                                             Residency residency,
                                             std::optional<QType> exact_format = {});
    [[nodiscard]] const Use& use(std::string_view parameter, std::string_view input) const;
    [[nodiscard]] bool contains(std::string_view parameter) const;

    [[nodiscard]] const Reader& reader() const noexcept { return reader_; }

    void require_device(ObjectHandle object, std::uint64_t alignment = 256);
    // Declares a routed expert bank as tier-split (strategy SP-4): `hot` and `warm` are
    // disjoint expert id lists covering the bank's experts. The hot experts materialize on
    // the device as a compact hot mini-bank in expert id order; the warm experts live in a
    // pinned-host warm mini-bank in expert id order; parts remap to their expert's
    // compacted position in the matching mini-bank.
    void set_tier(ObjectHandle object, const std::vector<std::uint32_t>& hot,
                  const std::vector<std::uint32_t>& warm);
    // SP-4 tiered bank state: the registered expert permutation (the hot experts in expert
    // id order, then the warm experts), empty for an untiered object.
    [[nodiscard]] std::span<const std::uint32_t> tier_new_order(ObjectHandle object) const;
    // SP-4 tiered bank state: the hot expert count — the permutation prefix on the device.
    [[nodiscard]] std::uint32_t tier_hot_experts(ObjectHandle object) const;
    // SP-4 adaptive cache: the total device-bytes budget for the materialization plan. The
    // plain device objects and the tiered hot mini-banks consume first; every tiered bank then
    // gains a cache tail of equal slot count (a promoted expert needs its paired banks to
    // match), sized so one slot costs its per-expert bytes in every bank. Zero (or a budget
    // the weights already exceed) keeps the tiered placement at its planned hot set with no
    // tail.
    void set_device_budget(std::uint64_t bytes);
    // SP-4 adaptive cache: hard cap on tail slots per bank, overriding the budget-derived
    // count. Used to force the tail-limited case (where the tail cannot cover the warm tier)
    // on models whose free VRAM would otherwise cover it.
    void set_cache_slot_cap(std::uint32_t slots);
    [[nodiscard]] std::span<const std::byte> host_object(ObjectHandle object);
    // SP-4 tier planning inputs: the touched tiered banks with their permutations, for the
    // materialization planner to place after the plain device objects.
    [[nodiscard]] std::vector<TierPlanInput> tier_inputs() const;
    [[nodiscard]] std::uint64_t device_budget() const noexcept { return device_budget_; }
    [[nodiscard]] std::uint32_t cache_slot_cap() const noexcept { return cache_slot_cap_; }
    [[nodiscard]] std::span<const std::byte> resource_bytes(ObjectHandle object);
    [[nodiscard]] ObjectHandle resource(std::string_view component, std::string_view role);
    [[nodiscard]] HostValues values(const Binding& binding, std::optional<QType> format = {});
    [[nodiscard]] PreparedResources finish() &&;
    // Device-resident tensor objects bound so far, deduplicated in first-binding order. Callers
    // turn them into PlacementRequests for plan_materialization.
    [[nodiscard]] std::vector<ObjectHandle> device_objects() const;

private:
    const Reader& reader_;
    std::vector<PreparedResource> resources_;
    std::vector<ObjectHandle> device_objects_;
    // SP-4 tiered banks: per-object tier metadata (the permutation and the hot count), plus
    // the adaptive-cache budget inputs applied by the materialization plan.
    struct TierState {
        std::vector<std::uint32_t> new_order;
        std::uint32_t hot_experts = 0;
        bool device               = false;
        bool host                 = false;
    };
    [[nodiscard]] TierState& tier_state(ObjectHandle object);
    std::vector<TierState> tier_states_;
    std::uint64_t device_budget_  = 0;
    std::uint32_t cache_slot_cap_ = 0;
    std::uint64_t read_bytes_        = 0;
    std::uint64_t owned_value_bytes_ = 0;
};

} // namespace ninfer::artifact
