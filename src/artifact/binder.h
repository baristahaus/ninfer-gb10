#pragma once

#include "artifact/materializer.h"
#include "artifact/reader.h"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::artifact {

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
                                               Residency residency = Residency::Device,
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
    [[nodiscard]] ObjectHandle resource(std::string_view component, std::string_view role);
    [[nodiscard]] HostValues values(const Binding& binding, std::optional<QType> format = {});
    [[nodiscard]] MaterializationPlan finish() &&;

private:
    struct Demand {
        bool device             = false;
        bool host               = false;
        std::uint64_t alignment = 256;
        std::vector<std::byte> host_data;
        // SP-4 tiered bank: nonempty marks the split; device/host demands then describe the
        // hot and warm mini-banks instead of the whole object. The permutation lists the hot
        // experts (expert id order) followed by the warm experts; `tier_hot_experts` is the
        // boundary between the two tiers.
        std::vector<std::uint32_t> tier_new_order;
        std::uint32_t tier_hot_experts = 0;
    };

    const Reader& reader_;
    std::vector<Demand> demands_;
    std::uint64_t device_budget_ = 0;
    std::uint32_t cache_slot_cap_ = 0;
    std::uint64_t read_bytes_        = 0;
    std::uint64_t owned_value_bytes_ = 0;
};

} // namespace ninfer::artifact
