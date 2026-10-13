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
    // Declares a routed expert bank as tier-split (strategy SP-4 phase 2a): the first
    // `hot_rows` rows of the object materialize on the device as a compact hot mini-bank,
    // the remaining rows live in a pinned-host warm mini-bank, and Host-residency parts on
    // this object address the warm mini-bank at suffix-relative element ranges.
    void set_tier(ObjectHandle object, std::uint64_t hot_rows);
    // SP-4 tiered bank state: the registered hot-row prefix, or zero for an untiered object.
    [[nodiscard]] std::uint64_t tier_hot_rows(ObjectHandle object) const;
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
        // SP-4 tiered bank: nonzero marks the split; device/host demands then describe the
        // hot and warm mini-banks instead of the whole object.
        std::uint64_t tier_hot_rows = 0;
    };

    const Reader& reader_;
    std::vector<Demand> demands_;
    std::uint64_t read_bytes_        = 0;
    std::uint64_t owned_value_bytes_ = 0;
};

} // namespace ninfer::artifact
