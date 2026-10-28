#include "artifact/binder.h"

#include "artifact/framing.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <utility>

namespace ninfer::artifact {

float HostValues::scalar_f32() const {
    if (format != QType::FP32 || elements != 1 || data.size() != 4) {
        throw ArtifactError("auxiliary requires one represented FP32 value");
    }
    return std::bit_cast<float>(read_u32_le(data.data()));
}

std::vector<std::int32_t> HostValues::integers() const {
    if (format != QType::INT32 || data.size() != checked_mul(elements, 4, "integer values")) {
        throw ArtifactError("semantic table requires INT32 values");
    }
    std::vector<std::int32_t> out;
    out.reserve(static_cast<std::size_t>(elements));
    for (std::size_t i = 0; i < elements; ++i) {
        out.push_back(std::bit_cast<std::int32_t>(read_u32_le(data.data() + i * 4)));
    }
    return out;
}

Binder::Binder(const Reader& reader) : reader_(reader) {}

ParameterReference Binder::parameter(std::string_view name, Shape shape,
                                     std::optional<QType> exact_format) {
    const auto found = reader_.directory().bindings.find(name);
    if (found == reader_.directory().bindings.end()) {
        throw ArtifactError("missing logical parameter " + std::string(name));
    }
    return binding(std::string(name), found->second, std::move(shape), exact_format);
}

ParameterReference Binder::parameter(std::string_view name, Shape shape, Residency residency,
                                     std::optional<QType> exact_format) {
    auto out = parameter(name, std::move(shape), std::move(exact_format));
    out.residency = residency;
    if (residency == Residency::Device) {
        for (const auto& part : out.binding.parts) {
            const bool known =
                std::find(device_objects_.begin(), device_objects_.end(), part.object) !=
                device_objects_.end();
            if (!known) { device_objects_.push_back(part.object); }
        }
    }
    return out;
}

Binder::TierState& Binder::tier_state(ObjectHandle object) {
    if (object.index >= tier_states_.size()) {
        tier_states_.resize(static_cast<std::size_t>(object.index) + 1);
    }
    return tier_states_[object.index];
}

void Binder::set_tier(ObjectHandle object, const std::vector<std::uint32_t>& hot,
                      const std::vector<std::uint32_t>& warm) {
    reader_.validate_object(object);
    const auto& tensor = reader_.directory().tensor(object);
    auto& tier         = tier_state(object);
    if (tier.device || tier.host || !tier.new_order.empty() || tensor.shape.size() != 2 ||
        tensor.shape[0] == 0 || tensor.shape[1] == 0 || hot.empty()) {
        throw ArtifactError("tiered bank requires an untouched two-dimensional object");
    }
    const auto& geometry = reader_.geometry(object);
    if (geometry.format == QType::EXL3) {
        // SP-4's tiered packing assumes one constant per-expert byte size per bank; EXL3's
        // per-expert bitrates give unequal extents, so tiering needs the SP-4 variable-extent
        // support that does not exist yet.
        throw ArtifactError("tiered EXL3 weights (banks and dense linears) require the "
                            "variable-extent tiering support that does not exist yet");
    }
    if (hot.size() > tensor.shape[0] ||
        hot.size() + warm.size() != tensor.shape[0]) {
        throw ArtifactError("tiered bank requires a strict, nonempty hot/warm expert split "
                            "covering the bank");
    }
    std::vector<std::uint32_t> cover;
    cover.reserve(hot.size() + warm.size());
    cover.insert(cover.end(), hot.begin(), hot.end());
    cover.insert(cover.end(), warm.begin(), warm.end());
    std::sort(cover.begin(), cover.end());
    if (std::adjacent_find(cover.begin(), cover.end()) != cover.end()) {
        throw ArtifactError("tiered bank requires a disjoint hot/warm expert cover");
    }
    // Both tiers normalize to expert id order: the per-expert views concatenate in expert
    // id order inside each mini-bank.
    tier.new_order.reserve(cover.size());
    tier.new_order.insert(tier.new_order.end(), hot.begin(), hot.end());
    tier.new_order.insert(tier.new_order.end(), warm.begin(), warm.end());
    std::sort(tier.new_order.begin(),
              tier.new_order.begin() + static_cast<std::ptrdiff_t>(hot.size()));
    std::sort(tier.new_order.begin() + static_cast<std::ptrdiff_t>(hot.size()),
              tier.new_order.end());
    tier.hot_experts = static_cast<std::uint32_t>(hot.size());
}

std::span<const std::uint32_t> Binder::tier_new_order(ObjectHandle object) const {
    return object.index < tier_states_.size() ? tier_states_[object.index].new_order
                                              : std::span<const std::uint32_t>{};
}

std::uint32_t Binder::tier_hot_experts(ObjectHandle object) const {
    return object.index < tier_states_.size() ? tier_states_[object.index].hot_experts : 0;
}

std::vector<TierPlanInput> Binder::tier_inputs() const {
    std::vector<TierPlanInput> out;
    for (std::size_t i = 0; i < tier_states_.size(); ++i) {
        const auto& tier = tier_states_[i];
        // A tier mark without any bound part is inert: the component owning the bank was not
        // selected, so the bank materializes nowhere (matching its plain unbound peers).
        if (tier.new_order.empty() || (!tier.device && !tier.host)) { continue; }
        const auto object = ObjectHandle{static_cast<std::uint32_t>(i)};
        out.push_back({object, tier.new_order, tier.hot_experts,
                       reader_.geometry(object).alignment});
    }
    return out;
}

ParameterReference Binder::binding(std::string name, const Binding& binding, Shape shape,
                                   Residency residency, std::optional<QType> exact_format) {
    auto out = this->binding(std::move(name), binding, std::move(shape), std::move(exact_format));
    out.residency = residency;
    for (auto& part : out.binding.parts) {
        if (part.object.index >= tier_states_.size()) { continue; }
        auto& tier = tier_states_[part.object.index];
        if (tier.new_order.empty()) { continue; }
        // SP-4 tiered bank: both residencies address the mini-banks, and each part
        // remaps to its expert's compacted position. The part must stay inside one
        // expert's row block so a whole-bank or straddling binding cannot alias the
        // compacted layout.
        const auto& geometry = reader_.geometry(part.object);
        const auto experts         = static_cast<std::uint64_t>(tier.new_order.size());
        const auto rows_per_expert = geometry.shape[0] / experts;
        const auto begin_row       = part.begin / geometry.shape[1];
        const auto end_row         = part.end / geometry.shape[1];
        if (geometry.shape[0] % experts != 0 ||
            begin_row / rows_per_expert != (end_row - 1) / rows_per_expert) {
            throw ArtifactError(out.name + ": tiered part must stay inside one expert's rows");
        }
        const auto expert = begin_row / rows_per_expert;
        const auto within = begin_row % rows_per_expert;
        const auto position = static_cast<std::size_t>(
            std::find(tier.new_order.begin(), tier.new_order.end(),
                      static_cast<std::uint32_t>(expert)) - tier.new_order.begin());
        if (position >= tier.new_order.size()) {
            throw ArtifactError(out.name + ": tiered bank lost its expert permutation");
        }
        const auto hot_experts = static_cast<std::uint64_t>(tier.hot_experts);
        if (residency == Residency::Device) {
            if (position >= hot_experts) {
                throw ArtifactError(out.name + ": warm-tier expert bound with device residency");
            }
            const auto rank = static_cast<std::uint64_t>(position);
            part.begin      = checked_mul(rank * rows_per_expert + within, geometry.shape[1],
                                          "tiered hot rows");
            part.end        = checked_add(part.begin,
                                          checked_mul(end_row - begin_row, geometry.shape[1],
                                                      "tiered span"),
                                          "tiered hot end");
            tier.device = true;
        } else if (residency == Residency::Host) {
            if (position < hot_experts) {
                throw ArtifactError(out.name + ": hot-tier expert bound with host residency");
            }
            const auto rank = static_cast<std::uint64_t>(position) - hot_experts;
            part.begin      = checked_mul(rank * rows_per_expert + within, geometry.shape[1],
                                          "tiered warm rows");
            part.end        = checked_add(part.begin,
                                          checked_mul(end_row - begin_row, geometry.shape[1],
                                                      "tiered span"),
                                          "tiered warm end");
            tier.host = true;
        } else {
            throw ArtifactError(out.name + ": tiered banks do not support value residency");
        }
    }
    if (residency == Residency::Device) {
        for (const auto& part : out.binding.parts) {
            const bool known =
                std::find(device_objects_.begin(), device_objects_.end(), part.object) !=
                device_objects_.end();
            if (!known) { device_objects_.push_back(part.object); }
        }
    }
    return out;
}

ParameterReference Binder::binding(std::string name, const Binding& binding, Shape shape,
                                   std::optional<QType> exact_format) {
    if (weight_element_count(shape) != binding.elements ||
        (binding.whole_object &&
         reader_.directory().tensor(binding.parts.at(0).object).shape != shape)) {
        throw ArtifactError(name + ": logical shape differs from Binding coverage");
    }
    Binding bound = binding;
    for (auto& part : bound.parts) {
        const auto& geometry = reader_.geometry(part.object);
        if (exact_format && geometry.format != *exact_format) {
            throw ArtifactError(name +
                                ": representation does not match its mathematical value type");
        }
    }
    return {std::move(name), std::move(shape), binding};
}

const Use& Binder::use(std::string_view parameter, std::string_view input) const {
    const auto found = reader_.directory().uses.find({std::string(parameter), std::string(input)});
    if (found == reader_.directory().uses.end()) {
        throw ArtifactError("missing Use " + std::string(parameter) + "@" + std::string(input));
    }
    return found->second;
}

bool Binder::contains(std::string_view parameter) const {
    return reader_.directory().bindings.contains(parameter);
}

std::span<const std::byte> Binder::resource_bytes(ObjectHandle object) {
    reader_.validate_object(object);
    if (std::holds_alternative<TensorObject>(reader_.directory().object(object))) {
        throw ArtifactError("tensor storage belongs to materialization, not prepared resources");
    }
    for (const auto& resource : resources_) {
        if (resource.object == object) { return resource.data; }
    }
    auto bytes  = reader_.read_object(object);
    read_bytes_ = checked_add(read_bytes_, bytes.size(), "resource read bytes");
    resources_.push_back({object, std::move(bytes)});
    return resources_.back().data;
}

ObjectHandle Binder::resource(std::string_view component, std::string_view role) {
    const auto& resources = reader_.directory().component(component).resources;
    const auto found      = resources.find(role);
    if (found == resources.end()) {
        throw ArtifactError(std::string(component) + ": missing resource " + std::string(role));
    }
    (void)resource_bytes(found->second);
    return found->second;
}

HostValues Binder::values(const Binding& binding, std::optional<QType> format) {
    HostValues out;
    out.elements = binding.elements;
    if (binding.parts.empty()) { throw ArtifactError("value Binding is empty"); }
    out.format = format.value_or(reader_.geometry(binding.parts.front().object).format);
    std::uint64_t word_bytes = 0;
    if (out.format == QType::BF16) {
        word_bytes = 2;
    } else if (out.format == QType::FP32 || out.format == QType::INT32) {
        word_bytes = 4;
    } else {
        throw ArtifactError("owning Host values require a direct numeric format");
    }
    const auto total_bytes = checked_mul(out.elements, word_bytes, "Host value bytes");
    if (total_bytes > std::numeric_limits<std::size_t>::max()) {
        throw ArtifactError("Host values exceed size_t");
    }
    out.data.resize(static_cast<std::size_t>(total_bytes));
    std::size_t destination = 0;
    for (const auto& part : binding.parts) {
        const auto& geometry = reader_.geometry(part.object);
        if (geometry.layout != QuantLayout::Contiguous || geometry.format != out.format ||
            part.begin >= part.end || part.end > geometry.elements) {
            throw ArtifactError("Host value Binding has an incompatible representation or range");
        }
        const auto bytes = checked_mul(part.end - part.begin, word_bytes, "value range");
        if (bytes > out.data.size() - destination) {
            throw ArtifactError("Host value coverage exceeds declared size");
        }
        const auto source = checked_mul(part.begin, word_bytes, "value offset");
        auto target = std::span(out.data).subspan(destination, static_cast<std::size_t>(bytes));
        const auto& object = reader_.directory().tensor(part.object);
        reader_.read_into(checked_add(object.offset, source, "value file offset"), target);
        read_bytes_ = checked_add(read_bytes_, bytes, "Host read bytes");
        destination += target.size();
    }
    if (destination != out.data.size()) {
        throw ArtifactError("Host value coverage is incomplete");
    }
    owned_value_bytes_ = checked_add(owned_value_bytes_, out.data.size(), "owning value bytes");
    return out;
}

std::vector<ObjectHandle> Binder::device_objects() const { return device_objects_; }

PreparedResources Binder::finish() && {
    PreparedResources out;
    out.source            = &reader_;
    out.objects           = std::move(resources_);
    out.read_bytes        = read_bytes_;
    out.owned_value_bytes = owned_value_bytes_;
    return out;
}

} // namespace ninfer::artifact
