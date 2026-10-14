#include "artifact/binder.h"

#include "artifact/framing.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

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

Binder::Binder(const Reader& reader)
    : reader_(reader), demands_(reader.directory().objects.size()) {}

ParameterReference Binder::parameter(std::string_view name, Shape shape, Residency residency,
                                     std::optional<QType> exact_format) {
    const auto found = reader_.directory().bindings.find(name);
    if (found == reader_.directory().bindings.end()) {
        throw ArtifactError("missing logical parameter " + std::string(name));
    }
    return binding(std::string(name), found->second, std::move(shape), residency, exact_format);
}

void Binder::set_tier(ObjectHandle object, const std::vector<std::uint32_t>& hot,
                      const std::vector<std::uint32_t>& warm) {
    auto& demand      = demands_.at(object.index);
    const auto& shape = reader_.geometry(object).shape;
    if (demand.device || demand.host || !demand.tier_new_order.empty() || shape.size() != 2 ||
        shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("tiered bank requires an untouched two-dimensional object");
    }
    if (hot.empty() || warm.empty() || shape[0] % (hot.size() + warm.size()) != 0) {
        throw ArtifactError("tiered bank requires a strict, nonempty hot/warm expert split "
                            "whose experts divide the rows");
    }
    std::vector<std::uint32_t> cover;
    cover.reserve(hot.size() + warm.size());
    cover.insert(cover.end(), hot.begin(), hot.end());
    cover.insert(cover.end(), warm.begin(), warm.end());
    std::sort(cover.begin(), cover.end());
    for (std::size_t i = 0; i < cover.size(); ++i) {
        if (cover[i] != i) {
            throw ArtifactError("tiered bank requires a disjoint hot/warm expert cover");
        }
    }
    // Both tiers normalize to expert id order: the per-expert views concatenate in expert
    // id order, so the compacted mini-bank slots must follow the same order.
    demand.tier_new_order.reserve(cover.size());
    demand.tier_new_order.insert(demand.tier_new_order.end(), hot.begin(), hot.end());
    demand.tier_new_order.insert(demand.tier_new_order.end(), warm.begin(), warm.end());
    std::sort(demand.tier_new_order.begin(),
              demand.tier_new_order.begin() +
                  static_cast<std::ptrdiff_t>(hot.size()));
    std::sort(demand.tier_new_order.begin() + static_cast<std::ptrdiff_t>(hot.size()),
              demand.tier_new_order.end());
    demand.tier_hot_experts = static_cast<std::uint32_t>(hot.size());
}

std::span<const std::uint32_t> Binder::tier_new_order(ObjectHandle object) const {
    return demands_.at(object.index).tier_new_order;
}

std::uint32_t Binder::tier_hot_experts(ObjectHandle object) const {
    return demands_.at(object.index).tier_hot_experts;
}

void Binder::set_device_budget(std::uint64_t bytes) { device_budget_ = bytes; }

ParameterReference Binder::binding(std::string name, const Binding& binding, Shape shape,
                                   Residency residency, std::optional<QType> exact_format) {
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
        const auto& tier_order = demands_.at(part.object.index).tier_new_order;
        if (!tier_order.empty()) {
            // SP-4 tiered bank: both residencies address the mini-banks, and each part
            // remaps to its expert's compacted position. The part must stay inside one
            // expert's row block so a whole-bank or straddling binding cannot alias the
            // compacted layout.
            auto& demand = demands_.at(part.object.index);
            const auto experts         = static_cast<std::uint64_t>(tier_order.size());
            const auto rows_per_expert = geometry.shape[0] / experts;
            const auto begin_row       = part.begin / geometry.shape[1];
            const auto end_row         = part.end / geometry.shape[1];
            if (geometry.shape[0] % experts != 0 ||
                begin_row / rows_per_expert != (end_row - 1) / rows_per_expert) {
                throw ArtifactError(name + ": tiered part must stay inside one expert's rows");
            }
            const auto expert = begin_row / rows_per_expert;
            const auto within = begin_row % rows_per_expert;
            const auto position = static_cast<std::size_t>(
                std::find(tier_order.begin(), tier_order.end(),
                          static_cast<std::uint32_t>(expert)) - tier_order.begin());
            if (position >= tier_order.size()) {
                throw ArtifactError(name + ": tiered bank lost its expert permutation");
            }
            const auto hot_experts = static_cast<std::uint64_t>(demand.tier_hot_experts);
            if (residency == Residency::Device) {
                if (position >= hot_experts) {
                    throw ArtifactError(name + ": warm-tier expert bound with device residency");
                }
                const auto rank = static_cast<std::uint64_t>(position);
                part.begin      = checked_mul(rank * rows_per_expert + within, geometry.shape[1],
                                              "tiered hot rows");
                part.end        = checked_add(part.begin,
                                              checked_mul(end_row - begin_row, geometry.shape[1],
                                                          "tiered span"),
                                              "tiered hot end");
                demand.device = true;
            } else if (residency == Residency::Host) {
                if (position < hot_experts) {
                    throw ArtifactError(name + ": hot-tier expert bound with host residency");
                }
                const auto rank = static_cast<std::uint64_t>(position) - hot_experts;
                part.begin      = checked_mul(rank * rows_per_expert + within, geometry.shape[1],
                                              "tiered warm rows");
                part.end        = checked_add(part.begin,
                                              checked_mul(end_row - begin_row, geometry.shape[1],
                                                          "tiered span"),
                                              "tiered warm end");
                demand.host = true;
            } else {
                throw ArtifactError(name + ": tiered banks do not support value residency");
            }
            continue;
        }
        if (residency == Residency::Device) {
            require_device(part.object);
        } else if (residency == Residency::Host) {
            (void)host_object(part.object);
        }
    }
    return {std::move(name), std::move(shape), std::move(bound), residency};
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

void Binder::require_device(ObjectHandle object, std::uint64_t alignment) {
    const auto& geometry = reader_.geometry(object);
    if (!alignment || (alignment & (alignment - 1))) {
        throw ArtifactError("device alignment must be a power of two");
    }
    auto& demand     = demands_.at(object.index);
    demand.device    = true;
    demand.alignment = std::max({demand.alignment, alignment, geometry.alignment});
}

std::span<const std::byte> Binder::host_object(ObjectHandle object) {
    reader_.validate_object(object);
    auto& demand = demands_.at(object.index);
    if (!demand.host) {
        demand.host_data = reader_.read_object(object);
        read_bytes_      = checked_add(read_bytes_, demand.host_data.size(), "Host read bytes");
        demand.host      = true;
    }
    return demand.host_data;
}

ObjectHandle Binder::resource(std::string_view component, std::string_view role) {
    const auto& resources = reader_.directory().component(component).resources;
    const auto found      = resources.find(role);
    if (found == resources.end()) {
        throw ArtifactError(std::string(component) + ": missing resource " + std::string(role));
    }
    (void)host_object(found->second);
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
        const auto& cached = demands_[part.object.index];
        if (cached.host) {
            std::memcpy(target.data(), cached.host_data.data() + source, target.size());
        } else {
            const auto& object = reader_.directory().tensor(part.object);
            reader_.read_into(checked_add(object.offset, source, "value file offset"), target);
            read_bytes_ = checked_add(read_bytes_, bytes, "Host read bytes");
        }
        destination += target.size();
    }
    if (destination != out.data.size()) {
        throw ArtifactError("Host value coverage is incomplete");
    }
    owned_value_bytes_ = checked_add(owned_value_bytes_, out.data.size(), "owning value bytes");
    return out;
}

MaterializationPlan Binder::finish() && {
    MaterializationPlan plan;
    plan.source            = &reader_;
    plan.object_count      = demands_.size();
    plan.prior_read_bytes  = read_bytes_;
    plan.owned_value_bytes = owned_value_bytes_;
    for (std::size_t i = 0; i < demands_.size(); ++i) {
        auto& demand = demands_[i];
        const ObjectHandle handle{i};
        if (!demand.tier_new_order.empty()) { continue; }
        if (demand.device) {
            const auto& geometry = reader_.geometry(handle);
            const auto offset =
                align_up(plan.device_capacity_bytes, demand.alignment, "device offset");
            plan.device_objects.push_back({handle, offset, geometry.bytes, demand.alignment});
            plan.device_capacity_bytes = checked_add(offset, geometry.bytes, "device capacity");
        }
        if (demand.host) {
            plan.host_objects.push_back({handle, std::move(demand.host_data)});
        }
    }
    // SP-4 tiered banks follow the plain device objects so the arena's sequential allocation
    // order matches the plan offsets: the device carries only each compact hot mini-bank; the
    // pinned-host warm mini-bank is allocated and filled by the materializer.
        struct TieredBank {
        ObjectHandle handle;
        std::size_t demand;
    };
    std::vector<TieredBank> tiered;
for (std::size_t i = 0; i < demands_.size(); ++i) {
        auto& demand = demands_[i];
        // A tier mark without any bound part is inert: the component owning the bank was not
        // selected, so the bank materializes nowhere (matching its plain unbound peers).
        if (demand.tier_new_order.empty() || (!demand.device && !demand.host)) { continue; }
        tiered.push_back({ObjectHandle{i}, i});
    }
    // SP-4 adaptive cache: one uniform slot count across every tiered bank — a promoted
    // expert's paired banks (routed gate/up and down) must carry matching slots, and one slot
    // costs its per-expert bytes in every bank. The device budget covers the whole plan: the
    // plain device objects and the tiered hot mini-banks consume first, and the remaining
    // bytes buy as many whole expert slices as fit, never exceeding the smallest bank's warm
    // count.
    std::uint32_t cache_slots = 0;
    if (!tiered.empty() && device_budget_ != 0) {
        std::uint64_t slice_cost = 0;
        std::uint64_t min_warm   = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t hot_bytes  = 0;
        for (const auto& bank : tiered) {
            const auto& demand   = demands_[bank.demand];
            const auto& geometry = reader_.geometry(bank.handle);
            const auto experts   = demand.tier_new_order.size();
            slice_cost           = checked_add(slice_cost, geometry.bytes / experts,
                                               "expert cache slice cost");
            min_warm = std::min(min_warm, experts - demand.tier_hot_experts);
            hot_bytes = checked_add(
                hot_bytes,
                checked_mul(demand.tier_hot_experts, geometry.bytes / experts, "tiered hot bytes"),
                "tiered hot total");
        }
        const auto weights_bytes = checked_add(plan.device_capacity_bytes, hot_bytes,
                                               "tiered weights bytes");
        const auto tail_budget   = device_budget_ > weights_bytes ? device_budget_ - weights_bytes
                                                                : 0;
        cache_slots = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(min_warm, tail_budget / slice_cost));
    }
    for (const auto& bank : tiered) {
        const auto& demand   = demands_[bank.demand];
        const auto& geometry = reader_.geometry(bank.handle);
        const auto experts   = demand.tier_new_order.size();
        const auto bank_bytes =
            checked_mul(demand.tier_hot_experts + cache_slots, geometry.bytes / experts,
                        "tiered device bytes");
        const auto offset = align_up(plan.device_capacity_bytes, demand.alignment, "device offset");
        plan.tier_objects.push_back({bank.handle, offset, bank_bytes, demand.alignment,
                                     demand.tier_hot_experts, cache_slots, demand.tier_new_order});
        plan.device_capacity_bytes = checked_add(offset, bank_bytes, "device capacity");
    }
    return plan;
}

} // namespace ninfer::artifact
