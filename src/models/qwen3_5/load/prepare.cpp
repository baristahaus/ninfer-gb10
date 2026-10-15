#include "models/qwen3_5/load/bindings.h"

#include "artifact/views.h"

#include <cmath>

namespace ninfer::models::qwen3_5::loading {

WeightUseId Bindings::use(WeightId id, std::string_view input) const {
    const auto& parameter = at(id);
    for (std::size_t i = 0; i < parameter.uses.size(); ++i) {
        if (parameter.uses[i].input == input) { return {id, i}; }
    }
    throw artifact::ArtifactError(parameter.reference.name + ": unresolved Use " +
                                  std::string(input));
}

WeightId Bindings::parameter(std::string name, artifact::Shape shape,
                             std::vector<std::string> inputs, std::optional<QType> exact_format) {
    if (parameters_.contains(name)) {
        throw artifact::ArtifactError(name + ": duplicate model parameter declaration");
    }
    PendingWeight pending;
    pending.reference = bind_parameter(name, shape, exact_format);
    for (const auto& input : inputs) {
        const auto& use = binder.use(name, input);
        if (!use.activation_policy) {
            throw artifact::ArtifactError(name + "@" + input + ": missing activation policy");
        }
        WeightUse result;
        result.input = input;
        switch (*use.activation_policy) {
        case artifact::ActivationPolicy::A16Only:
            result.policy = ops::LinearPolicy::A16Only;
            break;
        case artifact::ActivationPolicy::AllowA8:
            result.policy = ops::LinearPolicy::AllowA8;
            break;
        case artifact::ActivationPolicy::AllowA4:
            result.policy = ops::LinearPolicy::AllowA4;
            break;
        case artifact::ActivationPolicy::AllowA16:
            // AllowA16 admits BF16 activations only (W4A16 expert banks); the
            // activation-side contract is the A16 path.
            result.policy = ops::LinearPolicy::A16Only;
            break;
        }
        for (const auto& [role, binding] : use.auxiliaries) {
            if (role != "activation_input_divisor") {
                throw artifact::ArtifactError(name + "@" + input + ": unknown auxiliary " + role);
            }
            const auto value = binder.values(binding, QType::FP32).scalar_f32();
            if (!std::isfinite(value) || value <= 0) {
                throw artifact::ArtifactError(name + "@" + input +
                                              ": activation divisor must be positive finite FP32");
            }
            result.activation_input_divisor = value;
        }
        pending.uses.push_back(std::move(result));
    }
    for (const auto& part : pending.reference.binding.parts) {
        pending.source_objects.push_back(
            artifact::object_id(binder.reader().directory().object(part.object)));
    }
    const WeightId id{weights.size()};
    parameters_.emplace(std::move(name), id);
    weights.push_back(std::move(pending));
    return id;
}

// SP-4: a parameter's residency follows its parts' tier state. A part inside a tiered
// bank's hot experts stays Device-resident; a part on a warm expert becomes Host-resident
// and addresses the pinned-host warm mini-bank. A parameter mixing hot and warm parts falls
// through as Host and the binder rejects its hot part at the residency check.
artifact::ParameterReference
Bindings::bind_parameter(const std::string& name, const artifact::Shape& shape,
                         const std::optional<QType>& exact_format) {
    const auto& directory = binder.reader().directory();
    const auto found      = directory.bindings.find(name);
    if (found == directory.bindings.end()) {
        throw artifact::ArtifactError("missing logical parameter " + name);
    }
    auto residency = artifact::Residency::Device;
    for (const auto& part : found->second.parts) {
        const auto& tier_order = binder.tier_new_order(part.object);
        if (tier_order.empty()) { continue; }
        const auto& geometry = binder.reader().geometry(part.object);
        const auto experts   = static_cast<std::uint64_t>(tier_order.size());
        const auto row       = part.begin / geometry.shape[1];
        const auto expert    = row / (geometry.shape[0] / experts);
        const auto position  = static_cast<std::uint64_t>(
            std::find(tier_order.begin(), tier_order.end(),
                      static_cast<std::uint32_t>(expert)) - tier_order.begin());
        if (position >= experts || position >= binder.tier_hot_experts(part.object)) {
            residency = artifact::Residency::Host;
        }
    }
    return binder.binding(name, found->second, shape, residency, exact_format);
}

WeightId Bindings::direct(std::string name, artifact::Shape shape, QType format) {
    return parameter(std::move(name), std::move(shape), {}, format);
}

std::vector<BoundWeight> resolve_weights(std::vector<PendingWeight>&& pending,
                                         const artifact::MaterializedArtifact& materialized) {
    std::vector<BoundWeight> out;
    out.reserve(pending.size());
    for (auto& item : pending) {
        auto view = artifact::bind_view(item.reference, materialized);
        out.push_back({std::move(item.reference.name), std::move(item.source_objects),
                       std::move(view), std::move(item.uses)});
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
