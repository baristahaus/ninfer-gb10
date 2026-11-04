#include <cstdio>
#include "artifact/views.h"

namespace ninfer::artifact {

WeightView bind_view(const ParameterReference& reference, const MaterializedArtifact& materialized,
                     Device device) {
    WeightView out;
    out.shape             = reference.shape;
    std::uint64_t covered = 0;
    for (const auto& part : reference.binding.parts) {
        const bool auto_device = device.type() == DeviceType::None;
        const auto& parent =
            (device.type() == DeviceType::CUDA || (auto_device && materialized.has_device(part.object)))
                ? materialized.device_parent(part.object)
                : materialized.host_parent(part.object);
        if (!auto_device && parent.device != device) {
            throw ArtifactError("weight backing device differs");
        }
        if (part.begin >= part.end || part.end > parent.geometry.elements) {
            std::fprintf(stderr, "view-fail: %s part [%zu,%zu) parent elements=%zu shape0=%zu\n",
                         reference.name.c_str(), part.begin, part.end,
                         parent.geometry.elements,
                         parent.geometry.shape.empty() ? 0 : parent.geometry.shape[0]);
            throw ArtifactError(reference.name + ": invalid materialized region");
        }
        covered = checked_add(covered, part.end - part.begin, reference.name);
        out.parts.push_back({&parent, part.begin, part.end});
    }
    if (!reference.tiered && covered != weight_element_count(out.shape)) {
        throw ArtifactError(reference.name + ": materialized coverage differs from logical shape");
    }
    return out;
}

} // namespace ninfer::artifact
