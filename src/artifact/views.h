#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/weight_view.h"

namespace ninfer::artifact {

// An explicit CUDA or CPU device binds that storage. A default (None) device binds the device
// storage when the object has one and the Host storage otherwise; master-generation programs
// that describe residency through the plan itself bind this way.
[[nodiscard]] WeightView bind_view(
    const ParameterReference& reference, const MaterializedArtifact& materialized,
    Device device = {});

} // namespace ninfer::artifact
