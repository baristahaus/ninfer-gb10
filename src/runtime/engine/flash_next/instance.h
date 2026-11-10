#pragma once

// The Flash-Next family serves through its own master-generation engine route: the Program is
// called directly (no Executor protocol), and the Engine holds this instance beside the
// Executor-based model instances.

#include <ninfer/models/qwen3_8_flash_next_125b_a6b/package.h>

#include "artifact/materializer.h"
#include "runtime/contract/resources.h"

#include <memory>
#include <utility>
#include <vector>

namespace ninfer {
struct EngineOptions;
}

namespace ninfer::flashnext_runtime {

using namespace ninfer::runtime;

inline constexpr std::uint32_t kDiscretePrefillChunk   = 1024;
inline constexpr std::uint32_t kIntegratedPrefillChunk = 4096;

struct FlashNextInstance {
    using ModelContract = models::qwen3_8_flash_next_125b_a6b::Package;
    std::unique_ptr<ModelContract::LoadedModel> model;
    ModelContract::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<ModelContract::Program> program;

    FlashNextInstance(std::unique_ptr<ModelContract::LoadedModel> source,
                      const EngineOptions& options);
};

// One rank's immutable weights: the bindings that rank sees and the device bytes they address.
// Rank 0's live inside `FlashNextInstance::model`; the peers are carried on
// `runtime::ConstructedModel` so the mirrors and the sharded forward can take them without
// re-binding anything.
struct RankWeights {
    FlashNextInstance::ModelContract::LoadPlan plan;
    artifact::MaterializedArtifact backing;
};

} // namespace ninfer::flashnext_runtime
