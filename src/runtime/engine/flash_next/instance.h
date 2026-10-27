#pragma once

// The Flash-Next family serves through its own master-generation engine route: the Program is
// called directly (no Executor protocol), and the Engine holds this instance beside the
// Executor-based model instances.

#include <ninfer/models/qwen3_8_flash_next_125b_a6b/package.h>

#include "runtime/contract/resources.h"

#include <memory>
#include <utility>

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

} // namespace ninfer::flashnext_runtime
