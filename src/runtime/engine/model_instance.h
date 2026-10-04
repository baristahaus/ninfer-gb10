#pragma once

#include "models/qwen3_5/model.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/runtime_types.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/kv_capacity.h"

#include <memory>
#include <variant>
#include <ninfer/models/qwen3_8_flash_next_125b_a6b/package.h>

namespace ninfer::runtime {

// EngineOptions::prefill_chunk zero resolves to these once the device is known. GB10 measured
// 60K-token prefill at 1,549 tok/s with 1024-token chunks and 2,134 with 4096.
inline constexpr std::uint32_t kDiscretePrefillChunk   = 1024;
inline constexpr std::uint32_t kIntegratedPrefillChunk = 4096;

[[nodiscard]] EngineOptions normalize_engine_options(EngineOptions options);

struct ModelInstance {
    using ModelContract = models::qwen3_5::RuntimeTypes;

    std::unique_ptr<models::qwen3_5::Model> model;
    const models::qwen3_5::execution::Parameters parameters;
    models::qwen3_5::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<models::qwen3_5::Program> program;

    ModelInstance(std::unique_ptr<models::qwen3_5::Model> model, const EngineOptions& options);
    ~ModelInstance();
    ModelInstance(const ModelInstance&)            = delete;
    ModelInstance& operator=(const ModelInstance&) = delete;
};

struct FlashNextInstance {
    using ModelContract = models::qwen3_8_flash_next_125b_a6b::Package;
    std::unique_ptr<ModelContract::LoadedModel> model;
    ModelContract::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<ModelContract::Program> program;

    FlashNextInstance(std::unique_ptr<ModelContract::LoadedModel> source,
                      const EngineOptions& options)
        : model(std::move(source)), frontend(ModelContract::make_frontend(*model, options)),
          capacity(options.max_context) {}
};

using ActiveModel =
    std::variant<std::unique_ptr<ModelInstance>, std::unique_ptr<FlashNextInstance>>;

struct ConstructedModel {
    ActiveModel instance;
    LoadSummary load;
    ContextMachineCostModel context_cost;
};

[[nodiscard]] ConstructedModel construct_model(const EngineOptions& options, DeviceContext& device);

} // namespace ninfer::runtime
