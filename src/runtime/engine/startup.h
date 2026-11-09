#pragma once

#include "runtime/engine/instance.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/flash_next/instance.h"
#include "core/device.h"
#include "models/qwen3_5/runtime_types.h"
#include "models/qwen4_exp/runtime_types.h"

#include <memory>
#include <variant>

namespace ninfer::runtime {

[[nodiscard]] SessionEpoch next_session_epoch() noexcept;

[[nodiscard]] EngineOptions normalize_engine_options(EngineOptions options);

// Finite architecture dispatch ends at Engine startup; inner execution remains statically typed.
using ActiveModel =
    std::variant<std::unique_ptr<InferenceInstance<models::qwen3_5::RuntimeTypes>>,
                 std::unique_ptr<InferenceInstance<models::qwen4_exp::RuntimeTypes>>,
                 std::unique_ptr<flashnext_runtime::FlashNextInstance>>;

struct ConstructedModel {
    ActiveModel instance;
    LoadSummary load;
    ContextMachineCostModel context_cost;
    // The Flash-Next route owns its device group here; the other routes build devices inside
    // their Startup classes. Rank 0 is the execution device, so a width-1 group behaves exactly
    // as a single DeviceContext did.
    std::unique_ptr<ExecutionContext> execution;
};

[[nodiscard]] ConstructedModel construct_model(EngineOptions&);

} // namespace ninfer::runtime
