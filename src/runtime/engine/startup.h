#pragma once

#include "runtime/engine/instance.h"
#include "runtime/executor/rank_worker.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/flash_next/instance.h"
#include "core/device.h"
#include "models/qwen3_5/runtime_types.h"
#include "models/qwen4_exp/runtime_types.h"

#include <memory>
#include <variant>
#include <vector>

namespace ninfer::runtime {

[[nodiscard]] SessionEpoch next_session_epoch() noexcept;

[[nodiscard]] EngineOptions normalize_engine_options(EngineOptions options);

// Finite architecture dispatch ends at Engine startup; inner execution remains statically typed.
using ActiveModel =
    std::variant<std::unique_ptr<InferenceInstance<models::qwen3_5::RuntimeTypes>>,
                 std::unique_ptr<InferenceInstance<models::qwen4_exp::RuntimeTypes>>,
                 std::unique_ptr<flashnext_runtime::FlashNextInstance>>;

struct ConstructedModel {
    // The group is declared first so it is destroyed last. Both routes now borrow their rank's
    // device rather than owning it: the Qwen3.5 worker holds a reference, and the Flash-Next
    // Program holds references to the pools it built on it.
    std::unique_ptr<ExecutionContext> execution;
    ActiveModel instance;
    LoadSummary load;
    ContextMachineCostModel context_cost;
    // Flash-Next only: the peer ranks' immutable weights, one entry per rank after rank 0, in
    // rank order. Rank 0's plan and backing live inside `instance`. Empty at width 1 and on the
    // other routes, so the single-device lane carries nothing.
    std::vector<flashnext_runtime::RankWeights> peer_weights;
    // The Qwen3.5 route's peer ranks' materialized shards (ranks 1..width-1), read from their own
    // definitions. Empty at width 1 and on the other routes. The peer Program and its mirrors are
    // the next step; this is the storage they will be built on.
    std::vector<std::unique_ptr<RankModelInstance<models::qwen3_5::RuntimeTypes>>> qwen3_5_peers;
};

[[nodiscard]] ConstructedModel construct_model(EngineOptions&);

} // namespace ninfer::runtime
