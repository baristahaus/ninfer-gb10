#pragma once

#include "models/common/sharded_group.h"
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
    // The two-GPU lane's group-scoped resources: the ONE mailbox and the ONE partial pair (plus the
    // logits pair) that both Programs share. Declared BEFORE `instance` so it is destroyed AFTER the
    // Programs that borrow it: the mailbox's pinned slab, its per-device words and the partial
    // addresses are baked into every exchange, so nothing that uses them may outlive them. Null at
    // width 1, where nothing is constructed.
    //
    // THIS FIELD IS THE OWNER. `build_local_model` builds it in a local `BuiltModel`, and before
    // this field existed the local's destructor freed it at the end of `construct_model` — during
    // startup — while both Programs kept raw pointers into it. The first column-parallel projection
    // then wrote into freed device memory (cudaErrorIllegalAddress, Xid 31). Carrying the owner
    // across the boundary is what keeps it alive for the Engine's lifetime.
    std::unique_ptr<models::ShardedGroupOwner> sharded;
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
