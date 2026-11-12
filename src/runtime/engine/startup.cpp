#include "runtime/engine/startup.h"
#include "runtime/engine/startup.inl"
#include "runtime/engine/flash_next/startup.h"
#include "models/qwen3_5/startup.h"
#include "models/qwen3_5/measurement.h"
#include "models/qwen4_exp/startup.h"
#if NINFER_ENABLE_DISTRIBUTED
#    include "runtime/engine/distributed_startup.inl"
#    include "models/qwen4_exp/parallel/startup.h"
#endif

#include <atomic>
#include <limits>
#include <stdexcept>

namespace ninfer::runtime {
namespace {
using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    // A device group is a tensor-parallel request, not a preference: an architecture without a
    // group route would silently run on rank 0's device alone and the caller would have no way to
    // tell. Rejecting here is the difference between "the lane is not implemented for this model"
    // and "it ran, slowly, on one card".
    for (const int device : options.devices) {
        if (device < 0) {
            throw std::invalid_argument("Engine device ordinals must not be negative");
        }
    }
    if (options.devices.size() > 1) {
        const artifact::Reader probe(options.artifact_path);
        switch (models::resolve_architecture(probe.directory())) {
        case models::Architecture::Qwen3_8FlashNext:
            break;
        case models::Architecture::Qwen3_5:
            // The 27B's group route: one process, two devices, one plan per rank (the same shape
            // the Flash-Next lane uses). The MoE architecture stays refused until its expert
            // placement is wired.
            break;
        case models::Architecture::Qwen3_5Moe:
            throw std::invalid_argument(
                "tensor parallelism is not implemented for the MoE architecture yet; pass a single "
                "device with --device");
        case models::Architecture::Qwen4Exp:
#if NINFER_ENABLE_DISTRIBUTED
            break;
#else
            throw std::invalid_argument(
                "tensor parallelism for this architecture needs NINFER_ENABLE_DISTRIBUTED");
#endif
        }
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

} // namespace

namespace {

// Width > 1 option combinations: one row per option this engine has that touches the sharded lane.
// A row is either validated — a two-rank round has been compared token for token against width 1 —
// or it refuses. Promotion means adding the evidence to a row, never deleting a refusal, and a row
// whose option disappears is deleted with the option: this is a register of what exists, not a
// framework.
//
// The sharded path's own checks ("the collective requires exactly two ranks") guard the collective's
// SHAPE; they say nothing about whether an option's semantics were ever compared at width 2. That is
// the gap this table closes.
struct WidthTwoOption {
    std::string_view option;
    bool (*requested)(const EngineOptions&);
    bool validated;
    // The two-rank round that validated it, or empty while it is unvalidated.
    std::string_view evidence;
    // Why it is unvalidated, and therefore what the refusal is protecting.
    std::string_view reason;
};

constexpr WidthTwoOption kWidthTwoOptions[] = {
    {"--spec mtp",
     [](const EngineOptions& o) { return o.speculative.backend == SpeculativeBackend::Mtp; },
     false, "",
     "the MTP block's projections are Q8 and the Q8 projection table has no rank-sized shape: a "
     "width-2 round now gets past linear_pair (its N was hard-coded to 1024) and throws "
     "'q8 linear: unsupported shape N7168K5120 T1' -- the MTP attention projection's packed rows "
     "at width 2 (14336 halved). The width-2 Q8 set the MTP block needs is N7168K5120 (attention "
     "in), N5120K3072 (attention out), N8704K5120 (mlp gate/up), N5120K8704 (mlp down) and "
     "N124160K5120 (mtp/output_head, shared with text/output_head); none is registered"},
    {"--spec dflash",
     [](const EngineOptions& o) { return o.speculative.backend == SpeculativeBackend::DFlash; },
     false, "",
     "the DFlash drafter has no two-rank round compared against width 1"},
    {"--spec dflash2",
     [](const EngineOptions& o) { return o.speculative.backend == SpeculativeBackend::DFlash2; },
     false, "",
     "the DFlash2 drafter has no two-rank round compared against width 1"},
    {"--causal-scoring",
     [](const EngineOptions& o) { return o.purpose == EnginePurpose::CausalScoring; },
     false, "",
     "the scoring purpose runs a different frame kind and its sharded path has never been "
     "exercised"},
    {"prefix reuse / context cache",
     [](const EngineOptions& o) { return o.context_cache.enabled; },
     false, "",
     "the context cache's slot ownership and reuse accounting are per Program and no two-rank round "
     "has compared them"},
    {"--kv-dtype (any quantized KV)",
     [](const EngineOptions& o) {
         return o.kv_cache != KvCacheStorage::BFloat16 &&
                o.kv_cache != KvCacheStorage::BFloat16KeyValue;
     },
     false, "",
     "the sharded KV mirror has only been exercised with bf16; no two-rank round has compared a "
     "quantized pool against width 1"},
    {"--vision",
     [](const EngineOptions& o) { return o.enable_vision; },
     false, "",
     "the Vision tower is not sharded and no two-rank round has run a vision request"},
};

[[noreturn]] void refuse_unvalidated_width_two(const WidthTwoOption& row) {
    throw std::invalid_argument(
        std::string(row.option) +
        " has no validated two-rank round: " + std::string(row.reason) +
        ". The refusal is removed by a two-rank round compared token for token against width 1; "
        "until then pass a single device with --device");
}

} // namespace

EngineOptions normalize_engine_options(EngineOptions options) {
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.kv_capacity          = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache =
            ContextCacheOptions{.enabled = false, .extra_state_slots = 0, .host_capacity_bytes = 0};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    ContextCacheOptions& cache      = options.context_cache;

    // Tensor parallelism has no Host tier. A rank-1 KV replica is replayed on rank 1's stream, and
    // the Program's host-tier spill and restore call the physical pool directly rather than the
    // mirror funnel — so at width > 1 a host transfer would move rank 0's KV and leave rank 1's
    // replica stale with no error at all. capy reaches the same conclusion by the same route: both
    // Host tiers are zero at tp 2. Refusing the request is better than silently zeroing it, because
    // the caller asked for memory the lane cannot keep coherent.
    if (options.devices.size() > 1) {
        const bool host_tier_requested =
            cache.host_capacity_bytes.value_or(0) != 0 || cache.host_state_slots != 0 ||
            cache.host_kv_capacity_bytes != 0;
        if (host_tier_requested) {
            throw std::invalid_argument(
                "tensor parallelism has no Host KV or state tier: the rank-1 replica is replayed on "
                "rank 1's stream and a host transfer would leave it stale. Clear the Host tier "
                "options (--host-kv-mib, --host-state-slots, host context-cache capacity)");
        }
        // Graph capture is per Program, and each rank has its own Program, so two ranks capture
        // separately. That is only sound while no captured node touches both devices; the moment a
        // captured round contains a cross-rank exchange the two captures must be joint, which the
        // per-Program capture cannot express. Refusing here is deliberate: a subtly wrong capture
        // is silent, a refusal is not. Eager rounds are correct today, so the lane runs without
        // graphs until joint capture lands.
        if (options.use_cuda_graph) {
            throw std::invalid_argument(
                "tensor parallelism cannot capture CUDA graphs yet: each rank captures its own "
                "graph and a round with a cross-rank exchange needs one joint capture. Pass "
                "--no-cuda-graph; the lane runs eager until joint capture lands");
        }
        // Every other option that reaches the sharded lane must be validated before it can run at
        // width > 1. The rows are the register: an unvalidated combination refuses here, in the one
        // place that knows the width, instead of loading and producing plausible wrong numbers
        // under the inference escape.
        for (const auto& row : kWidthTwoOptions) {
            if (row.requested(options) && !row.validated) { refuse_unvalidated_width_two(row); }
        }
    }

    const std::uint32_t concurrency = options.max_concurrency;
    cache.extra_state_slots         = cache.extra_state_slots.value_or(concurrency);
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.extra_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    return options;
}

SessionEpoch next_session_epoch() noexcept {
    static std::atomic<std::uint64_t> next{0};
    return {next.fetch_add(1, std::memory_order_relaxed) + 1};
}

ConstructedModel construct_model(EngineOptions& options) {
    validate_options(options);
    if (options.deployment) {
        validate_deployment(*options.deployment);
        if (options.deployment->ranks.size() == 1)
            options.device = options.deployment->ranks.front().device;
    }
    const auto start = Clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    const models::Architecture architecture = models::resolve_architecture(reader.directory());

    // `EngineOptions::token_logprobs` says a request asking for probabilities against an engine
    // loaded without the capability is *rejected* rather than answered with placeholders. Only the
    // Flash-Next family implements the egress; the others silently returned empty score spans, which
    // is the placeholder the contract forbids. Rejecting at load is the right place: the capability
    // is fixed for the engine's lifetime, so a per-request check would only move the surprise later.
    if (options.token_logprobs && architecture != models::Architecture::Qwen3_8FlashNext) {
        throw std::invalid_argument(
            "this architecture does not report token log-probabilities; load it without "
            "token_logprobs or use a Flash-Next artifact");
    }

    switch (architecture) {
    case models::Architecture::Qwen4Exp: {
#if NINFER_ENABLE_DISTRIBUTED
        if (options.deployment && options.deployment->ranks.size() > 1) {
            auto built = detail::build_distributed_model<models::qwen4_exp::ParallelStartup>(
                reader, options, start);
            ConstructedModel constructed;
            constructed.instance    = std::move(built.instance);
            constructed.load        = std::move(built.load);
            constructed.context_cost = std::move(built.context_cost);
            return constructed;
        }
#endif
        auto built = detail::build_local_model<models::qwen4_exp::Startup>(reader, options, start);
        ConstructedModel constructed;
        constructed.execution    = std::move(built.execution);
        constructed.sharded      = std::move(built.sharded);
        constructed.instance     = std::move(built.instance);
        constructed.load         = std::move(built.load);
        constructed.context_cost = std::move(built.context_cost);
        return constructed;
    }
    case models::Architecture::Qwen3_5:
    case models::Architecture::Qwen3_5Moe: {
        auto built = detail::build_local_model<models::qwen3_5::Startup>(reader, options, start);
        ConstructedModel constructed;
        constructed.execution    = std::move(built.execution);
        // The owner must cross this boundary. `built` is a local, so leaving `sharded` in it frees
        // the mailbox and the partial pair here — during startup — while both Programs keep raw
        // pointers into the group.
        constructed.sharded      = std::move(built.sharded);
        constructed.instance     = std::move(built.instance);
        constructed.load         = std::move(built.load);
        constructed.context_cost = std::move(built.context_cost);
        constructed.qwen3_5_peers = std::move(built.peers);
        return constructed;
    }
    case models::Architecture::Qwen3_8FlashNext:
        return flashnext_runtime::construct_flash_next(options, reader, start);
    }
    throw std::logic_error("architecture has no Engine construction route");
}

} // namespace ninfer::runtime
