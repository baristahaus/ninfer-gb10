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
        case models::Architecture::Qwen3_5Moe:
            throw std::invalid_argument(
                "tensor parallelism is not implemented for this architecture yet; pass a single "
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
    switch (models::resolve_architecture(reader.directory())) {
    case models::Architecture::Qwen4Exp: {
#if NINFER_ENABLE_DISTRIBUTED
        if (options.deployment && options.deployment->ranks.size() > 1) {
            auto built = detail::build_distributed_model<models::qwen4_exp::ParallelStartup>(
                reader, options, start);
            return {std::move(built.instance), std::move(built.load),
                    std::move(built.context_cost)};
        }
#endif
        auto built = detail::build_local_model<models::qwen4_exp::Startup>(reader, options, start);
        return {std::move(built.instance), std::move(built.load), std::move(built.context_cost)};
    }
    case models::Architecture::Qwen3_5:
    case models::Architecture::Qwen3_5Moe: {
        auto built = detail::build_local_model<models::qwen3_5::Startup>(reader, options, start);
        return {std::move(built.instance), std::move(built.load), std::move(built.context_cost)};
    }
    case models::Architecture::Qwen3_8FlashNext:
        return flashnext_runtime::construct_flash_next(options, reader, start);
    }
    throw std::logic_error("architecture has no Engine construction route");
}

} // namespace ninfer::runtime
