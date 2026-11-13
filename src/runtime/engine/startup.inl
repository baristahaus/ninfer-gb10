#pragma once

#include "runtime/engine/instance.h"
#include "runtime/executor/local_executor.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/startup_memory.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "models/registry.h"
#include "models/common/startup.h"
#include "models/common/sharded_group.h"
#include "core/startup.h"
#include "core/platform.h"

#include <algorithm>
#include <chrono>
#include <charconv>
#include <cstdlib>
#include <limits>
#include <cerrno>
#include <system_error>
#include <set>

namespace ninfer::runtime::detail {


template <class Plan>
struct CalibratedPlan {
    Plan startup;
    GraphMemorySummary graph;
};

template <class Setup>
std::optional<CalibratedPlan<typename Setup::ExecutionPlan>>
calibrate_plan(const Setup& setup, RankModelInstance<typename Setup::Types>& instance,
               DeviceContext& device, const EngineOptions& options,
               const StartupMemoryBudget& budget, typename Setup::ExecutionPlan candidate,
               std::vector<typename Setup::Calibration>& measured, std::size_t minimum_rebuild = 0,
               std::size_t control_rebuild = 0) {
    for (;;) {
        auto probe_options          = options;
        probe_options.prefill_chunk = candidate.chunk;
        const auto key              = setup.graph_plan(probe_options, candidate.preparation);
        auto found                  = std::find_if(measured.begin(), measured.end(),
                                                   [&](const auto& item) { return item.plan == key; });
        if (found == measured.end()) {
            measured.push_back(
                setup.calibrate(instance.parameters, device, probe_options, candidate.preparation));
            found = std::prev(measured.end());
        }
        const auto graph_bytes = std::max(found->rebuild_bytes, minimum_rebuild);
        const GraphMemorySummary graph{
            .rebuild_estimate_bytes      = graph_bytes,
            .calibration_increment_bytes = found->preparation_increment_bytes,
            .calibration_retained_bytes  = found->retained_increment_bytes,
        };
        auto available = budget.available(device_memory_snapshot(device.device()));
        if (control_rebuild > available) return std::nullopt;
        available -= control_rebuild;
        if (graph_bytes > available) return std::nullopt;
        auto next = setup.plan(device, options, candidate.placement, available - graph_bytes);
        if (!next) return std::nullopt;
        const auto next_key = setup.graph_plan(probe_options, next->preparation);
        if (next_key == key) return CalibratedPlan{std::move(*next), graph};
        const bool revisited = std::any_of(measured.begin(), measured.end(),
                                           [&](const auto& item) { return item.plan == next_key; });
        if (revisited && candidate.sequence.device_reservation_bytes() <= available - graph_bytes) {
            // At a variant boundary choose the feasible measured side, then fill that interval.
            auto low  = candidate.sequence.device_reservation_bytes();
            auto high = available - graph_bytes;
            while (low < high) {
                const auto middle = low + (high - low + 1) / 2;
                auto trial        = setup.plan(device, options, candidate.placement, middle);
                if (trial && setup.graph_plan(probe_options, trial->preparation) == key) {
                    low       = middle;
                    candidate = std::move(*trial);
                } else
                    high = middle - 1;
            }
            return CalibratedPlan{std::move(candidate), graph};
        }
        candidate = std::move(*next);
    }
}

template <class Setup>
std::optional<CalibratedPlan<typename Setup::ExecutionPlan>>
prepare_program(const Setup& setup, RankModelInstance<typename Setup::Types>& instance,
                std::optional<typename Setup::Types::PreparedSession>& prepared_session,
                StartupMemorySummary& memory, DeviceContext& device, EngineOptions& options,
                const StartupMemoryBudget& budget,
                CalibratedPlan<typename Setup::ExecutionPlan> plan,
                std::vector<typename Setup::Calibration>& measured) {
    for (unsigned attempts = 0; attempts < 2; ++attempts) {
        auto& startup             = plan.startup;
        const auto explicit_bytes = startup.sequence.device_reservation_bytes();
        const auto host_bytes     = startup.sequence.host_capacity_bytes();
        const auto retained =
            budget.retained(device_memory_snapshot(device.device()),
                            instance.model->storage_stats().device_capacity_bytes);
        if (explicit_bytes != startup.capacity.runtime_reservation_bytes ||
            startup.sequence.kv_capacity() != startup.capacity.resolved_tokens)
            throw std::logic_error("resolved capacity differs from the finalized Program plan");
        std::optional<DeviceMemorySnapshot> ready;
        try {
            StartupPhaseScope phase(options.startup_observer, StartupPhase::ProgramInitialize);
            auto control_plan = plan_session(startup.sequence);
            instance.program = create_rank_program(instance.parameters, std::move(startup.sequence),
                                                   device, options.startup_observer,
                                                   instance.sharded);
            auto transfers   = instance.program->context_transfer_profiles();
            if (transfers.size() != control_plan.context.pools.size()) {
                throw std::logic_error("rank context pools differ from the session plan");
            }
            for (std::size_t i = 0; i < transfers.size(); ++i) {
                control_plan.context.pools[i].transfer = std::move(transfers[i]);
            }
            prepared_session.emplace(prepare_session(setup.definition(), std::move(control_plan)));
            device.synchronize();
            ready = device_memory_snapshot(device.device());
            if (budget.fits(*ready)) {
                options.prefill_chunk                                  = startup.chunk;
                options.context_cache.host_capacity_bytes              = host_bytes;
                memory.kv_capacity_resolution                          = startup.capacity;
                memory.kv_capacity_resolution.automatic_headroom_bytes = budget.headroom();
                memory.device_budget_bytes                             = budget.limit();
                memory.startup_retained_bytes                          = retained;
                memory.cuda_graph = instance.program->graph_memory_observation();
                if (memory.cuda_graph) {
                    plan.graph.preparation_increment_bytes =
                        memory.cuda_graph->preparation_increment_bytes;
                    memory.cuda_graph = plan.graph;
                }
                phase.complete();
                return plan;
            }
        } catch (const DeviceAllocationError&) {
            // A failed preparation invalidates its observation; the next probe measures it again.
            auto probe_options          = options;
            probe_options.prefill_chunk = startup.chunk;
            const auto key              = setup.graph_plan(probe_options, startup.preparation);
            std::erase_if(measured, [&](const auto& entry) { return entry.plan == key; });
        }
        prepared_session.reset();
        const auto after_control_cleanup = device_memory_snapshot(device.device());
        instance.program.reset();
        device.synchronize();
        if (attempts == 1) return std::nullopt;
        const auto after_cleanup = device_memory_snapshot(device.device());
        const auto available     = budget.available(after_cleanup);
        const auto recovered = ready ? device_memory_growth(after_cleanup, after_control_cleanup) -
                                           static_cast<std::int64_t>(explicit_bytes)
                                     : 0;
        const auto control_rebuild =
            ready ? static_cast<std::size_t>(std::max<std::int64_t>(
                        0, device_memory_growth(after_control_cleanup, *ready)))
                  : 0;
        const auto reserve =
            std::max(plan.graph.rebuild_estimate_bytes,
                     static_cast<std::size_t>(std::max<std::int64_t>(0, recovered)));
        if (reserve > available || control_rebuild > available - reserve) return std::nullopt;
        auto candidate =
            setup.plan(device, options, startup.placement, available - reserve - control_rebuild);
        if (!candidate) return std::nullopt;
        auto corrected = calibrate_plan(setup, instance, device, options, budget,
                                        std::move(*candidate), measured, reserve, control_rebuild);
        if (!corrected) return std::nullopt;
        plan = std::move(*corrected);
    }
    return std::nullopt;
}

template <class Types>
struct BuiltModel {
    // The group is declared first so it is destroyed last: every worker borrows its rank's device.
    std::unique_ptr<ExecutionContext> execution;
    // The two-GPU lane's group-scoped resources: the ONE mailbox and the ONE partial pair both
    // Programs share. Declared after the devices and before the Programs, so it is destroyed AFTER
    // the Programs and BEFORE the devices — the mailbox's pinned slab, its per-device words and the
    // partial addresses are baked into every captured exchange, so nothing that uses them may
    // outlive them. Null at width 1, where nothing is constructed.
    std::unique_ptr<models::ShardedGroupOwner> sharded;
    std::unique_ptr<InferenceInstance<Types>> instance;
    LoadSummary load;
    ContextMachineCostModel context_cost;
    // Ranks 1..width-1 of the group: their own materialized shards, read from their own
    // definitions. Empty at width 1. The peer Program and its mirrors are the next step; this is
    // the storage they will be built on.
    std::vector<std::unique_ptr<RankModelInstance<Types>>> peers;
};

template <class Setup>
std::optional<CalibratedPlan<typename Setup::ExecutionPlan>>
initialize_program(const Setup& setup, RankModelInstance<typename Setup::Types>& instance,
                   std::optional<typename Setup::Types::PreparedSession>& prepared_session,
                   StartupMemorySummary& memory, DeviceContext& device, EngineOptions& options,
                   const StartupMemoryBudget& budget, typename Setup::ExecutionPlan candidate) {
    std::vector<typename Setup::Calibration> measured;
    StartupPhaseScope phase(options.startup_observer, StartupPhase::TargetFinalize);
    auto calibrated =
        calibrate_plan(setup, instance, device, options, budget, std::move(candidate), measured);
    if (!calibrated) return {};
    phase.complete();
    return prepare_program(setup, instance, prepared_session, memory, device, options, budget,
                           std::move(*calibrated), measured);
}

template <class Setup>
BuiltModel<typename Setup::Types> build_local_model(const artifact::Reader& reader,
                                                    EngineOptions& options,
                                                    std::chrono::steady_clock::time_point start) {
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    std::unique_ptr<artifact::Reader> rank_reader;
    if (options.deployment && options.deployment->ranks.size() == 1) {
        const auto& node = options.deployment->nodes.front();
        for (const auto& [key, value] : node.environment)
            if (::setenv(key.c_str(), value.c_str(), 1))
                throw std::system_error(errno, std::generic_category(),
                                        "set local rank environment");
        rank_reader = std::make_unique<artifact::Reader>(node.artifact);
    }
    Setup setup(reader, rank_reader ? *rank_reader : reader, options, Device::cuda(options.device));
    // The engine opens a group, not one device: empty `devices` is the single `device` field, so
    // the width-1 lane gets a one-element group and behaves exactly as it did with a bare device.
    std::unique_ptr<ExecutionContext> execution;
    {
        StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaInitialize);
        execution = std::make_unique<ExecutionContext>(
            options.devices.empty() ? std::vector<int>{options.device} : options.devices);
        phase.complete();
    }
    auto& device       = execution->primary();
    const auto initial = device_memory_snapshot(device.device());
    const auto limit   = device_budget(initial.available_bytes());
    const auto headroom =
        std::max(kDefaultKvCapacityHeadroomBytes, options.kv_capacity.automatic_headroom_bytes);
    const auto host_reservation = (NINFER_SHARED_PHYSICAL_MEMORY != 0) && options.enable_vision
                                      ? options.media_live_bytes
                                      : 0;
    StartupMemoryBudget budget(initial, limit, headroom, host_reservation);
    const auto choices   = setup.placements();
    std::size_t selected = 0;
    const auto choose    = [&](std::size_t available) {
        for (; selected < choices.size(); ++selected) {
            const auto& choice = choices[selected];
            if (choice.device_bytes > available) continue;
            auto plan =
                setup.plan(device, options, choice.placement, available - choice.device_bytes);
            if (plan) return plan;
        }
        return std::optional<typename Setup::ExecutionPlan>{};
    };
    const std::size_t budget_available = budget.available(device_memory_snapshot(device.device()));
    auto candidate                     = choose(budget_available);
    if (!candidate) {
        // Say how much is needed and how much there is. "It does not fit" leaves the reader to guess
        // whether the model is too large, the context too long or the headroom too generous; the two
        // numbers make the next move obvious, and on the two-GPU lane the per-rank footprint is the
        // number that matters.
        std::string detail = "Device budget cannot hold model weights and required execution "
                             "resources: " +
                             std::to_string(budget_available) + " bytes available";
        if (!choices.empty()) {
            std::size_t smallest = choices.front().device_bytes;
            for (const auto& choice : choices) {
                if (choice.device_bytes < smallest) { smallest = choice.device_bytes; }
            }
            detail += ", the smallest placement needs " + std::to_string(smallest) + " bytes";
        }
        throw std::invalid_argument(detail);
    }
    binding.complete();
    using Types = typename Setup::Types;
    // Declared before the instance so it is destroyed AFTER the Program that borrows it.
    std::unique_ptr<models::ShardedGroupOwner> sharded_owner;
    std::unique_ptr<RankModelInstance<Types>> instance;
    std::optional<typename Types::PreparedSession> prepared_session;
    StartupMemorySummary memory;
    std::optional<CalibratedPlan<typename Setup::ExecutionPlan>> calibrated;
    // The one capacity the whole group runs at, decided once below and published to every rank.
    // Empty at width 1, where rank 0 is the group.
    std::optional<std::uint32_t> published_group_tokens;
    std::size_t available_after_weights = 0;
    auto frontend                       = [&] {
        StartupPhaseScope phase(options.startup_observer, StartupPhase::FrontendInitialize);
        auto result = setup.frontend(options);
        phase.complete();
        return result;
    }();
    const auto& definition_stats = setup.definition()->resource_stats();
    LoadSummary summary;
    summary.artifact_bytes_read = definition_stats.read_bytes;
    for (;;) {
        try {
            auto model =
                setup.materialize(device, choices[selected].placement, options.startup_observer);
            device.synchronize();
            available_after_weights = device_memory_snapshot(device.device()).available_bytes();
            const auto& stats       = model->storage_stats();
            summary.upload_seconds += stats.upload_seconds;
            summary.artifact_bytes_read += stats.read_bytes;
            summary.host_to_device_bytes += stats.h2d_bytes;
            summary.max_rank_staging_bytes =
                std::max(summary.max_rank_staging_bytes, stats.peak_staging_bytes);
            instance = std::make_unique<RankModelInstance<Types>>(std::move(model), setup.profile);
            // One capacity for the whole group, decided once and published to both ranks. Rank 0's
            // plan resolved against its own budget; the peer's curve is needed before its Program
            // exists, and the group bottleneck is the smallest count either rank can hold. Without
            // this the lane resolves two capacities and refuses at the rank-1 re-plan, turning a
            // legitimate asymmetric box into an error instead of running it at the common capacity.
            // The group outlives the instance (BuiltModel declares it first). The two-GPU lane's
            // shared mailbox and partial pair are owned by the executor that builds both ranks and
            // are installed here; null until then, which the Program refuses at width > 1.
            // The two-GPU lane's group-scoped resources, built once per attempt from this plan's
            // geometry: one mailbox for the whole group, one full-width BF16 partial per rank. Width
            // 1 constructs nothing, so the single-device lane pays no pinned memory and no device
            // words. Rebuilt only when the plan is retried at a smaller capacity; the old owner is
            // released after the instance (see the retry below), never before it.
            //
            // The rows are the one family-specific extent (the residual stream's width); the token
            // capacity is the resolved plan's, which every family has. A family without the hook has
            // no two-GPU lane and never reaches this branch at width 2 (its per-rank materialization
            // is refused below).
            if (execution->width() > 1) {
                auto group_policy = options.kv_capacity;
                // budget.available() already removes the headroom, so the group decision must not
                // subtract it a second time.
                group_policy.automatic_headroom_bytes = 0;
                auto& peer_device       = execution->device(1);
                const auto peer_budget =
                    budget.available(device_memory_snapshot(peer_device.device()));
                const std::array<RankCapacityOffer, 2> offers{
                    RankCapacityOffer{candidate->curve,
                                      budget_available - choices[selected].device_bytes},
                    RankCapacityOffer{
                        setup.capacity_curve(peer_device, options, choices[selected].placement),
                        peer_budget - choices[selected].device_bytes}};
                const auto capacities = resolve_group_kv_capacity(group_policy, offers);
                published_group_tokens = capacities.front().resolved_tokens;
                if (*published_group_tokens != candidate->capacity.resolved_tokens) {
                    // The peer's budget is the bottleneck: rank 0 is re-planned at the group's
                    // count, never the other way round.
                    auto replanned =
                        setup.plan(device, options, choices[selected].placement,
                                   budget_available - choices[selected].device_bytes,
                                   published_group_tokens);
                    if (!replanned) {
                        throw std::invalid_argument(
                            "rank 0 cannot hold the group's KV capacity of " +
                            std::to_string(*published_group_tokens) +
                            " tokens, which the peer's budget lowered it to");
                    }
                    candidate = std::move(replanned);
                }
            }
            if (execution->width() > 1) {
                if constexpr (requires(Setup& s) { s.sharded_partial_rows(); }) {
                    const auto rows = setup.sharded_partial_rows();
                    if (rows) {
                        // The logits pair is the second group-scoped pair: one full-vocabulary BF16
                        // buffer per rank, because the head is row-sharded and the gather writes
                        // both. Its rows are the vocabulary the frame's logits carry; its columns
                        // are the widest round the plan can present — (draft_window + 1) * batch for
                        // a decode/verify round, and one score tile for a causal-scoring lane, whose
                        // frame is one column wide but projects a tile of targets per call.
                        const std::int32_t vocabulary =
                            static_cast<std::int32_t>(setup.profile->config.text.vocab_size);
                        const std::uint32_t width =
                            options.speculative.backend == SpeculativeBackend::None
                                ? 1U
                                : options.speculative.draft_tokens + 1U;
                        const std::int32_t logits_columns = static_cast<std::int32_t>(
                            std::max<std::uint32_t>(
                                width * candidate->sequence.max_concurrency(),
                                options.purpose == EnginePurpose::CausalScoring
                                    ? static_cast<std::uint32_t>(models::kShardedScoreTile)
                                    : 0U));
                        sharded_owner = std::make_unique<models::ShardedGroupOwner>(
                            *execution, *rows,
                            static_cast<std::int32_t>(candidate->capacity.resolved_tokens),
                            vocabulary, logits_columns);
                        instance->sharded = &sharded_owner->group();
                    }
                }
            }
            // The refusal lives here, in the one place that knows the width: a two-GPU lane without
            // the bundle would otherwise run the unsharded projections through sharded weights,
            // which produces plausible wrong numbers rather than an error.
            if (execution->width() > 1 && instance->sharded == nullptr) {
                throw std::logic_error("two-GPU lane: the sharded group bundle was not built");
            }
            calibrated = initialize_program(setup, *instance, prepared_session, memory, device,
                                            options, budget, std::move(*candidate));
        } catch (const DeviceAllocationError&) {
            if (selected + 1 == choices.size()) throw;
            calibrated.reset();
        }
        if (calibrated) break;
        // The Program borrows the owner, so it is released first; the owner is rebuilt from the next
        // attempt's capacity.
        instance.reset();
        sharded_owner.reset();
        device.synchronize();
        if (++selected == choices.size())
            throw std::invalid_argument(
                "Device budget cannot hold the calibrated runtime and requested KV capacity");
        candidate = choose(budget.available(device_memory_snapshot(device.device())));
        if (!candidate)
            throw std::invalid_argument("Device budget cannot hold model execution resources");
    }
    memory.kv_capacity_resolution.available_after_weights_bytes = available_after_weights;
    memory.kv_capacity_resolution.available_after_startup_bytes =
        device_memory_snapshot(device.device()).available_bytes();
    const auto signature =
        setup.prefill_signature(calibrated->startup.preparation, options.prefill_chunk);
    auto context_cost = resolve_context_machine_cost(
        {.hardware_class = context_cost_hardware_class(
             device.properties().name, device.properties().major, device.properties().minor),
         .prefill_signature = signature},
        options.context_cost.preset_path);
    const auto& stats         = instance->model->storage_stats();
    summary.architecture      = models::architecture_name(setup.architecture());
    summary.model_name        = setup.definition()->info().name;
    summary.prefill_signature = signature;
    std::set<std::string> formats;
    for (const auto& weight : setup.definition()->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    summary.device_object_count = stats.device_object_count;
    summary.host_object_count   = stats.host_object_count + definition_stats.host_object_count;
    summary.context_cost        = std::move(context_cost.summary);
    summary.cuda_sync_mode      = device.sync_mode();
    const auto& capacity        = memory.kv_capacity_resolution;
    memory.rank_budgets.push_back(
        {memory.device_budget_bytes, capacity.runtime_reservation_bytes,
         capacity.minimum_runtime_reservation_bytes,
         capacity.bytes_per_additional_main_page_group, capacity.available_after_weights_bytes,
         capacity.available_after_startup_bytes, capacity.automatic_headroom_bytes,
         capacity.planned_slack_bytes, memory.startup_retained_bytes, memory.cuda_graph});
    // Every other rank materializes its own bindings on its own device. Rank 0's instance above is
    // the local one; the peers are the storage the rank-1 Program and its mirrors will be built on.
    std::vector<std::unique_ptr<RankModelInstance<Types>>> peers;
    if constexpr (requires(Setup& s, DeviceContext& d, const StartupObserver& o) {
                      s.materialize(d, models::WeightPlacement::Resident, o, 1);
                  }) {
        for (int rank = 1; rank < execution->width(); ++rank) {
            auto shard = setup.materialize(execution->device(rank), choices[selected].placement,
                                           options.startup_observer, rank);
            peers.push_back(
                std::make_unique<RankModelInstance<Types>>(std::move(shard), setup.profile));
            // The same bundle as rank 0: one mailbox for the group and one partial per rank, shared
            // by both Programs. A per-Program bundle would never rendezvous.
            peers.back()->sharded = instance->sharded;
        }
    } else if (execution->width() > 1) {
        throw std::invalid_argument("this model's local route has no per-rank materialization");
    }
    // The executor drives every rank of the group in rank order. Rank 0 is always present, so a
    // width-1 group is the one-element case of the same loop and takes exactly the path it took
    // before the loop existed.
    std::vector<std::unique_ptr<RankWorker<Types>>> workers;
    std::unique_ptr<RankWorker<Types>> peer_worker;
    if (execution->width() > 1) {
        if constexpr (requires(Setup& s, DeviceContext& d, const StartupObserver& o) {
                          s.materialize(d, models::WeightPlacement::Resident, o, 1);
                      }) {
            // Rank 1's Program is planned at the capacity the group decided above: it never
            // resolves a capacity of its own, because the mirror addresses rank 0's pages by index
            // and two decisions are two answers.
            if (peers.size() != 1) {
                throw std::invalid_argument("a two-rank lane needs exactly one peer instance");
            }
            if (!published_group_tokens) {
                throw std::logic_error("two-GPU lane: the group capacity was never published");
            }
            auto& peer_device = execution->device(1);
            const auto peer_available =
                budget.available(device_memory_snapshot(peer_device.device()));
            auto peer_plan = setup.plan(peer_device, options, choices[selected].placement,
                                        peer_available, published_group_tokens);
            if (!peer_plan) {
                throw std::invalid_argument(
                    "rank 1 cannot hold the model weights and required execution resources at the "
                    "group's KV capacity of " +
                    std::to_string(*published_group_tokens) + " tokens");
            }
            // An invariant check, not a second decision: both ranks must report the published
            // capacity, so a future re-plan that quietly resolves its own number fails here.
            if (peer_plan->capacity.resolved_tokens != *published_group_tokens ||
                calibrated->startup.capacity.resolved_tokens != *published_group_tokens) {
                throw std::logic_error(
                    "rank capacity escaped the group decision: rank 0 has " +
                    std::to_string(calibrated->startup.capacity.resolved_tokens) +
                    " tokens, rank 1 has " +
                    std::to_string(peer_plan->capacity.resolved_tokens) + ", the group published " +
                    std::to_string(*published_group_tokens));
            }
            // The startup record describes every rank the executor drives, and the per-rank report
            // indexes `startup.ranks` by the description's rank index. Rank 1 materializes on its
            // own device and has its own budget, so it needs an entry or the report reads past the
            // end of a one-element vector.
            memory.rank_budgets.push_back(
                {peer_available,
                 peer_plan->capacity.runtime_reservation_bytes,
                 peer_plan->capacity.minimum_runtime_reservation_bytes,
                 peer_plan->capacity.bytes_per_additional_main_page_group,
                 peer_plan->capacity.available_after_weights_bytes,
                 peer_plan->capacity.available_after_startup_bytes,
                 peer_plan->capacity.automatic_headroom_bytes,
                 peer_plan->capacity.planned_slack_bytes,
                 memory.startup_retained_bytes,
                 memory.cuda_graph});
            peers.front()->program = create_rank_program(
                peers.front()->parameters, std::move(peer_plan->sequence), peer_device,
                options.startup_observer, peers.front()->sharded);
            // Both Programs exist and neither has run a round: the mirror contract's window.
            attach_rank_mirrors(*instance->program, *peers.front()->program, peer_device,
                                &*prepared_session);
            peer_worker =
                std::make_unique<RankWorker<Types>>(peer_device, std::move(peers.front()));
        } else {
            throw std::invalid_argument("this model's local route has no per-rank Program");
        }
    }
    // Rank 0 first: the executor's authority, its frame pool and the returned event all come from
    // `workers_.front()`, and the peers follow in rank order.
    workers.push_back(std::make_unique<RankWorker<Types>>(device, std::move(instance)));
    if (peer_worker) { workers.push_back(std::move(peer_worker)); }
    // PHASE MARKER at a point this lane is known to reach: the executor exists by construction (the
    // counter and the give-up line both prove it runs). Placed HERE rather than in a startup
    // function the lane may not use -- the previous placement in construct_flash_next never fired,
    // which is how the launch counter's first bucket stayed labelled startup+round0. Everything
    // before this point is startup; everything after is round traffic.
    if (&ninfer_launch_phase_marker != nullptr) { ninfer_launch_phase_marker("executor_created"); }
    auto executor =
        std::make_unique<LocalExecutor<Types>>(std::move(workers), next_session_epoch());
    auto inference = std::make_unique<InferenceInstance<Types>>(
        setup.definition(), std::move(frontend), std::move(executor), std::move(*prepared_session),
        std::move(memory));
    return {std::move(execution), std::move(sharded_owner), std::move(inference),
            std::move(summary), std::move(context_cost.model), std::move(peers)};
}

} // namespace ninfer::runtime::detail
