#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/session/session.h"
#include "models/qwen3_5/execution/text.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

RankProgramImpl::RankProgramImpl(const execution::Parameters& values, const SequencePlanImpl& plan,
                                 DeviceContext& context, const StartupObserver& observer,
                                 const models::ShardedGroup* sharded)
    : parameters(values), device(context), capacity(plan.capacity), kv_capacity(plan.kv_capacity),
      max_concurrency(plan.max_concurrency), prefill_chunk(plan.prefill_chunk),
      draft_window(plan.draft_window), backend(plan.speculative_backend),
      proposal_head(plan.proposal_head), use_cuda_graph(plan.use_cuda_graph),
      kv_storage(plan.kv_storage), kv_payload_bytes(plan.persistent.kv_payload_bytes),
      workspace_plan(plan.workspace), sharded_(sharded),
      resources(context, plan.persistent, plan.workspace, observer),
      context_counts(plan.causal_scoring ? 0
                                         : std::size_t(values.profile->public_tokens) *
                                               plan.max_concurrency * sizeof(std::int32_t)),
      result_ready(context), drafts_ready(context), compute_retired(context),
      context_ready(context), prefill_timer(context), gpu_timer(context, plan.gpu_timing) {
    if (parameters.profile != plan.profile || parameters.profile->options != plan.features)
        throw std::invalid_argument("Qwen3.5 rank parameters differ from the prepared plan");
    // The per-rank view of the group's collective. The mailbox and the partial pair are owned by
    // whoever builds both ranks — one mailbox for the whole group, because two would never
    // rendezvous — and this Program only derives its own rank from them. The 27B plan does not carry
    // the group width yet, so "width 2 without the bundle" cannot be detected here; the executor
    // that owns the group is the place that must always hand it over.
    if (sharded_ != nullptr) {
        if (sharded_->execution == nullptr || sharded_->mailbox == nullptr) {
            throw std::invalid_argument(
                "Qwen3.5 rank: the sharded group bundle is not fully constructed");
        }
        const ExecutionContext& group = *sharded_->execution;
        if (group.width() != 2) {
            throw std::invalid_argument(
                "Qwen3.5 rank: the sharded group requires an execution group of exactly two ranks");
        }
        // This Program's rank is the group member whose device is this Program's device. A Program
        // that is not in the group is a wiring error, and it must fail here rather than in a
        // reduce: the rank decides which stream the exchange lands on.
        std::vector<int> device_indices;
        device_indices.reserve(static_cast<std::size_t>(group.width()));
        for (int member = 0; member < group.width(); ++member) {
            device_indices.push_back(group.device(member).device().index());
        }
        const int rank = models::sharded_rank_of(device_indices, device.device().index());
        if (rank < 0) {
            throw std::logic_error(
                "Qwen3.5 rank: the Program's device is not a member of its execution group");
        }
        collective_.emplace(execution::ShardedCollective{group, *sharded_->mailbox, rank});
    }
    std::vector<context::ContextPoolStorage> pools{
        {0, &resources.decoder.text_kv.pages(), &resources.decoder.text_kv.tables()}};
    if (auto* cache = resources.backend_kv())
        pools.push_back({1, &cache->pages(), &cache->tables()});
    storage = std::make_unique<context::ContextStorage>(resources.state_images.storage(), pools, 0,
                                                        resources.host_context.span());
    compute_actions =
        std::make_unique<context::ContextMigration>(device, *storage, device.execution_view());
    context_actions = std::make_unique<context::ContextMigration>(device, *storage);
    if (plan.moe) {
        std::vector<const moe::Parameters*> layers;
        for (const auto& block : parameters.text.layers)
            if (const auto* layer = std::get_if<moe::Parameters>(&block.ffn))
                layers.push_back(layer);
        resources.moe = plan.moe->create(layers, parameters.model.expert_banks(), device,
                                         plan.graph_probe ? moe::ResourcePurpose::GraphCalibration
                                                          : moe::ResourcePurpose::Execution);
    }
}

RankProgramImpl::~RankProgramImpl() {
    try {
        drain();
    } catch (...) {
        device.bind_to_current_thread_noexcept();
        (void)cudaStreamSynchronize(device.stream());
        (void)cudaStreamSynchronize(device.transfer_stream());
    }
}

void RankProgramImpl::install_residencies(const ContextWork& work) {
    const auto vocabulary = static_cast<std::size_t>(parameters.profile->public_tokens);
    for (auto id : work.releases) {
        auto& residency = residencies[id.lane.value];
        if (!residency.installed || residency.id != id)
            throw std::logic_error("Qwen3.5 release no longer matches its residency");
        residency.vision.reset();
        residency.vision_input.reset();
        residency.installed = false;
    }
    auto* counts = static_cast<std::int32_t*>(context_counts.data());
    for (const auto& binding : work.bindings) {
        const auto lane = binding.member.residency.lane.value;
        auto& residency = residencies[lane];
        residency.vision.reset();
        residency.vision_input.reset();
        residency.id        = binding.member.residency;
        residency.sequence  = binding.member.sequence;
        residency.installed = true;
        residency.penalties =
            binding.sampling.presence_penalty != 0.0F || binding.sampling.frequency_penalty != 0.0F;
        if (!resources.token_counts.data || !residency.penalties) continue;
        Tensor destination = resources.token_counts.slice(1, static_cast<int>(lane), 1);
        if (binding.committed_generated_tokens.empty()) {
            CUDA_CHECK(cudaMemsetAsync(destination.data, 0, destination.bytes(),
                                       device.transfer_stream()));
            continue;
        }
        auto* row_counts = counts + lane * vocabulary;
        std::fill_n(row_counts, vocabulary, 0);
        for (auto token : binding.committed_generated_tokens) ++row_counts[token];
        CUDA_CHECK(cudaMemcpyAsync(destination.data, row_counts, vocabulary * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, device.transfer_stream()));
    }
}

void RankProgramImpl::start(runtime::ExecutionRequest<ExecutionTypes>&& request) {
    // The lane loads but has never produced a validated round at width 2. Both ranks now take the
    // sharded call sites -- the three column-parallel projections branch on the collective, and
    // both Programs are handed the same ShardedGroup -- but no two-GPU round has been compared
    // against a single-GPU one, and every defect this lane has found produced numbers that looked
    // plausible.
    // So the refusal stands until the tokens match, and it names that reason rather than claiming
    // the loop is unwired.
    if (peer_text_kv != nullptr) {
        throw std::logic_error(
            "the sharded forward has no validated round yet: this lane loads but cannot infer. A "
            "width-2 round has never been compared token for token against a width-1 round, and "
            "a sharded projection that is subtly wrong produces plausible numbers rather than an "
            "error. Remove this refusal only with that comparison passing");
    }
    if (request.ticket.frame.kind == runtime::FrameKind::Context) {
        if (context_operation) throw std::logic_error("Qwen3.5 context frame is occupied");
        context_operation =
            std::make_unique<ContextOperation>(ContextOperation{std::move(request), {}});
        context_operation->last_command = context_operation->request.command;
        context_actions->submit(context_operation->request.work.context);
        install_residencies(std::get<ContextWork>(context_operation->request.work.operation));
        context_ready.record(device.transfer_stream());
    } else {
        if (compute) throw std::logic_error("Qwen3.5 compute frame is occupied");
        gpu_timer.begin();
        compute = std::make_unique<ComputeOperation>(std::move(request));
        compute_actions->submit(compute->request.work.context);
        initialize_inputs(*compute);
    }
    progress();
}

void RankProgramImpl::supply(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                             runtime::DecisionInput&& input) {
    auto& operation             = require_compute(ticket);
    operation.last_command      = command;
    operation.decision          = std::move(input);
    operation.decision_supplied = true;
    progress();
}

void RankProgramImpl::commit(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                             Commit&& value) {
    auto& operation = require_compute(ticket);
    gpu_timer.resolve();
    operation.last_command = command;
    operation.commit       = std::move(value);
    operation.stage        = ComputeStage::Commit;
    progress();
}

void RankProgramImpl::discard(runtime::ExecutionTicket ticket, runtime::CommandSequence command) {
    auto& operation = require_compute(ticket);
    gpu_timer.resolve();
    operation.last_command = command;
    operation.cancelled    = true;
    progress();
}

std::optional<runtime::ExecutionEvent<ExecutionTypes>> RankProgramImpl::poll() {
    progress();
    if (events.empty()) return std::nullopt;
    auto event = std::move(events.front());
    events.pop_front();
    return event;
}

void RankProgramImpl::drain() {
    std::exception_ptr failure;
    const auto attempt = [&](auto&& action) {
        try {
            action();
        } catch (...) {
            if (!failure) failure = std::current_exception();
        }
    };
    attempt([&] { device.bind_to_current_thread(); });
    if (compute && compute->text) attempt([&] { compute->text->cancel(); });
    if (graphs) attempt([&] { graphs->cancel(); });
    if (resources.moe) {
        attempt([&] { resources.moe->cancel(); });
        attempt([&] { resources.moe->synchronize(); });
    }
    attempt([&] { compute_actions->drain(); });
    attempt([&] { context_actions->drain(); });
    const auto wait_stream = [](cudaStream_t stream) {
        const auto status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("Qwen3.5 stream drain: ") +
                                     cudaGetErrorString(status));
    };
    attempt([&] { wait_stream(device.transfer_stream()); });
    attempt([&] { wait_stream(device.stream()); });
    compute.reset();
    context_operation.reset();
    events.clear();
    for (auto& residency : residencies) {
        residency.vision.reset();
        residency.vision_input.reset();
    }
    if (failure) std::rethrow_exception(failure);
}


} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5 {

void attach_rank_mirrors(RankProgram& origin, RankProgram& peer, DeviceContext& peer_device,
                         PreparedSession* session) {
    if (&origin == &peer) {
        throw std::invalid_argument("rank mirror attaches a Program to itself");
    }
    // The route to the pools is the ContextStorage each Program owns: pool 0 is the text KV.
    auto& origin_pages  = *origin.impl_->storage->pool(0).pages;
    auto& origin_tables = *origin.impl_->storage->pool(0).tables;
    auto& peer_pages    = *peer.impl_->storage->pool(0).pages;
    auto& peer_tables   = *peer.impl_->storage->pool(0).tables;
    if (origin_pages.capacity_pages() != peer_pages.capacity_pages()) {
        throw std::invalid_argument(
            "tensor-parallel ranks resolved different KV capacities (" +
            std::to_string(origin_pages.capacity_pages()) + " vs " +
            std::to_string(peer_pages.capacity_pages()) +
            " pages): the rank-1 mirror addresses rank 0's pages by index, so both ranks must "
            "resolve the same capacity. The box is asymmetric or the options differ per rank");
    }
    const tp2::MirrorTarget where{.device          = &peer_device,
                                  .stream          = peer_device.stream(),
                                  .transfer_stream = peer_device.transfer_stream()};
    origin.impl_->peer_text_kv = std::make_unique<tp2::MirroredKVPagePool>(
        origin_pages.mirror_origin(), peer_pages.mirror_origin(), where);
    origin.impl_->peer_text_kv_tables = std::make_unique<tp2::MirroredKVExecutionTables>(
        *origin.impl_->peer_text_kv, origin_tables.table_origin(), peer_tables.table_origin());
    origin_pages.attach_mirror(*origin.impl_->peer_text_kv);
    origin_tables.attach_mirror(*origin.impl_->peer_text_kv_tables);
    // The row leases live on the control side, so the same mirror is attached to the session's
    // directory. Both attaches happen here, before either Program has run a round.
    if (session != nullptr) { session->attach_kv_mirror(*origin.impl_->peer_text_kv_tables); }
}

} // namespace ninfer::models::qwen3_5
