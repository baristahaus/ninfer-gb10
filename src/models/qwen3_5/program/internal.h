#pragma once

#include "runtime/observability/gpu_timing.h"

#include "ninfer/ops/kv_cache_append.h"

#include "core/tp2/kv_mirror.h"
#include "models/common/context/transfer.h"
#include "models/common/qwen/vision/prefill.h"
#include "models/qwen3_5/program/program.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/resources.h"
#include "models/qwen3_5/program/graphs.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/execution/sharded.h"
#include "ninfer/ops/peer_mailbox.h"

#include <array>
#include <chrono>
#include <deque>
#include <memory>
#include <optional>

namespace ninfer::models::qwen3_5::execution {
class TextRun;
}

namespace ninfer::models::qwen3_5::detail {

struct Residency {
    runtime::ResidencyId id;
    runtime::SequenceId sequence;
    bool installed = false;
    bool penalties = false;
    std::shared_ptr<const qwen::VisionExecutionInput> vision_input;
    std::unique_ptr<qwen::VisionPrefillExecution> vision;
};

enum class ComputeStage : std::uint8_t {
    Inputs,
    Forward,
    History,
    AwaitDecision,
    Readback,
    AwaitCommit,
    Commit,
    Retiring,
};

struct ComputeOperation {
    runtime::ExecutionRequest<ExecutionTypes> request;
    runtime::CommandSequence last_command;
    std::vector<ExecutionRow> rows;
    std::vector<SamplingInput> sampling;
    std::uint32_t width          = 0;
    std::uint32_t batch          = 0;
    std::uint32_t drafts         = 0;
    std::uint32_t forward_row    = 0;
    std::uint32_t forward_offset = 0;
    std::uint32_t score_cursor   = 0;
    std::uint32_t score_count    = 0;
    double vision_before         = 0.0;
    ComputeStage stage           = ComputeStage::Inputs;
    std::uint32_t decision_stage = 0;
    bool decision_requested      = false;
    bool decision_supplied       = false;
    bool result_published        = false;
    bool cancelled               = false;
    bool speculative             = false;
    bool row_finished            = false;
    bool graph_started           = false;
    bool fused_decision          = false;
    bool projected               = false;
    std::array<std::uint32_t, kMaximumConcurrency> dead_positions{};
    std::optional<runtime::DecisionInput> decision;
    std::optional<Commit> commit;
    std::optional<Result> result;
    std::vector<float> scores;
    GpuExecutionObservation gpu;
    std::optional<Tensor> input_embeddings;
    std::unique_ptr<execution::TextRun> text;

    ComputeOperation(runtime::ExecutionRequest<ExecutionTypes>&&);
    ~ComputeOperation();
};

struct ContextOperation {
    runtime::ExecutionRequest<ExecutionTypes> request;
    runtime::CommandSequence last_command;
};

// The sharded-lane refusal, as a pure function so it can be tested without constructing a whole
// sharded program. It returns the refusal message when the peer mirrors are present (the flag the
// round entry refuses on) and an empty string otherwise. RankProgramImpl::progress() calls it with
// `peer_text_kv != nullptr`; the test calls it with both values. It does NOT cover the attach path
// that makes the mirrors present -- that is the fixture this test deliberately avoids.
[[nodiscard]] std::string sharded_forward_refusal(bool has_peer_mirrors);

class RankProgramImpl {
public:
    // `sharded` is the group-scoped resource bundle of a two-GPU lane — the ONE mailbox and the
    // ONE partial pair the whole group shares — borrowed from its owner. The Program builds its
    // per-rank collective from it here, before any graph capture, and refuses to run at width > 1
    // without it: a mailbox per Program would never rendezvous.
    RankProgramImpl(const execution::Parameters&, const SequencePlanImpl&, DeviceContext&,
                    const StartupObserver&, const models::ShardedGroup* sharded = nullptr);
    ~RankProgramImpl();

    void validate_work(const Work&, runtime::FrameKind) const;
    void validate_decision(runtime::ExecutionTicket, const runtime::DecisionInput&) const;
    void prepare_commit(runtime::ExecutionTicket, Commit&) const;
    // Two-phase dispatch. `start_frame` enqueues the frame, its actions and its inputs and returns
    // without running the stage machine; `progress` then advances that machine until it must wait.
    // `start` is the single-rank entry point (WorkerLoop): start_frame + progress.
    void start_frame(runtime::ExecutionRequest<ExecutionTypes>&&);
    void start(runtime::ExecutionRequest<ExecutionTypes>&&);
    void supply(runtime::ExecutionTicket, runtime::CommandSequence, runtime::DecisionInput&&);
    void commit(runtime::ExecutionTicket, runtime::CommandSequence, Commit&&);
    void discard(runtime::ExecutionTicket, runtime::CommandSequence);
    [[nodiscard]] std::optional<runtime::ExecutionEvent<ExecutionTypes>> poll();
    void drain();
    [[nodiscard]] RankMemoryUsage observe_resources() const;
    [[nodiscard]] RankExecutionObservation retirement_observation() const noexcept;
    void reset_measurement_peaks() noexcept;

    void validate_row(const ExecutionRow&) const;
    void install_residencies(const ContextWork&);
    void initialize_inputs(ComputeOperation&);
    void initialize_decode(ComputeOperation&, const DecodeWork&);
    void initialize_forward(ComputeOperation&, const ForwardWork&);
    void initialize_score(ComputeOperation&, const ScoreWork&);
    void upload_input(std::uint32_t width, std::uint32_t batch);
    void upload_positions(std::uint32_t width, std::uint32_t batch);
    void upload_ingress();
    void begin_target(ComputeOperation&);
    bool advance_target(ComputeOperation&);
    void begin_draft(ComputeOperation&);
    void begin_mtp_draft(ComputeOperation&);
    void prepare_mtp_draft(ComputeOperation&);
    void run_mtp_draft(std::uint32_t batch, std::uint32_t steps, std::uint32_t frontier);
    void run_dflash_draft(std::uint32_t batch, std::uint32_t minimum, std::uint32_t maximum);
    void project_decode(std::uint32_t batch, std::uint32_t width);
    void decide_decode(std::uint32_t batch, std::uint32_t width);
    void readback_decode();
    void initialize_graphs();
    void release_graphs();
    void append_draft_history(ComputeOperation&);
    void append_draft_history(std::uint32_t batch, ops::KVCacheAppendPrefixExecutionEnvelope);
    void commit_draft_history(ComputeOperation&);
    void align_mtp_history(ComputeOperation&);
    void rebuild_mtp_history(std::uint32_t batch, std::uint32_t columns);
    void request_decision(ComputeOperation&);
    void launch_decision(ComputeOperation&);
    void launch_commit(ComputeOperation&);
    void launch_score(ComputeOperation&);
    void publish_result(ComputeOperation&);
    void retire_compute();
    void progress();
    [[nodiscard]] ComputeOperation& require_compute(runtime::ExecutionTicket);
    [[nodiscard]] const ComputeOperation& require_compute(runtime::ExecutionTicket) const;
    [[nodiscard]] ops::SamplingConfig sampling_config(const ExecutionRow&, const SamplingInput&,
                                                      std::size_t compact_row) const;

    // The two-GPU collective, or null at width 1. The sharded forward takes it; a caller that has
    // none keeps the single-device path, which is what makes width 1 unchanged by construction.
    [[nodiscard]] const execution::ShardedCollective* sharded_collective() const noexcept {
        return collective_ ? &*collective_ : nullptr;
    }

    // The full-width partial pair the collective reduces over, one buffer per rank's device, from
    // the group bundle: BOTH Programs pass the SAME pair, because the combine reads and writes both.
    [[nodiscard]] const std::array<Tensor, 2>& sharded_partials() const noexcept {
        static const std::array<Tensor, 2> none{};
        return sharded_ != nullptr ? sharded_->partials : none;
    }

    const execution::Parameters& parameters;
    DeviceContext& device;
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t max_concurrency;
    const std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    const SpeculativeBackend backend;
    const ProposalHead proposal_head;
    const bool use_cuda_graph;
    const KvCacheStorage kv_storage;
    const std::size_t kv_payload_bytes;
    const WorkspaceSpec workspace_plan;
    // Borrowed from the group's owner; outlives the Program. Null on the single-device lane.
    const models::ShardedGroup* sharded_ = nullptr;
    // This rank's view of the shared collective: the group's mailbox plus this Program's rank.
    std::optional<execution::ShardedCollective> collective_;
    Resources resources;
    std::unique_ptr<context::ContextStorage> storage;
    // tp 2: rank 1's mirrors, owned here so they outlive every mutation they replay. Null on the
    // single-device lane. Their presence is also the sharded-lane flag the round entry refuses on:
    // the sharded forward is not wired, so a round would project sharded weights through the
    // unsharded path and produce wrong numbers.
    std::unique_ptr<tp2::MirroredKVPagePool> peer_text_kv;
    std::unique_ptr<tp2::MirroredKVExecutionTables> peer_text_kv_tables;
    std::unique_ptr<context::ContextMigration> compute_actions;
    std::unique_ptr<context::ContextMigration> context_actions;
    std::unique_ptr<ContextOperation> context_operation;
    PinnedHostBuffer context_counts;
    std::array<Residency, kMaximumConcurrency> residencies;
    qwen::VisionHandoffState vision_handoff;
    std::size_t workspace_peak = 0;
    std::size_t vision_peak    = 0;
    // Handoff peak folded across measurement resets; the within-step peak clears below.
    std::size_t vision_lifetime_peak = 0;
    std::unique_ptr<ComputeOperation> compute;
    CudaCompletionEvent result_ready;
    CudaCompletionEvent drafts_ready;
    CudaCompletionEvent compute_retired;
    CudaCompletionEvent context_ready;
    CudaEventTimer prefill_timer;
    runtime::ExecutionGpuTimer gpu_timer;
    std::deque<runtime::ExecutionEvent<ExecutionTypes>> events;
    std::optional<Graphs> graphs;
    std::optional<GraphMemorySummary> graph_memory;
    std::size_t workspace_lifetime_peak = 0;
};

} // namespace ninfer::models::qwen3_5::detail
