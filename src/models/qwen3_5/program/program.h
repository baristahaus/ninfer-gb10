#pragma once

#include "models/qwen3_5/planning.h"
#include "models/qwen3_5/work.h"

#include <memory>
#include <optional>

namespace ninfer::models::qwen3_5 {

namespace detail {
class RankProgramImpl;
}

class RankProgram {
public:
    ~RankProgram();
    RankProgram(const RankProgram&)            = delete;
    RankProgram& operator=(const RankProgram&) = delete;

    [[nodiscard]] runtime::ExecutionCapacity capacity() const noexcept;
    void validate_work(const Work&, runtime::FrameKind) const;
    void validate_decision(runtime::ExecutionTicket, const runtime::DecisionInput&) const;
    void prepare_commit(runtime::ExecutionTicket, Commit&) const;

    // Two-phase dispatch for a local group. `start_frame` enqueues the frame and its inputs but
    // does not run the stage machine; `progress` then advances it. The executor enqueues every
    // rank's frame before any rank progresses, and then drives the ranks' progress concurrently,
    // so no rank can block on a collective before every rank has been dispatched and the peer can
    // keep enqueueing while one rank's host is blocked. `start` is the single-rank entry point
    // (the distributed WorkerLoop): start_frame followed by progress.
    void start_frame(runtime::ExecutionRequest<ExecutionTypes>&&);
    void start(runtime::ExecutionRequest<ExecutionTypes>&&);
    void progress();
    void supply(runtime::ExecutionTicket, runtime::CommandSequence, runtime::DecisionInput&&);
    void commit(runtime::ExecutionTicket, runtime::CommandSequence, Commit&&);
    void discard(runtime::ExecutionTicket, runtime::CommandSequence);
    [[nodiscard]] std::optional<runtime::ExecutionEvent<ExecutionTypes>> poll();
    void drain();

    [[nodiscard]] RankMemoryUsage observe_resources() const;
    [[nodiscard]] std::optional<GraphMemorySummary> graph_memory_observation() const;
    [[nodiscard]] std::vector<context::KVTransferProfile> context_transfer_profiles() const;
    void reset_measurement_peaks() noexcept;

private:
    explicit RankProgram(std::unique_ptr<detail::RankProgramImpl>);
    std::unique_ptr<detail::RankProgramImpl> impl_;

    friend std::unique_ptr<RankProgram> create_rank_program(const execution::Parameters&,
                                                            SequencePlan&&, DeviceContext&,
                                                            const StartupObserver&,
                                                            const models::ShardedGroup*);
    friend void attach_rank_mirrors(RankProgram&, RankProgram&, DeviceContext&,
                                    PreparedSession*);
};

} // namespace ninfer::models::qwen3_5
