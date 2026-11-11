#pragma once

#include "runtime/executor/executor.h"
#include "runtime/executor/rank_worker.h"
#include "runtime/observability/snapshot.h"

#include <exception>
#include <type_traits>

namespace ninfer::runtime {

template <class Types>
class LocalExecutor final : public Executor<typename Types::ExecutionTypes> {
public:
    using Protocol = typename Types::ExecutionTypes;
    using Event    = ExecutionEvent<Protocol>;

    // `workers` are the group's ranks in rank order. Width 1 is a one-element vector and every
    // call below takes exactly the path it took when the executor held one worker; width 2 adds
    // rank 1 to the same loops, in the same order, which is the ordering argument the rank-1
    // mirrors rely on.
    LocalExecutor(std::vector<std::unique_ptr<RankWorker<Types>>> workers, SessionEpoch epoch)
        : workers_(std::move(workers)), epoch_(epoch),
          pool_(workers_.front()->program().capacity()),
          compute_(pool_.capacity().compute_frames), context_(pool_.capacity().context_frames) {
        if (workers_.empty()) { throw std::invalid_argument("executor needs at least one rank"); }
        this->observations_.resize(workers_.size());
        for (std::size_t rank = 0; rank < workers_.size(); ++rank) {
            this->observations_[rank].hardware = workers_[rank]->device_name();
            this->observations_[rank].device   = workers_[rank]->device().index();
        }
        sample_resources(true);
    }

    ~LocalExecutor() override {
        try {
            bind_to_current_thread();
            for_each_program([](auto& program) { program.drain(); });
        } catch (...) {
            // RankProgram's teardown retains its allocation owners until outstanding work ends.
        }
    }

    // Group-wide bind: selects every rank's device in rank order, so the LAST rank is current when
    // this returns. That is what a caller acting on all ranks in sequence wants (a collective that
    // enqueues both ranks from one thread), and it is NOT a per-rank bind: it says nothing about
    // which device must be current for a later per-rank call. A group-wide bind is not a per-rank
    // bind -- the last rank wins -- and any per-rank CUDA work must bind its own rank first
    // (`bind_rank` below); every dispatch in this class does so. At width 1 the single rank is also
    // the last, which is why the distinction is invisible on the one-GPU lane.
    void bind_to_current_thread() override {
        for (auto& worker : workers_) { worker->bind_to_current_thread(); }
    }

    // Rank 0's Program is the authority: it owns the capacity, the frame pool and the events the
    // engine consumes. A rank above 0 replays, so its results are issued but never read here. A
    // caller that executes on it binds rank 0 first; the dispatch methods below do.
    [[nodiscard]] typename Types::RankProgram& program() noexcept {
        return workers_.front()->program();
    }

    // Every mutation is issued on every rank, rank 0 first, in the same order. Each rank's device
    // is bound immediately before its Program is touched, because the caller's current device is
    // not necessarily that rank's (see bind_to_current_thread above).
    template <typename Fn>
    void for_each_program(Fn&& fn) {
        for (std::size_t rank = 0; rank < workers_.size(); ++rank) {
            bind_rank(rank);
            fn(workers_[rank]->program());
        }
    }

    [[nodiscard]] SessionEpoch epoch() const noexcept override { return epoch_; }

    [[nodiscard]] ExecutionCapacity capacity() const noexcept override { return pool_.capacity(); }

    [[nodiscard]] std::optional<ExecutionReservation> try_reserve(FrameKind kind) override {
        require_available();
        return pool_.try_reserve(kind);
    }

    [[nodiscard]] ExecutionTicket submit(ExecutionReservation reservation,
                                         typename Protocol::Work work) override {
        require_available();
        if (!reservation) throw std::logic_error("empty execution reservation");
        bind_rank(0);
        program().validate_work(work, reservation.frame().kind);
        const ExecutionTicket ticket{epoch_, ++next_execution_,
                                     pool_.accept(std::move(reservation))};
        auto& flight = slot(ticket.frame);
        flight       = {ticket, next_command(), Phase::Running, {}};
        flight.clock.start(Protocol::describe_work(work), observation_time_ns());
        try {
            observe_host_call(host_.submit_ns, [&] {
                for_each_program([&](auto& program) {
                    program.start({ticket, flight.command, work});
                });
            });
        } catch (...) { record_failure(ticket); }
        return ticket;
    }

    void supply(ExecutionTicket ticket, DecisionInput input) override {
        require_available();
        auto& flight = find(ticket);
        if (flight.phase != Phase::Decision || flight.decision_stage != input.stage) {
            throw std::logic_error("decision input does not match the pending execution stage");
        }
        bind_rank(0);
        program().validate_decision(ticket, input);
        flight.command = next_command();
        flight.phase   = Phase::Running;
        try {
            observe_host_call(host_.supply_ns, [&] {
                // The payload is per-round control data, identical on every rank; rank 0 takes
                // the move and the others take a copy, in rank order.
                for (std::size_t rank = 0; rank < workers_.size(); ++rank) {
                    bind_rank(rank);
                    auto& program = workers_[rank]->program();
                    if (rank == 0) {
                        program.supply(ticket, flight.command, std::move(input));
                    } else {
                        auto copy = input;
                        program.supply(ticket, flight.command, std::move(copy));
                    }
                }
            });
        } catch (...) { record_failure(ticket); }
    }

    [[nodiscard]] CommitReceipt commit(ExecutionTicket ticket,
                                       typename Protocol::Commit commit) override {
        require_available();
        auto& flight = find(ticket);
        if (flight.phase != Phase::Result) {
            throw std::logic_error("execution has no result awaiting commit");
        }
        bind_rank(0);
        program().prepare_commit(ticket, commit);
        flight.command = next_command();
        flight.phase   = Phase::Retiring;
        try {
            observe_host_call(host_.commit_ns, [&] {
                // EVERY rank must receive the SAME commit, so the move happens on the LAST rank
                // only and every earlier rank takes a copy. Moving on rank 0 emptied the object
                // before rank 1 was handed it, and rank 1 then committed a commit with no rows —
                // which `prepare_commit` could not catch because it runs on rank 0 only, and which
                // `launch_commit` caught as a segfault before it learned to refuse.
                for (std::size_t rank = 0; rank < workers_.size(); ++rank) {
                    bind_rank(rank);
                    auto& program = workers_[rank]->program();
                    if (rank + 1 == workers_.size()) {
                        program.commit(ticket, flight.command, std::move(commit));
                    } else {
                        program.commit(ticket, flight.command, typename Protocol::Commit{commit});
                    }
                }
            });
        } catch (...) {
            record_failure(ticket);
            throw;
        }
        flight.clock.resolve(observation_time_ns());
        return {ticket, flight.command};
    }

    void discard(ExecutionTicket ticket) override {
        require_available();
        auto& flight = find(ticket);
        if (flight.phase == Phase::Retiring || flight.phase == Phase::Discarding) return;
        flight.command = next_command();
        flight.phase   = Phase::Discarding;
        try {
            observe_host_call(host_.discard_ns,
                              [&] {
                                  for_each_program(
                                      [&](auto& program) {
                                          program.discard(ticket, flight.command);
                                      });
                              });
            flight.clock.resolve(observation_time_ns());
        } catch (...) { record_failure(ticket); }
    }

    [[nodiscard]] std::optional<Event> poll() override {
        if (failure_) return Event{std::exchange(failure_, {}).value()};
        if (failed_ || closed_) return {};
        sample_resources(false);
        std::optional<Event> event;
        try {
            for (;;) {
                event =
                    // Rank 0's event is the ticket's progress. That is sound only while every
                    // peer's work is strictly stream-ordered behind rank 0's, which is what the
                    // rank-1 mirrors' ordering argument gives: both ranks issue the same sequence
                    // in the same order, so rank 0 cannot complete a round before a peer has been
                    // issued the same round. If a peer ever gains independent progress (its own
                    // scheduler decision, a conditional exchange), this becomes a race and the
                    // ticket must wait for every rank's event instead.
                    observe_host_call(host_.progress_ns, [&] {
                        bind_rank(0);
                        auto event = program().poll();
                        for (std::size_t rank = 1; rank < workers_.size(); ++rank) {
                            bind_rank(rank);
                            (void)workers_[rank]->program().poll();
                        }
                        return event;
                    });
                if (!event) return {};
                if (std::holds_alternative<ExecutionFailed>(*event)) {
                    failed_ = true;
                    return event;
                }
                const auto ticket =
                    std::visit([](const auto& value) { return value.ticket; }, *event);
                const auto& flight = find(ticket);
                if (flight.phase != Phase::Discarding ||
                    (!std::holds_alternative<DecisionNeeded>(*event) &&
                     !std::holds_alternative<ResultReady<Protocol>>(*event)))
                    break;
            }
            std::visit(
                [&](auto& value) {
                    using Value  = std::decay_t<decltype(value)>;
                    auto& flight = find(value.ticket);
                    if constexpr (std::is_same_v<Value, DecisionNeeded>) {
                        if (flight.phase != Phase::Running) {
                            throw std::logic_error("execution requested a decision out of order");
                        }
                        flight.phase          = Phase::Decision;
                        flight.decision_stage = value.request.stage;
                    } else if constexpr (std::is_same_v<Value, DecisionConsumed>) {
                        if (value.stage != flight.decision_stage) {
                            throw std::logic_error("decision upload refers to another stage");
                        }
                    } else if constexpr (std::is_same_v<Value, ResultReady<Protocol>>) {
                        if (flight.phase != Phase::Running) {
                            throw std::logic_error("execution returned a result out of order");
                        }
                        flight.phase = value.ticket.frame.kind == FrameKind::Compute
                                           ? Phase::Result
                                           : Phase::Retiring;
                        flight.clock.result(observation_time_ns());
                        value.execution = flight.clock.finish(observation_time_ns());
                    } else if constexpr (std::is_same_v<Value, DeviceRetired>) {
                        if ((flight.phase != Phase::Retiring &&
                             flight.phase != Phase::Discarding) ||
                            value.command != flight.command) {
                            throw std::logic_error("execution retired before its final command");
                        }
                        value.execution              = flight.clock.finish(observation_time_ns());
                        value.observed.host          = host_;
                        value.execution.max_rank_gpu = value.observed.gpu;
                        this->observe_retirement(0, value.observed,
                                                 value.ticket.frame.kind == FrameKind::Compute);
                        accumulate(this->execution_statistics_, value.execution);
                        pool_.retire(value.ticket.frame);
                        flight = {};
                    } else {
                        failed_ = true;
                    }
                },
                *event);
        } catch (...) {
            record_failure(event
                               ? std::visit([](const auto& value) { return value.ticket; }, *event)
                               : ExecutionTicket{});
            return Event{std::exchange(failure_, {}).value()};
        }
        return event;
    }

    void flush() override {}

    void drain() override {
        closed_ = true;
        std::exception_ptr error;
        try {
            for_each_program([](auto& program) { program.drain(); });
        } catch (...) { error = std::current_exception(); }
        pool_.retire_submitted();
        std::fill(compute_.begin(), compute_.end(), Flight{});
        std::fill(context_.begin(), context_.end(), Flight{});
        if (error) std::rethrow_exception(error);
    }

    void fail(std::string message) override {
        if (failed_) return;
        failed_  = true;
        failure_ = ExecutionFailed{{}, std::move(message)};
    }

    [[nodiscard]] bool failed() const noexcept override { return failed_; }

    [[nodiscard]] bool has_pending_work() const noexcept override { return pool_.busy(); }

    void reset_measurement_peaks() override {
        for_each_program([](auto& program) { program.reset_measurement_peaks(); });
        for (std::size_t rank = 0; rank < workers_.size(); ++rank) {
            bind_rank(rank);
            this->observations_[rank].memory = workers_[rank]->program().observe_resources();
        }
    }

private:
    // A group-wide bind is not a per-rank bind -- the last rank wins. This selects one rank's
    // device so the caller can touch that rank's Program; see bind_to_current_thread above.
    void bind_rank(std::size_t rank) { workers_[rank]->bind_to_current_thread(); }

    void sample_resources(bool force) {
        bind_rank(0);
        auto& snapshot = this->observations_[0];
        const auto now = observation_time_ns();
        // The program observation is cheap and carries the measurement peaks; only the
        // physical device sample keeps its period.
        snapshot.memory = program().observe_resources();
        snapshot.host   = host_;
        if (!force && now - snapshot.stamp.sampled_ns < kResourceSamplePeriodNs) return;
        if (force || !pool_.in_flight())
            snapshot.physical_memory = sample_physical_memory(workers_.front()->device());
        snapshot.stamp       = {snapshot.stamp.sequence + 1, now};
        snapshot.received_ns = now;
    }
    enum class Phase : std::uint8_t { Empty, Running, Decision, Result, Retiring, Discarding };

    struct Flight {
        ExecutionTicket ticket;
        CommandSequence command;
        Phase phase                  = Phase::Empty;
        std::uint32_t decision_stage = 0;
        ExecutionClock clock;
    };

    [[nodiscard]] Flight& slot(FrameId frame) {
        return (frame.kind == FrameKind::Compute ? compute_ : context_).at(frame.index);
    }

    [[nodiscard]] Flight& find(ExecutionTicket ticket) {
        auto& flight = slot(ticket.frame);
        if (ticket.epoch != epoch_ || flight.ticket != ticket || !pool_.submitted(ticket.frame)) {
            throw std::logic_error("execution ticket is stale");
        }
        return flight;
    }

    [[nodiscard]] CommandSequence next_command() noexcept { return {++next_command_}; }

    void require_available() const {
        if (closed_) throw std::runtime_error("executor is closed");
        if (failed_) throw std::runtime_error("executor is unavailable after an execution failure");
    }

    void record_failure(ExecutionTicket ticket) {
        std::string message;
        try {
            throw;
        } catch (const std::exception& error) { message = error.what(); } catch (...) {
            message = "unknown rank execution failure";
        }
        failed_ = true;
        if (!failure_) failure_ = ExecutionFailed{ticket, std::move(message)};
    }

    std::vector<std::unique_ptr<RankWorker<Types>>> workers_;
    SessionEpoch epoch_;
    ExecutionPool pool_;
    std::vector<Flight> compute_, context_;
    std::uint64_t next_execution_ = 0, next_command_ = 0;
    bool failed_ = false, closed_ = false;
    RankHostStatistics host_;
    std::optional<ExecutionFailed> failure_;
};

} // namespace ninfer::runtime
