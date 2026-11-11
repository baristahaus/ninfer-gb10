#pragma once

#include "runtime/executor/executor.h"
#include "runtime/executor/rank_worker.h"
#include "runtime/executor/round_trace.h"
#include "runtime/observability/snapshot.h"

#include <array>
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
        if (round_trace_enabled()) {
            if (!trace_gate_printed_) {
                trace_gate_printed_ = true;
                round_trace(
                    "gate=submit.serial ranks=%zu: LocalExecutor::submit calls start rank 0 first "
                    "and does not issue rank 1's start until rank 0's start returns "
                    "(for_each_program). A block or throw in rank 0's start is a block or throw "
                    "before rank 1's start is called.",
                    workers_.size());
                round_trace(
                    "gate=executor.poll rank=0-only: LocalExecutor::poll returns rank 0's event and "
                    "polls-then-discards every rank>=1 event; EngineCore::poll_execution consumes "
                    "that one event, clears compute_/retiring_ and releases the next round "
                    "(engine_execution.inl poll_execution; engine_schedule.inl worker_loop).");
            }
            round_trace("submit ticket=%llu round=%llu kind=%s frame=%u gen=%llu command=%llu "
                        "ranks=%zu",
                        static_cast<unsigned long long>(ticket.id),
                        static_cast<unsigned long long>(ticket.id),
                        round_trace_frame_kind_name(ticket.frame.kind), ticket.frame.index,
                        static_cast<unsigned long long>(ticket.frame.generation),
                        static_cast<unsigned long long>(flight.command.value), workers_.size());
        }
        try {
            observe_host_call(host_.submit_ns, [&] {
                // The trace needs the rank and the frame kind at each start, so this is
                // for_each_program written out: same rank order, same per-rank bind, same call.
                for (std::size_t rank = 0; rank < workers_.size(); ++rank) {
                    bind_rank(rank);
                    trace_start_enter(rank, ticket, flight.command);
                    const auto began = round_trace_now_ns();
                    try {
                        workers_[rank]->program().start({ticket, flight.command, work});
                    } catch (const std::exception& error) {
                        trace_start_throw(rank, ticket, flight.command, began, error.what());
                        throw;
                    } catch (...) {
                        trace_start_throw(rank, ticket, flight.command, began,
                                          "non-std exception");
                        throw;
                    }
                    trace_start_return(rank, ticket, flight.command, began);
                }
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
                    if (round_trace_enabled()) {
                        round_trace("commit.enter rank=%zu dev=%d ticket=%llu kind=%s command=%llu",
                                    rank,
                                    static_cast<int>(workers_[rank]->device().index()),
                                    static_cast<unsigned long long>(ticket.id),
                                    round_trace_frame_kind_name(ticket.frame.kind),
                                    static_cast<unsigned long long>(flight.command.value));
                    }
                    const auto began = round_trace_now_ns();
                    auto& program    = workers_[rank]->program();
                    try {
                        if (rank + 1 == workers_.size()) {
                            program.commit(ticket, flight.command, std::move(commit));
                        } else {
                            program.commit(ticket, flight.command,
                                           typename Protocol::Commit{commit});
                        }
                    } catch (const std::exception& error) {
                        if (round_trace_enabled()) {
                            round_trace("commit.throw rank=%zu dev=%d ticket=%llu kind=%s "
                                        "elapsed-us=%llu what=\"%s\"",
                                        rank,
                                        static_cast<int>(workers_[rank]->device().index()),
                                        static_cast<unsigned long long>(ticket.id),
                                        round_trace_frame_kind_name(ticket.frame.kind),
                                        static_cast<unsigned long long>(
                                            (round_trace_now_ns() - began) / 1000ULL),
                                        error.what());
                        }
                        throw;
                    }
                    if (round_trace_enabled()) {
                        round_trace("commit.return rank=%zu dev=%d ticket=%llu kind=%s "
                                    "elapsed-us=%llu",
                                    rank,
                                    static_cast<int>(workers_[rank]->device().index()),
                                    static_cast<unsigned long long>(ticket.id),
                                    round_trace_frame_kind_name(ticket.frame.kind),
                                    static_cast<unsigned long long>(
                                        (round_trace_now_ns() - began) / 1000ULL));
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
        if (failure_) {
            if (round_trace_enabled()) {
                round_trace("poll.failure pending=yes ticket=%llu message=\"%s\"",
                            static_cast<unsigned long long>(failure_->ticket.id),
                            failure_->message.c_str());
            }
            return Event{std::exchange(failure_, {}).value()};
        }
        if (failed_ || closed_) {
            if (round_trace_enabled()) {
                round_trace("poll.stop failed=%s closed=%s (no event returned)",
                            failed_ ? "yes" : "no", closed_ ? "yes" : "no");
            }
            return {};
        }
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
                        auto rank0_event = program().poll();
                        trace_poll_result(0, rank0_event);
                        for (std::size_t rank = 1; rank < workers_.size(); ++rank) {
                            bind_rank(rank);
                            auto peer_event = workers_[rank]->program().poll();
                            // Rank 1's event is polled and discarded: only rank 0's is returned.
                            // The trace prints it because a failure on the peer would otherwise
                            // be invisible here.
                            trace_poll_result(rank, peer_event);
                        }
                        return rank0_event;
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
            if (round_trace_enabled()) {
                round_trace("poll.return rank=0 ticket=%llu event=%s (the only event returned; "
                            "rank>=1 events above were discarded)",
                            static_cast<unsigned long long>(trace_event_ticket(*event)),
                            trace_event_name(*event));
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
            if (round_trace_enabled()) {
                round_trace("poll.throw (a per-rank poll or event handler threw; the failure is "
                            "recorded and returned immediately)");
            }
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

    // DIAGNOSTIC (NINFER_ROUND_TRACE=1). The poll counters are members indexed by rank, not
    // process-wide statics: a shared counter merges the two ranks, which has already produced a
    // false result in this investigation. The line itself states the throttle.
    static constexpr std::uint64_t kTracePollFirst    = 8;
    static constexpr std::uint64_t kTracePollPeriodNs = 500'000'000ULL;

    static const char* trace_event_name(const Event& event) {
        return std::visit(
            [](const auto& value) -> const char* {
                using Value = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<Value, DecisionNeeded>) {
                    return "decision-needed";
                } else if constexpr (std::is_same_v<Value, DecisionConsumed>) {
                    return "decision-consumed";
                } else if constexpr (std::is_same_v<Value, ResultReady<Protocol>>) {
                    return "result-ready";
                } else if constexpr (std::is_same_v<Value, DeviceRetired>) {
                    return "device-retired";
                } else {
                    return "execution-failed";
                }
            },
            event);
    }

    static std::uint64_t trace_event_ticket(const Event& event) {
        return std::visit([](const auto& value) { return value.ticket.id; }, event);
    }

    void trace_start_enter(std::size_t rank, const ExecutionTicket& ticket,
                           CommandSequence command) {
        if (!round_trace_enabled()) return;
        round_trace("submit.start.enter rank=%zu dev=%d ticket=%llu kind=%s frame=%u gen=%llu "
                    "command=%llu (LocalExecutor::submit -> program.start)",
                    rank, static_cast<int>(workers_[rank]->device().index()),
                    static_cast<unsigned long long>(ticket.id),
                    round_trace_frame_kind_name(ticket.frame.kind), ticket.frame.index,
                    static_cast<unsigned long long>(ticket.frame.generation),
                    static_cast<unsigned long long>(command.value));
    }

    void trace_start_return(std::size_t rank, const ExecutionTicket& ticket,
                            CommandSequence command, std::uint64_t began) {
        if (!round_trace_enabled()) return;
        round_trace("submit.start.return rank=%zu dev=%d ticket=%llu kind=%s command=%llu "
                    "elapsed-us=%llu",
                    rank, static_cast<int>(workers_[rank]->device().index()),
                    static_cast<unsigned long long>(ticket.id),
                    round_trace_frame_kind_name(ticket.frame.kind),
                    static_cast<unsigned long long>(command.value),
                    static_cast<unsigned long long>((round_trace_now_ns() - began) / 1000ULL));
    }

    void trace_start_throw(std::size_t rank, const ExecutionTicket& ticket,
                           CommandSequence command, std::uint64_t began, const char* what) {
        if (!round_trace_enabled()) return;
        round_trace("submit.start.throw rank=%zu dev=%d ticket=%llu kind=%s command=%llu "
                    "elapsed-us=%llu what=\"%s\"",
                    rank, static_cast<int>(workers_[rank]->device().index()),
                    static_cast<unsigned long long>(ticket.id),
                    round_trace_frame_kind_name(ticket.frame.kind),
                    static_cast<unsigned long long>(command.value),
                    static_cast<unsigned long long>((round_trace_now_ns() - began) / 1000ULL),
                    what);
    }

    void trace_poll_result(std::size_t rank, const std::optional<Event>& event) {
        if (!round_trace_enabled()) return;
        const int dev  = static_cast<int>(workers_[rank]->device().index());
        const auto now = round_trace_now_ns();
        const bool has_event     = event.has_value();
        const bool event_failure = has_event && std::holds_alternative<ExecutionFailed>(*event);
        const char* failure_state = failure_.has_value() ? "pending" : "no";
        if (rank < trace_poll_calls_.size()) {
            const std::uint64_t call = ++trace_poll_calls_[rank];
            const bool first         = call <= kTracePollFirst;
            const bool periodic      = now - trace_poll_last_ns_[rank] >= kTracePollPeriodNs;
            if (!has_event && !event_failure && !first && !periodic) return;
            trace_poll_last_ns_[rank] = now;
            if (has_event) {
                round_trace("poll rank=%zu dev=%d n=%llu event=%s ticket=%llu failure=%s "
                            "failed=%s",
                            rank, dev, static_cast<unsigned long long>(call),
                            trace_event_name(*event),
                            static_cast<unsigned long long>(trace_event_ticket(*event)),
                            failure_state, failed_ ? "yes" : "no");
            } else {
                round_trace("poll rank=%zu dev=%d n=%llu event=none failure=%s failed=%s "
                            "(throttle: first %llu polls per rank then at most one per %llu ms; "
                            "an event or a recorded failure always prints)",
                            rank, dev, static_cast<unsigned long long>(call), failure_state,
                            failed_ ? "yes" : "no",
                            static_cast<unsigned long long>(kTracePollFirst),
                            static_cast<unsigned long long>(kTracePollPeriodNs / 1'000'000ULL));
            }
            return;
        }
        // Width above two: this trace's per-rank arrays cover the two-rank lane. Print without a
        // throttle rather than silently dropping a rank.
        if (has_event) {
            round_trace("poll rank=%zu dev=%d event=%s ticket=%llu failure=%s failed=%s "
                        "(width>2: no throttle)",
                        rank, dev, trace_event_name(*event),
                        static_cast<unsigned long long>(trace_event_ticket(*event)),
                        failure_state, failed_ ? "yes" : "no");
        } else {
            round_trace("poll rank=%zu dev=%d event=none failure=%s failed=%s "
                        "(width>2: no throttle)",
                        rank, dev, failure_state, failed_ ? "yes" : "no");
        }
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
    // DIAGNOSTIC (NINFER_ROUND_TRACE=1): per-rank poll counters and throttle stamps. The rank is
    // the index into these arrays; there is no process-wide counter.
    std::array<std::uint64_t, 2> trace_poll_calls_{};
    std::array<std::uint64_t, 2> trace_poll_last_ns_{};
    bool trace_gate_printed_ = false;
};

} // namespace ninfer::runtime
