#include <cstdio>
#include <cstdlib>
#include <thread>
#include <functional>
#include <ctime>
#include <map>
#include <mutex>
#include <utility>
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/draft.h"
#include "models/qwen3_5/execution/text.h"

#include <algorithm>

namespace ninfer::models::qwen3_5::detail {
namespace {
// A SPIN MUST NOT BE ABLE TO FILL THE DISK. These two stamps are printed at the top of the stage
// machine, and a stalled round enters it millions of times a second: one arm-2 run wrote 53.7 GB in
// 585 s, so the diagnostic cost more than the defect it was meant to explain and buried its own
// useful lines. The rule is 'suppress an IDENTICAL line repeated within 100 ms', not 'suppress this
// site': a stamp whose value changed always prints, so the round sequence is never thinned, while a
// stall produces at most ten lines a second.
constexpr long long kStampPeriodNs = 100'000'000;

bool stamp_due(int rank, const char* where, int round) {
    static const bool on = std::getenv("NINFER_LANE_TRACE") != nullptr;
    if (!on) { return false; }
    std::timespec ts{};
    (void)::clock_gettime(CLOCK_MONOTONIC, &ts);
    const long long now = static_cast<long long>(ts.tv_sec) * 1000000000LL + (long long)ts.tv_nsec;
    // THE KEY INCLUDES THE RANK. With the site alone, two ranks alternating on the same site each
    // saw the other's entry as 'a different value' and both printed every time -- the throttle was
    // defeated exactly when it was needed, and one stalled run wrote 89.7 GB.
    struct Last {
        int rank   = -1;
        int round  = 0;
        long long ns = 0;
    };
    static std::mutex mutex;
    static std::map<std::pair<int, const char*>, Last> last;
    std::lock_guard<std::mutex> lock(mutex);
    Last& slot = last[std::make_pair(rank, where)];
    if (slot.rank == rank && slot.round == round && slot.ns != 0 && now - slot.ns < kStampPeriodNs) {
        return false;
    }
    slot = Last{rank, round, now};
    return true;
}
} // namespace

void stage_stamp(int rank, const char* where) {
    if (!stamp_due(rank, where, 0)) { return; }
    std::timespec ts{};
    (void)::clock_gettime(CLOCK_MONOTONIC, &ts);
    std::fprintf(stderr, "[stage-time] rank=%d at=%s ns=%lld thread=%zu\n", rank, where,
                 (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec,
                 std::hash<std::thread::id>{}(std::this_thread::get_id()));
    std::fflush(stderr);
}

void lane_stamp(int rank, const char* where, int round) {
    if (!stamp_due(rank, where, round)) { return; }
    std::timespec ts{};
    (void)::clock_gettime(CLOCK_MONOTONIC, &ts);
    std::fprintf(stderr, "[lane-time] rank=%d at=%s round=%d ns=%lld\n", rank, where, round,
                 (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec);
    std::fflush(stderr);
}

// THE STAGE IS WHAT DISTINGUISHES 'WAITING' FROM 'BLOCKED'. The entry stamp used to say only that
// progress() had been entered, so a rank spinning in Readback and a rank spinning in AwaitDecision
// produced identical lines, and a rank blocked INSIDE progress() produced none at all -- which reads
// exactly like a rank that is never polled. Naming the stage makes the three cases distinguishable
// from the log alone, which is what the width-2 lane needs: its stalls have been read as 'rank 1 is
// never polled' and as 'rank 1 is blocked' and those want different fixes.
void stage_entry_stamp(int rank, const char* stage) {
    if (!stamp_due(rank, stage, 0)) { return; }
    std::timespec ts{};
    (void)::clock_gettime(CLOCK_MONOTONIC, &ts);
    std::fprintf(stderr, "[stage-entry] rank=%d stage=%s ns=%lld thread=%zu\n", rank, stage,
                 (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec,
                 std::hash<std::thread::id>{}(std::this_thread::get_id()));
    std::fflush(stderr);
}


std::string sharded_forward_refusal(bool has_peer_mirrors, bool forward_wired) {
    if (!has_peer_mirrors) { return {}; }  // the single-device lane
    if (forward_wired) { return {}; }      // sharded lane, forward wired: run it
    return "qwen3.5 sharded lane: this rank carries the peer's KV mirrors, which is the flag that "
           "the sharded forward is not wired; a round would project sharded weights through the "
           "unsharded path and produce wrong numbers. Refusing the round. See "
           "RankProgramImpl::peer_text_kv (models/qwen3_5/program/internal.h) and "
           "attach_rank_mirrors (models/qwen3_5/program/lifecycle.cpp).";
}

std::size_t RankProgramImpl::collective_count() const noexcept {
    // The collective is the only thing that makes one rank's progress depend on another's, so its
    // counter is the rank's position in the group's rendezvous order. `reduce_index` is advanced by
    // ShardedCollective::reduce at ENQUEUE time (execution/sharded.h), which is what the executor
    // needs: the question is what this rank has asked its peer for, not what the device has finished.
    return collective_ ? collective_->reduce_index : 0;
}

void RankProgramImpl::progress() {
    const int rank = collective_ ? collective_->rank : 0;
    stage_entry_stamp(rank, compute ? compute_stage_name(compute->stage) : "no-compute");
    lane_stamp(rank, "round_entry",
               static_cast<int>(collective_ ? collective_->reduce_index : 0));
    // THE REFUSAL THE DECLARATION PROMISES. internal.h:190-194 says the peer mirrors' presence is
    // the sharded-lane flag the round entry refuses on, because the sharded forward is not wired and
    // a round would project sharded weights through the unsharded path and produce wrong numbers.
    // The flag was documented and never checked: the mirrors are attached to the ORIGIN rank only
    // (lifecycle.cpp:370-376), so one rank ran the unsharded path while the other did not, and the
    // mailbox reported the divergence as a give-up rather than as the unwired lane it is.
    if (const auto refusal = sharded_forward_refusal(peer_text_kv != nullptr,
                                                   kShardedForwardWired);
        !refusal.empty()) {
        throw std::logic_error(refusal);
    }
    if (context_operation && context_ready.ready()) {
        const auto ticket    = context_operation->request.ticket;
        const auto command   = context_operation->last_command;
        const auto transfers = context_actions->observations();
        ContextResult result;
        result.transfers.assign(transfers.begin(), transfers.end());
        context_actions->finish();
        events.emplace_back(
            runtime::ResultReady<ExecutionTypes>{ticket, Result{std::move(result)}});
        events.emplace_back(runtime::DeviceRetired{ticket, command, retirement_observation()});
        context_operation.reset();
    }
    if (!compute) return;
    auto& operation = *compute;
    if (operation.cancelled && operation.stage != ComputeStage::Retiring) {
        if (operation.text) {
            operation.text->cancel();
            operation.text.reset();
        }
        if (graphs) graphs->cancel();
        if (resources.moe) resources.moe->cancel();
        gpu_timer.record_retirement();
        compute_retired.record(device.stream());
        operation.stage = ComputeStage::Retiring;
    }
    for (;;) {
        switch (operation.stage) {
        case ComputeStage::Inputs:
            if (std::holds_alternative<NormalizeWork>(operation.request.work.operation)) {
                append_draft_history(operation);
                gpu_timer.record_result();
                result_ready.record(device.stream());
                operation.stage = ComputeStage::Readback;
                break;
            }
            if (std::holds_alternative<DecodeWork>(operation.request.work.operation)) {
                if (!operation.speculative || !graphs) append_draft_history(operation);
                if (operation.speculative) begin_draft(operation);
            }
            if (std::holds_alternative<ForwardWork>(operation.request.work.operation))
                append_draft_history(operation);
            request_decision(operation);
            begin_target(operation);
            operation.stage = ComputeStage::Forward;
            break;
        case ComputeStage::Forward:
            if (!advance_target(operation)) {
                request_decision(operation);
                return;
            }
            if (std::holds_alternative<DecodeWork>(operation.request.work.operation)) {
                if (backend == SpeculativeBackend::Mtp && !operation.speculative)
                    align_mtp_history(operation);
                if (operation.fused_decision) {
                    gpu_timer.record_result();
                    result_ready.record(device.stream());
                    operation.stage = ComputeStage::Readback;
                } else {
                    if (!operation.projected) {
                        resources.work.reset();
                        project_decode(operation.batch, operation.width);
                        operation.projected = true;
                    }
                    lane_stamp(collective_ ? collective_->rank : 0, "traversal_done",
                               static_cast<int>(collective_ ? collective_->reduce_index : 0));
                    stage_stamp(collective_ ? collective_->rank : 0, "traversal_return");
                    operation.stage = ComputeStage::AwaitDecision;
                    stage_stamp(collective_ ? collective_->rank : 0, "await_decision");
                }
                break;
            }
            if (operation.width) {
                if (backend == SpeculativeBackend::Mtp) align_mtp_history(operation);
                if (resources.dflash) {
                    const int width = static_cast<int>(operation.width);
                    auto& draft     = *resources.dflash;
                    auto& frame     = resources.frame;
                    execution::draft_append_context(
                        {parameters, resources.work, device.execution_view(), draft.local,
                         draft.full ? &*draft.full : nullptr, proposal_head},
                        draft.prefill_features.slice(1, 0, width)
                            .view({draft.prefill_features.ne[0], width, 1}),
                        draft.prefill_positions.slice(0, 0, width).view({width, 1}),
                        frame.valid_columns.slice(0, 0, 1),
                        frame.state_destination_slots.slice(0, 0, 1),
                        frame.backend_table_rows.slice(0, 0, 1),
                        {operation.width, operation.width});
                }
                const auto source = resources.frame.input_view(static_cast<int>(operation.width), 1)
                                        .hidden.slice(1, static_cast<int>(operation.width - 1), 1);
                const auto target = resources.state_images.continuation_hidden_slot(
                    operation.rows[operation.forward_row].state_slots.destination);
                CUDA_CHECK(cudaMemcpyAsync(target.data, source.data, target.bytes(),
                                           cudaMemcpyDeviceToDevice, device.stream()));
            }
            if (operation.width) prefill_timer.record_stop();
            gpu_timer.record_result();
            result_ready.record(device.stream());
            operation.stage = ComputeStage::History;
            break;
        case ComputeStage::History:
            if (!result_ready.ready()) return;
            if (operation.width)
                operation.gpu.add_prefill(
                    static_cast<std::uint64_t>(prefill_timer.elapsed_ms() * 1.0e6));
            if (const auto* forward = std::get_if<ForwardWork>(&operation.request.work.operation)) {
                auto& result       = std::get<ForwardResult>(operation.result->output);
                const auto& window = forward->inputs[operation.forward_row];
                operation.forward_offset += operation.width;
                const auto lane = operation.rows[operation.forward_row].member.residency.lane.value;
                auto& residency = residencies[lane];
                if (residency.vision) {
                    operation.gpu.add_vision(static_cast<std::uint64_t>(
                        (residency.vision->elapsed_seconds() - operation.vision_before) * 1.0e9));
                    if (window.prompt_begin + operation.forward_offset >=
                        residency.vision_input->plan.uses.back().end) {
                        residency.vision.reset();
                        residency.vision_input.reset();
                    }
                }
                if (operation.forward_offset == window.tokens.size()) {
                    result.processed_tokens.push_back(operation.forward_offset);
                    operation.forward_offset = 0;
                    ++operation.forward_row;
                }
                if (operation.forward_row < operation.rows.size()) {
                    initialize_forward(operation, *forward);
                    append_draft_history(operation);
                    begin_target(operation);
                    operation.stage = ComputeStage::Forward;
                } else if (forward->sample != SamplePurpose::None) {
                    operation.stage = ComputeStage::AwaitDecision;
                } else {
                    operation.stage = ComputeStage::Readback;
                }
            } else if (std::holds_alternative<ScoreWork>(operation.request.work.operation)) {
                launch_score(operation);
            }
            break;
        case ComputeStage::AwaitDecision:
            request_decision(operation);
            if (!operation.decision_supplied) return;
            launch_decision(operation);
            break;
        case ComputeStage::Readback:
            if (!result_ready.ready()) return;
            if (const auto* score = std::get_if<ScoreWork>(&operation.request.work.operation)) {
                const auto* values =
                    static_cast<const float*>(resources.frame.host_score_logprobs.data);
                std::copy_n(values, operation.score_count,
                            operation.scores.begin() + operation.score_cursor);
                operation.score_cursor += operation.score_count;
                if (operation.score_cursor < score->targets.size()) {
                    launch_score(operation);
                    break;
                }
            }
            publish_result(operation);
            return;
        case ComputeStage::AwaitCommit:
            return;
        case ComputeStage::Commit:
            launch_commit(operation);
            if (!compute) return;
            break;
        case ComputeStage::Retiring:
            // THE RETIREMENT WAITS FOR THE ROUND'S LAST ENQUEUE, NOT FOR THE COMMIT POINT.
            // compute_retired is recorded at commit.cpp:145, which is BEFORE the decision/accept
            // stage enqueues the round's gather and its egress copy (decision.cpp:189-252). Waiting
            // on it alone therefore retires the compute frame -- and lets the next round start --
            // while this round's last launches are still in flight on this rank's stream. The next
            // round's first exchange then queues behind them and starts one launch later than the
            // other rank's, which is exactly the width-2 lane's 'peer epoch one below the target'
            // (gap=1 at slot=0).
            //
            // result_ready is the ROUND-END event: it is recorded after the last enqueue of the
            // round on the decision path (decision.cpp:252) and after the fused path's final graph
            // launch. Requiring it here makes the release criterion 'every launch the round
            // enqueued has completed' rather than 'the launches up to the commit point have
            // completed'. compute_retired keeps its own job: the Retiring stage still starts only
            // once the commit's launches are done, and both conditions are required.
            if (!compute_retired.ready() || !result_ready.ready()) return;
            retire_compute();
            return;
        }
    }
}

} // namespace ninfer::models::qwen3_5::detail
