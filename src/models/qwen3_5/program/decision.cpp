#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/linear.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/indexing.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/speculative_accept.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

// Weak per-site tag from the launch counter (tests/support/launch_counter.cpp). Null in production.
extern "C" void ninfer_launch_site(const char*) __attribute__((weak));
#define NINFER_TAG_SITE(name) do { if (&ninfer_launch_site != nullptr) ninfer_launch_site(name); } while (0)

namespace {
Tensor prefix(const Tensor& value, std::initializer_list<std::int32_t> shape) {
    return Tensor({value.data, value.bytes(), value.device}, value.dtype, shape);
}
} // namespace

void RankProgramImpl::request_decision(ComputeOperation& operation) {
    if (operation.decision_requested) return;
    if (const auto* forward = std::get_if<ForwardWork>(&operation.request.work.operation);
        forward && forward->sample == SamplePurpose::None)
        return;
    const bool constrained = std::any_of(operation.sampling.begin(), operation.sampling.end(),
                                         [](const auto& row) { return row.constrained; });
    if (!constrained) {
        operation.decision_supplied = true;
        return;
    }
    if (operation.speculative && !drafts_ready.ready()) return;
    runtime::DecisionRequest request;
    request.stage      = operation.decision_stage;
    request.vocabulary = parameters.profile->public_tokens;
    const auto* drafts = static_cast<const TokenId*>(resources.frame.host_drafts.data);
    for (std::uint32_t row = 0; row < operation.batch; ++row) {
        if (!operation.sampling[row].constrained) continue;
        runtime::DecisionRequest::Row member;
        member.member = row;
        if (operation.speculative) {
            const auto count = resources.frame.host_ingress->extents[row];
            member.drafts.assign(drafts + row * draft_window, drafts + row * draft_window + count);
        }
        request.rows.push_back(std::move(member));
    }
    operation.decision_requested = true;
    events.emplace_back(runtime::DecisionNeeded{operation.request.ticket, std::move(request)});
}

void RankProgramImpl::project_decode(std::uint32_t batch_value, std::uint32_t width_value) {
    auto& frame     = resources.frame;
    const int batch = static_cast<int>(batch_value), width = static_cast<int>(width_value);
    const int rows       = frame.spec.output_rows;
    const int vocabulary = execution::dimension(parameters.profile->public_tokens);
    const int columns    = width * batch;
    Tensor hidden        = frame.input_view(width, batch).hidden;
    Tensor flat_hidden   = hidden.view({hidden.ne[0], columns});
    Tensor flat_logits;
    if (sharded_ == nullptr) {
        // Width 1: the head is whole, so the projection's output extent is the frame's, and the
        // frame's buffer is where it lands. Unchanged by construction.
        flat_logits = prefix(frame.logits, {rows, columns});
        execution::project(flat_hidden, parameters.text.output_head, flat_logits, resources.work,
                           device.execution_view());
    } else {
        // Width 2: the head holds only this rank's block of the vocabulary (a Rows shard), so the
        // projection's output extent is the WEIGHT's row count, not the frame's. Project into a
        // rank-sized scratch and gather both ranks' halves into the group's full-vocabulary pair;
        // afterwards this rank's member holds the whole vocabulary, bit-identical to width 1's
        // logits, and sampling is the width-1 call over it.
        const int rank_rows = parameters.text.output_head.weight.n;
        auto scope          = resources.work.scope();
        Tensor scratch      = resources.work.alloc(DType::BF16, {rank_rows, columns});
        execution::project(flat_hidden, parameters.text.output_head, scratch, resources.work,
                           device.execution_view());
        Tensor full = (*sharded_).logits[static_cast<std::size_t>(collective_->rank)];
        execution::all_gather_logits(*sharded_, collective_->rank, collective_->reduce_index,
                                     scratch, full, device.execution_view(), "decision.sample");
        flat_logits = execution::sharded_logits_view(*sharded_, collective_->rank, rows, columns);
    }
    if (width == 1) return;
    Tensor argmax      = prefix(frame.target_argmax, {width, batch});
    Tensor flat_argmax = argmax.view({width * batch});
    ops::argmax(flat_logits, flat_argmax, vocabulary, device.execution_view());
}

void RankProgramImpl::decide_decode(std::uint32_t batch_value, std::uint32_t width_value) {
    auto& frame     = resources.frame;
    const int batch = static_cast<int>(batch_value), width = static_cast<int>(width_value);
    const int rows       = frame.spec.output_rows;
    const int vocabulary = execution::dimension(parameters.profile->public_tokens);
    if (width > 1) {
        Tensor logits = sharded_ != nullptr
                            ? execution::sharded_logits_view(*sharded_, collective_->rank, rows,
                                                             width * batch)
                            : prefix(frame.logits, {rows, width, batch});
        Tensor argmax = prefix(frame.target_argmax, {width, batch});
        auto drafts =
            prefix(backend == SpeculativeBackend::Mtp ? frame.mtp->drafts : frame.dflash->drafts,
                   {static_cast<int>(draft_window), batch});
        Tensor counts    = frame.output_counts.slice(0, 0, batch);
        Tensor accepted  = frame.accepted_drafts.slice(0, 0, batch);
        Tensor tokens    = prefix(frame.output_tokens, {width, batch});
        Tensor frontiers = frame.frontiers.slice(0, 0, batch);
        Tensor anchors   = frame.anchors.slice(0, 0, batch);
        if (backend == SpeculativeBackend::DFlash2) {
            auto& draft = *frame.dflash;
            ops::speculative_accept_sparse_drafts(
                argmax, logits, drafts,
                prefix(draft.candidate_ids,
                       {draft.candidate_ids.ne[0], static_cast<int>(draft_window), batch}),
                prefix(draft.proposal_q,
                       {draft.proposal_q.ne[0], static_cast<int>(draft_window), batch}),
                frame.extents.slice(0, 0, batch), frontiers, anchors, tokens, counts, accepted,
                vocabulary, frame.sampling, {false}, resources.work, device.execution_view());
        } else {
            ops::speculative_accept_greedy_drafts(
                argmax, logits, drafts, frame.extents.slice(0, 0, batch), frontiers, anchors,
                tokens, counts, accepted, vocabulary, frame.sampling, resources.work,
                device.execution_view());
        }
    } else {
        // The plain decode round samples the GATHERED full-vocabulary logits when the head is
        // row-sharded. `project_decode` has already run for this operation and put the whole
        // vocabulary in this rank's member of the group's logits pair; `frame.logits` is the
        // per-Program buffer the sharded projection never writes, so sampling from it argmaxes
        // zeros and emits token 0 every round. Width 1 keeps the frame buffer, unchanged.
        Tensor logits = sharded_ != nullptr
                            ? execution::sharded_logits_view(*sharded_, collective_->rank, rows,
                                                             batch)
                            : prefix(frame.logits, {rows, batch});
        Tensor tokens = prefix(frame.output_tokens, {batch});
        ops::sample(logits, tokens, vocabulary, frame.sampling, frame.frontiers.slice(0, 0, batch),
                    ops::kSamplePurposeDecode, resources.work, device.execution_view());        for (int row = 0; row < batch; ++row) {
            auto count = frame.output_counts.slice(0, row, 1);
            ops::set_i32_scalar(count, 1, device.execution_view());
        }
        NINFER_TAG_SITE("egress.accepted");
        CUDA_CHECK(cudaMemsetAsync(frame.accepted_drafts.data, 0, batch * sizeof(std::int32_t),
                                   device.stream()));
    }
    // MTP history still consumes the previous tail before commit publishes its replacement.
    if (backend != SpeculativeBackend::Mtp) {
        auto scope        = resources.work.scope();
        const auto hidden = frame.input_view(width, batch).hidden;
        auto selected     = hidden.view({frame.spec.hidden, width * batch});
        if (width > 1) {
            selected = resources.work.alloc(DType::BF16, {frame.spec.hidden, batch});
            ops::gather_bf16_batch_columns(hidden, frame.accepted_drafts.slice(0, 0, batch),
                                           selected, device.execution_view());
        }
        ops::scatter(selected, frame.state_destination_slots.slice(0, 0, batch),
                     resources.state_images.continuation_hidden_store(), device.execution_view());
    }
}

void RankProgramImpl::readback_decode() {
    CUDA_CHECK(cudaMemcpyAsync(resources.frame.host_egress, resources.frame.egress.data,
                               sizeof(FrameEgress), cudaMemcpyDeviceToHost, device.stream()));
}

void RankProgramImpl::launch_decision(ComputeOperation& operation) {
    auto& frame          = resources.frame;
    const int batch      = static_cast<int>(operation.batch);
    const int vocabulary = execution::dimension(parameters.profile->public_tokens);
    const int rows       = frame.spec.output_rows;
    if (operation.decision) {
        runtime::DecisionConsumed consumed{operation.request.ticket, operation.decision_stage, {}};
        for (const auto& input : operation.decision->rows) {
            const auto row = static_cast<int>(input.member);
            auto host      = frame.host_masks.slice(2, row, 1);
            auto target    = frame.masks.slice(2, row, 1);
            std::copy(input.words.begin(), input.words.end(),
                      static_cast<std::uint32_t*>(host.data));
            const auto bytes = input.words.size() * sizeof(std::uint32_t);
            CUDA_CHECK(cudaMemcpyAsync(target.data, host.data, bytes, cudaMemcpyHostToDevice,
                                       device.stream()));
            operation.dead_positions[row] = input.dead_positions;
            consumed.rows.push_back({input.member, bytes});
        }
        events.emplace_back(std::move(consumed));
    }
    resources.work.reset();
    if (std::holds_alternative<DecodeWork>(operation.request.work.operation)) {
        if (!operation.projected) project_decode(operation.batch, operation.width);
        if (graphs) {
            (void)graphs->start(operation.speculative ? GraphPhase::Accept : GraphPhase::Sample,
                                operation.batch, operation.rows.front().main_frontier,
                                device.execution_view());
        } else {
            decide_decode(operation.batch, operation.width);
            readback_decode();
        }
        gpu_timer.record_result();
        result_ready.record(device.stream());
        operation.stage = ComputeStage::Readback;
        return;
    }
    {
        const auto* forward = std::get_if<ForwardWork>(&operation.request.work.operation);
        Tensor hidden;
        std::int32_t purpose = ops::kSamplePurposeDecode;
        if (forward) {
            if (operation.batch != 1)
                throw std::logic_error("sampled prefill must contain one input window");
            if (operation.width)
                hidden = frame.input_view(static_cast<int>(operation.width), 1)
                             .hidden.slice(1, static_cast<int>(operation.width - 1), 1)
                             .view({frame.spec.hidden, 1});
            else
                hidden = resources.state_images
                             .continuation_hidden_slot(operation.rows[0].state_slots.source)
                             .view({frame.spec.hidden, 1});
            auto position = frame.frontiers.slice(0, 0, 1);
            ops::set_i32_scalar(position,
                                static_cast<int>(operation.rows[0].main_frontier + operation.width),
                                device.execution_view());
            purpose = forward->sample == SamplePurpose::FirstToken ? ops::kSamplePurposePrefill
                                                                   : ops::kSamplePurposeDecode;
        } else {
            hidden = frame.input_view(1, batch).hidden.view({frame.spec.hidden, batch});
        }
        Tensor logits;
        if (sharded_ == nullptr) {
            logits = prefix(frame.logits, {rows, batch});
            execution::project(hidden, parameters.text.output_head, logits, resources.work,
                               device.execution_view());
        } else {
            // The same row-sharded head as project_decode, for the one-column call: project into the
            // rank's rows, gather, then sample the whole vocabulary.
            const int rank_rows = parameters.text.output_head.weight.n;
            auto scope          = resources.work.scope();
            Tensor scratch      = resources.work.alloc(DType::BF16, {rank_rows, batch});
            execution::project(hidden, parameters.text.output_head, scratch, resources.work,
                               device.execution_view());
            Tensor full = (*sharded_).logits[static_cast<std::size_t>(collective_->rank)];
            execution::all_gather_logits(*sharded_, collective_->rank, collective_->reduce_index,
                                         scratch, full, device.execution_view(), "decision.accept");
            logits = execution::sharded_logits_view(*sharded_, collective_->rank, rows, batch);
        }
        Tensor tokens = prefix(frame.output_tokens, {batch});
        ops::sample(logits, tokens, vocabulary, frame.sampling, frame.frontiers.slice(0, 0, batch),
                    purpose, resources.work, device.execution_view());
        for (int row = 0; row < batch; ++row) frame.host_egress->counts[row] = 1;
        NINFER_TAG_SITE("egress.counts");
        CUDA_CHECK(cudaMemcpyAsync(frame.output_counts.data, frame.host_egress->counts.data(),
                                   batch * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   device.stream()));
        CUDA_CHECK(cudaMemsetAsync(frame.accepted_drafts.data, 0, batch * sizeof(std::int32_t),
                                   device.stream()));
    }
    NINFER_TAG_SITE("egress.frame");
    CUDA_CHECK(cudaMemcpyAsync(frame.host_egress, frame.egress.data, sizeof(FrameEgress),
                               cudaMemcpyDeviceToHost, device.stream()));
    gpu_timer.record_result();
    result_ready.record(device.stream());
    operation.stage = ComputeStage::Readback;
}

void RankProgramImpl::publish_result(ComputeOperation& operation) {
    gpu_timer.observe_result(operation.gpu);
    Result result;
    const auto* forward  = std::get_if<ForwardWork>(&operation.request.work.operation);
    const bool generated = std::holds_alternative<DecodeWork>(operation.request.work.operation) ||
                           (forward && forward->sample != SamplePurpose::None);
    if (generated) {
        const auto& output = *resources.frame.host_egress;
        GeneratedResult value;
        value.row_stride = operation.speculative ? draft_window + 1 : 1;
        value.tokens.assign(output.tokens.begin(),
                            output.tokens.begin() + value.row_stride * operation.batch);
        for (std::uint32_t row = 0; row < operation.batch; ++row) {
            const auto count  = output.counts[row];
            const auto extent = resources.frame.host_ingress->extents[row];
            if (count < 1 || count > extent + 1)
                throw std::runtime_error("Qwen3.5 sampling returned an invalid prefix length");
            value.counts.push_back(static_cast<std::uint32_t>(count));
            value.accepted_drafts.push_back(
                static_cast<std::uint32_t>(output.accepted_drafts[row]));
            value.draft_extents.push_back(static_cast<std::uint32_t>(extent));
            value.dead_positions.push_back(operation.dead_positions[row]);
            for (int column = 0; column < count; ++column) {
                const auto token = value.tokens[row * value.row_stride + column];
                if (token < 0 ||
                    static_cast<std::uint32_t>(token) >= parameters.profile->public_tokens)
                    throw std::runtime_error(
                        "Qwen3.5 sampling returned a token outside the vocabulary");
            }
        }
        if (forward) {
            auto info      = std::get<ForwardResult>(std::move(operation.result->output));
            info.generated = std::move(value);
            result.output  = std::move(info);
        } else
            result.output = std::move(value);
    } else if (std::holds_alternative<ScoreWork>(operation.request.work.operation)) {
        result.output = ScoreResult{operation.width, std::move(operation.scores)};
    } else if (std::holds_alternative<NormalizeWork>(operation.request.work.operation)) {
        result.output = NormalizeResult{};
    } else
        result = std::move(*operation.result);
    operation.result_published = true;
    operation.stage            = ComputeStage::AwaitCommit;
    events.emplace_back(
        runtime::ResultReady<ExecutionTypes>{operation.request.ticket, std::move(result)});
}

} // namespace ninfer::models::qwen3_5::detail
