#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/linear.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/target_logprobs.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

void RankProgramImpl::launch_score(ComputeOperation& operation) {
    const auto& work      = std::get<ScoreWork>(operation.request.work.operation);
    auto& frame           = resources.frame;
    operation.score_count = std::min<std::uint32_t>(
        frame.score_targets.ne[0],
        static_cast<std::uint32_t>(work.targets.size()) - operation.score_cursor);
    if (!operation.score_count) {
        gpu_timer.record_result();
        result_ready.record(device.stream());
        operation.stage = ComputeStage::Readback;
        return;
    }
    const int count = static_cast<int>(operation.score_count);
    // The History/readback fence retires the prior use of this pinned region.
    auto* host_targets = static_cast<std::int32_t*>(frame.host_input_ids.data);
    std::copy_n(work.targets.begin() + operation.score_cursor, count, host_targets);
    Tensor targets = frame.score_targets.slice(0, 0, count);
    CUDA_CHECK(cudaMemcpyAsync(targets.data, host_targets, targets.bytes(), cudaMemcpyHostToDevice,
                               device.stream()));
    resources.work.reset();
    Tensor hidden =
        frame.input_view(static_cast<int>(operation.width), 1)
            .hidden.slice(1, static_cast<int>(work.first_column + operation.score_cursor), count)
            .view({frame.spec.hidden, count});
    Tensor logits;
    if (sharded_ == nullptr) {
        logits = resources.work.alloc(DType::BF16, {frame.spec.output_rows, count});
        execution::project(hidden, parameters.text.output_head, logits, resources.work,
                           device.execution_view());
    } else {
        // The head is row-sharded here too: project this rank's block of the vocabulary, gather the
        // two blocks into the group's full-vocabulary pair, then take the target log probabilities
        // over the whole vocabulary exactly as width 1 does.
        const int rank_rows = parameters.text.output_head.weight.n;
        Tensor scratch      = resources.work.alloc(DType::BF16, {rank_rows, count});
        execution::project(hidden, parameters.text.output_head, scratch, resources.work,
                           device.execution_view());
        Tensor full = (*sharded_).logits[static_cast<std::size_t>(collective_->rank)];
        execution::all_gather_logits(*sharded_, collective_->rank, collective_->reduce_index,
                                     scratch, full, device.execution_view());
        logits = execution::sharded_logits_view(*sharded_, collective_->rank,
                                                frame.spec.output_rows, count);
    }
    // Diagnostic: NINFER_SCORE_DEBUG_ROW=<id> prints, for every position of this round whose target
    // is that id, the target row's logit and the vocabulary's max logit with its argmax id. It
    // separates "this vocabulary row is wrong" from "another row poisoned the log_softmax
    // denominator", which are different defects with different fixes. Off unless the variable is
    // set, and it writes no state.
    if (const char* debug = std::getenv("NINFER_SCORE_DEBUG_ROW"); debug != nullptr && *debug) {
        const auto wanted    = static_cast<std::int32_t>(std::strtol(debug, nullptr, 10));
        const int vocabulary = execution::dimension(parameters.profile->public_tokens);
        Tensor argmax_ids    = resources.work.alloc(DType::I32, {count});
        ops::argmax(logits, argmax_ids, vocabulary, device.execution_view());
        std::vector<std::int32_t> host_ids(static_cast<std::size_t>(count));
        std::vector<std::int32_t> host_targets(static_cast<std::size_t>(count));
        CUDA_CHECK(cudaMemcpy(host_ids.data(), argmax_ids.data, argmax_ids.bytes(),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_targets.data(), targets.data, targets.bytes(),
                              cudaMemcpyDeviceToHost));
        const auto* base = static_cast<const std::uint8_t*>(logits.data);
        const auto to_float = [](std::uint16_t bits) {
            const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
            float value              = 0.0F;
            std::memcpy(&value, &wide, sizeof(value));
            return value;
        };
        for (int column = 0; column < count; ++column) {
            if (host_targets[column] != wanted) continue;
            std::uint16_t target_bits = 0;
            std::uint16_t max_bits    = 0;
            CUDA_CHECK(cudaMemcpy(
                &target_bits, base + (static_cast<std::size_t>(column) * vocabulary + wanted) * 2,
                2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&max_bits,
                                  base + (static_cast<std::size_t>(column) * vocabulary +
                                          static_cast<std::size_t>(host_ids[column])) * 2,
                                  2, cudaMemcpyDeviceToHost));
            std::cerr << "score debug: sharded=" << (sharded_ != nullptr ? 1 : 0)
                      << " col=" << column << " target=" << wanted
                      << " target_logit=" << to_float(target_bits) << " argmax_id="
                      << host_ids[column] << " argmax_logit=" << to_float(max_bits) << '\n';
        }
    }
    Tensor scores = frame.score_logprobs.slice(0, 0, count);
    ops::target_logprobs(logits, targets, execution::dimension(parameters.profile->public_tokens),
                         scores, device.execution_view());
    CUDA_CHECK(cudaMemcpyAsync(frame.host_score_logprobs.data, scores.data, scores.bytes(),
                               cudaMemcpyDeviceToHost, device.stream()));
    gpu_timer.record_result();
    result_ready.record(device.stream());
    operation.stage = ComputeStage::Readback;
}

} // namespace ninfer::models::qwen3_5::detail
