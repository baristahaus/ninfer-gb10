#include "ninfer/ops/gated_delta_net.h"
#include "ops/linear_attention/gated_delta_net/replay/validation.h"
#include "ops/linear_attention/gated_delta_net/replay/launch.h"
#include <array>
#include <span>

namespace ninfer::ops {
namespace {
using namespace detail::gated_delta_net::replay;

void validate_replay_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, const Tensor& states,
                            const Tensor& valid_columns, const Tensor& initial_slots,
                            const Tensor& key_record, const Tensor& value_record,
                            const Tensor& gate_record, const Tensor& out, Device device) {
    constexpr const char* kOp      = "gated_delta_net_replay_record";
    const std::int32_t qk_heads    = q.ne[1];
    const std::int32_t value_heads = v.ne[1];
    const std::int32_t width       = q.ne[2];
    const std::int32_t rows        = q.ne[3];
    const bool registered_heads    = (qk_heads == 16 && (value_heads == 48 || value_heads == 32)) ||
                                  (qk_heads == 8 && value_heads == 24);
    if (!registered_heads || width < 2 || width > 16 || rows <= 0 || rows > kMaximumRows) {
        throw std::invalid_argument(std::string(kOp) + ": unsupported geometry");
    }
    if (states.ne[3] <= 0) {
        throw std::invalid_argument(std::string(kOp) + ": state slot count must be positive");
    }

    common::operand::tensor(q, DType::BF16, {kStateDim, qk_heads, width, rows}, 8, kOp, "q",
                            device);
    common::operand::tensor(k, DType::BF16, {kStateDim, qk_heads, width, rows}, 8, kOp, "k",
                            device);
    common::operand::tensor(v, DType::BF16, {kStateDim, value_heads, width, rows}, 2, kOp, "v",
                            device);
    common::operand::tensor(g, DType::FP32, {value_heads, width, rows}, 4, kOp, "g", device);
    common::operand::tensor(beta, DType::FP32, {value_heads, width, rows}, 4, kOp, "beta", device);
    common::operand::slotted_tensor(states, DType::FP32,
                                    {kStateDim, kStateDim, value_heads, states.ne[3]}, 3, 16, kOp,
                                    "states", device);
    common::operand::tensor(initial_slots, DType::I32, {rows}, 4, kOp, "initial state slots",
                            device);
    if (valid_columns.data != nullptr) {
        common::operand::tensor(valid_columns, DType::I32, {rows}, 4, kOp, "valid columns", device);
    }
    common::operand::tensor(key_record, DType::BF16, {kStateDim, qk_heads, width, rows}, 8, kOp,
                            "key record", device);
    common::operand::tensor(value_record, DType::BF16, {kStateDim, value_heads, width, rows}, 2,
                            kOp, "value record", device);
    common::operand::tensor(gate_record, DType::FP32, {2, value_heads, width, rows}, 8, kOp,
                            "gate record", device);
    common::operand::tensor(out, DType::BF16, {kStateDim, value_heads, width, rows}, 2, kOp, "out",
                            device);
    detail::gated_delta_net::require_scale(scale, kOp);

    const std::array<const Tensor*, 12> tensors{&q,
                                                &k,
                                                &v,
                                                &g,
                                                &beta,
                                                &states,
                                                &valid_columns,
                                                &initial_slots,
                                                &key_record,
                                                &value_record,
                                                &gate_record,
                                                &out};
    std::array<common::operand::Blocks, 12> ranges{};
    std::size_t count = 0;
    for (const Tensor* tensor : tensors) {
        if (tensor->data != nullptr) ranges[count++] = common::operand::blocks(*tensor, kOp);
    }
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (common::operand::overlaps(ranges[i], ranges[j]))
                throw std::invalid_argument("gated_delta_net_replay_record: tensors overlap");
}

} // namespace

void gated_delta_net_replay_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& ssm_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, GdnRecurrenceRecords records,
                                   Tensor& out, CudaExecutionView execution) {
    const auto stream = execution.stream();
    validate_replay_record(q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots,
                           records.key, records.value, records.gate, out, execution.device());
    detail::gated_delta_net::launch_recurrent_record(
        q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots, records.key,
        records.value, records.gate, out, stream);
}

void gated_delta_net_replay_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& ssm_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, GdnRecurrenceRecords records,
                                   Tensor& out, cudaStream_t stream) {
    validate_replay_record(q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots,
                           records.key, records.value, records.gate, out, q.device);
    detail::gated_delta_net::launch_recurrent_record(
        q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots, records.key,
        records.value, records.gate, out, stream);
}

void gated_delta_net_fold_replay_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                        const Tensor& g, const Tensor& beta, float scale,
                                        Tensor& ssm_states, const Tensor& valid_columns,
                                        const Tensor& initial_state_slots,
                                        GdnRecurrenceRecords records, Tensor& out,
                                        const GdnPendingFold& pending, cudaStream_t stream) {
    constexpr const char* kOp = "gated_delta_net_fold_replay_record";
    const auto device = q.device;
    validate_replay_record(q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots,
                           records.key, records.value, records.gate, out, device);
    const std::int32_t qk_heads     = q.ne[1];
    const std::int32_t value_heads  = v.ne[1];
    const std::int32_t width        = q.ne[2];
    const std::int32_t pending_rows = pending.rows.ne[1];
    if (pending_rows <= 0 || pending_rows > detail::gated_delta_net::replay::kMaximumRows) {
        throw std::invalid_argument(std::string(kOp) + ": pending row count must be in [1,8]");
    }
    namespace operand = common::operand;
    operand::tensor(pending.rows, DType::I32, {4, pending_rows}, 16, kOp, "pending rows", device);
    operand::tensor(pending.key_record, DType::BF16, {detail::gated_delta_net::kStateDim, qk_heads,
                                                      width, pending_rows},
                    8, kOp, "pending key record", device);
    operand::tensor(pending.value_record, DType::BF16,
                    {detail::gated_delta_net::kStateDim, value_heads, width, pending_rows}, 2, kOp,
                    "pending value record", device);
    operand::tensor(pending.gate_record, DType::FP32, {2, value_heads, width, pending_rows}, 8, kOp,
                    "pending gate record", device);
    const std::array<const Tensor*, 7> written{&ssm_states,  &records.key,   &records.value,
                                               &records.gate, &out,           &pending.key_record,
                                               &pending.rows};
    std::array<operand::Blocks, 9> ranges{};
    std::size_t count = 0;
    for (const Tensor* tensor : written) { ranges[count++] = operand::blocks(*tensor, kOp); }
    ranges[count++] = operand::blocks(pending.value_record, kOp);
    ranges[count++] = operand::blocks(pending.gate_record, kOp);
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (operand::overlaps(ranges[i], ranges[j]))
                throw std::invalid_argument(
                    std::string(kOp) + ": pending records must be a separate copy of the "
                                        "previous round and must not overlap the operands");
    static_assert(sizeof(detail::gated_delta_net::GdnReplayFoldKernelRow) == 4 * sizeof(std::int32_t));
    detail::gated_delta_net::launch_recurrent_fold_record(
        q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots, records.key,
        records.value, records.gate, out,
        static_cast<const detail::gated_delta_net::GdnReplayFoldKernelRow*>(pending.rows.data),
        pending_rows, pending.key_record, pending.value_record, pending.gate_record, stream);
}

} // namespace ninfer::ops
