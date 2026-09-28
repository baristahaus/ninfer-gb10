// Implements: include/ninfer/ops/target_logprobs.h
// Match: wrapper-validated contiguous tensors and valid vocabulary rows.
// Algorithm assumptions: one independent CTA per column; no global workspace.
#include "ops/launcher/target_logprobs.h"

#include "core/device.h"
#include "ops/kernel/target_logprobs.cuh"

namespace ninfer::ops::detail {

void target_logprobs_launch(const Tensor& logits, const Tensor& target_ids, std::int32_t valid_rows,
                            const TargetLogprobOptions& options, Tensor& output, Tensor* top_ids,
                            Tensor* top_logprobs, cudaStream_t stream) {
    const auto columns = static_cast<unsigned int>(logits.ne[1]);
    const auto top_k =
        top_ids != nullptr ? static_cast<std::int32_t>(top_ids->ne[0]) : static_cast<std::int32_t>(0);
    target_logprobs_kernel<kTargetLogprobsBlock>
        <<<columns, kTargetLogprobsBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data),
            static_cast<const std::int32_t*>(target_ids.data), static_cast<float*>(output.data),
            valid_rows, logits.ne[0], options.configs, options.penalty_overlay,
            options.overlay_rows, top_k,
            top_ids != nullptr ? static_cast<std::int32_t*>(top_ids->data) : nullptr,
            top_logprobs != nullptr ? static_cast<float*>(top_logprobs->data) : nullptr);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
