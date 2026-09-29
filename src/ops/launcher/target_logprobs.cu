// Implements: include/ninfer/ops/target_logprobs.h
// Match: wrapper-validated contiguous tensors and valid vocabulary rows.
// Algorithm assumptions: one independent CTA per column; no global workspace. The ranking lists live
// in shared memory, so reporting them is a compile-time choice and an unranked call allocates none.
#include "ops/launcher/target_logprobs.h"

#include "core/device.h"
#include "ops/kernel/target_logprobs.cuh"

namespace ninfer::ops::detail {

namespace {

template <bool ReportRanking>
void launch_columns(unsigned int columns, const Tensor& logits, const Tensor& target_ids,
                    float* output, std::int32_t valid_rows, std::int32_t top_k,
                    std::int32_t* top_ids, float* top_logprobs, cudaStream_t stream) {
    target_logprobs_kernel<kTargetLogprobsBlock, ReportRanking>
        <<<columns, kTargetLogprobsBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data),
            static_cast<const std::int32_t*>(target_ids.data), output, valid_rows,
            static_cast<std::int32_t>(logits.ne[0]), top_k, top_ids, top_logprobs);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void target_logprobs_launch(const Tensor& logits, const Tensor& target_ids, std::int32_t valid_rows,
                            Tensor& output, Tensor* top_ids, Tensor* top_logprobs,
                            cudaStream_t stream) {
    const unsigned int columns = static_cast<unsigned int>(logits.ne[1]);
    float* output_values       = static_cast<float*>(output.data);
    const std::int32_t top_k =
        top_ids != nullptr ? static_cast<std::int32_t>(top_ids->ne[0]) : static_cast<std::int32_t>(0);
    if (top_k > 0) {
        launch_columns<true>(columns, logits, target_ids, output_values, valid_rows, top_k,
                             static_cast<std::int32_t*>(top_ids->data),
                             static_cast<float*>(top_logprobs->data), stream);
        return;
    }
    launch_columns<false>(columns, logits, target_ids, output_values, valid_rows, 0, nullptr,
                          nullptr, stream);
}

} // namespace ninfer::ops::detail
