#include "ninfer/ops/activation_steering.h"
#include "core/device.h"
#include <cuda_bf16.h>
#include <stdexcept>

namespace ninfer::ops {
namespace {
__global__ void capture(const __nv_bfloat16* hyper, const __nv_bfloat16* normalized,
                        const __nv_bfloat16* gates, const __nv_bfloat16* mixed,
                        const int* positions, const int* ids, const int* valid,
                        const ActivationDevice* c, int layer, int width, bool speculative_columns) {
    const int row = blockIdx.x, site = blockIdx.y;
    const auto control = c->rows[row];
    if (!control.capture || !c->samples || site > c->completion_capacity) return;
    const int local = site != 0 && speculative_columns ? site - 1 : valid[row] - 1;
    if (local < 0 || local >= width || local >= valid[row]) return;
    const int token    = row * width + local;
    const int position = positions[token];
    if (site == 0 && position != control.prompt_last) return;
    const int sample = (control.lane * (1 + c->completion_capacity) + site) * 48 + layer;
    if (!threadIdx.x) {
        auto& check    = c->checks[sample];
        check.fires    = site == 0 ? check.fires + 1 : 1;
        check.position = position;
        check.token    = ids[token];
        check.pad      = local >= valid[row];
    }
    float* out = c->samples + static_cast<std::int64_t>(sample) * kActivationElements;
    for (int d = threadIdx.x; d < 10240; d += blockDim.x) {
        out[d]         = __bfloat162float(hyper[token * 10240 + d]);
        out[10240 + d] = __bfloat162float(normalized[token * 10240 + d]);
        out[20480 + d] = 1.0F / (1.0F + expf(-__bfloat162float(gates[token * 10240 + d])));
    }
    for (int d = threadIdx.x; d < 2560; d += blockDim.x)
        out[30720 + d] = __bfloat162float(mixed[token * 2560 + d]);
}
} // namespace

void activation_capture(const Tensor& hyper, const Tensor& normalized, const Tensor& gates,
                        const Tensor& mixed, const Tensor& positions, const Tensor& ids,
                        const Tensor& valid, const ActivationDevice* control, int layer, int width,
                        cudaStream_t stream, bool speculative_columns) {
    if (!control) return;
    if (width <= 0 || (speculative_columns && width > kActivationColumns))
        throw std::invalid_argument("activation_capture: verification_width check failed");
    for (const auto* t : {&hyper, &normalized, &gates, &mixed})
        if (t->dtype != DType::BF16 || !t->is_contiguous() || t->ne[1] != hyper.ne[1] ||
            t->ne[0] != (t == &mixed ? 2560 : 10240))
            throw std::invalid_argument("activation_capture: shape_dtype check failed");
    if (positions.dtype != DType::I32 || ids.dtype != DType::I32 || valid.dtype != DType::I32 ||
        positions.numel() != hyper.ne[1] || ids.numel() != hyper.ne[1])
        throw std::invalid_argument("activation_capture: index_shape_dtype check failed");
    capture<<<dim3(hyper.ne[1] / width, speculative_columns ? width + 1 : 2), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hyper.data),
        static_cast<const __nv_bfloat16*>(normalized.data),
        static_cast<const __nv_bfloat16*>(gates.data),
        static_cast<const __nv_bfloat16*>(mixed.data), static_cast<const int*>(positions.data),
        static_cast<const int*>(ids.data), static_cast<const int*>(valid.data), control, layer,
        width, speculative_columns);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
