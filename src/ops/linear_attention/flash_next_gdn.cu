#include "ninfer/ops/flash_next_gdn.h"

#include "ops/flash_next_work.h"

#include "core/layout.h"
#include "core/device.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/linear.h"
#include "ops/common/math.cuh"
#include "ops/linear_attention/gated_delta_net/launch.h"

#include <cuda_bf16.h>

#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr int kHidden = 2560;
constexpr int kHeads = 48;
constexpr int kQkHeads = 16;
constexpr int kDim = 128;
constexpr int kQk = kQkHeads * kDim;
constexpr int kValue = kHeads * kDim;
constexpr int kConvolution = 2 * kQk + kValue;

__global__ void project_control_gating_kernel(const __nv_bfloat16* input,
                                              const __nv_bfloat16* a_weight,
                                              const __nv_bfloat16* b_weight,
                                              const float* a_log, const float* dt_bias,
                                              float* g, float* beta) {
    const int row = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    float av = 0.0F;
    float bv = 0.0F;
    for (int k = static_cast<int>(threadIdx.x); k < kHidden; k += blockDim.x) {
        const float x = __bfloat162float(input[k + static_cast<std::int64_t>(kHidden) * token]);
        av = fmaf(__bfloat162float(a_weight[k + static_cast<std::int64_t>(kHidden) * row]), x, av);
        bv = fmaf(__bfloat162float(b_weight[k + static_cast<std::int64_t>(kHidden) * row]), x, bv);
    }
    for (int offset = 16; offset != 0; offset >>= 1) {
        av += __shfl_down_sync(0xffffffffU, av, offset);
        bv += __shfl_down_sync(0xffffffffU, bv, offset);
    }
    __shared__ float partial_a[8];
    __shared__ float partial_b[8];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) {
        partial_a[warp] = av;
        partial_b[warp] = bv;
    }
    __syncthreads();
    if (warp == 0) {
        av = lane < 8 ? partial_a[lane] : 0.0F;
        bv = lane < 8 ? partial_b[lane] : 0.0F;
        for (int offset = 16; offset != 0; offset >>= 1) {
            av += __shfl_down_sync(0xffffffffU, av, offset);
            bv += __shfl_down_sync(0xffffffffU, bv, offset);
        }
        if (lane == 0) {
            // Preserve the prior materialized BF16 projection boundary before gate preparation.
            const float represented_a = __bfloat162float(__float2bfloat16_rn(av));
            const float represented_b = __bfloat162float(__float2bfloat16_rn(bv));
            const int index = row + kHeads * token;
            g[index] = -expf(a_log[row]) * softplus(represented_a + dt_bias[row]);
            beta[index] = sigmoid(represented_b);
        }
    }
}

__global__ void conv_replay_record_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* states, const std::int32_t* valid_columns,
    const std::int32_t* source_slots, __nv_bfloat16* records,
    __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v,
    int width, int batch, std::int64_t slot_stride) {
    const int lane = static_cast<int>(blockIdx.y);
    for (int channel = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
         channel < kConvolution; channel += static_cast<int>(blockDim.x) * gridDim.x) {
        const __nv_bfloat16* source =
            states + static_cast<std::int64_t>(source_slots[lane]) * slot_stride;
        __nv_bfloat16 s0 = source[channel];
        __nv_bfloat16 s1 = source[kConvolution + channel];
        __nv_bfloat16 s2 = source[2LL * kConvolution + channel];
        const float w0 = __bfloat162float(weight[channel]);
        const float w1 = __bfloat162float(weight[kConvolution + channel]);
        const float w2 = __bfloat162float(weight[2LL * kConvolution + channel]);
        const float w3 = __bfloat162float(weight[3LL * kConvolution + channel]);
        const int valid = valid_columns[lane];
        for (int column = 0; column < width; ++column) {
            const std::int64_t token = static_cast<std::int64_t>(lane) * width + column;
            const std::int64_t index = channel + static_cast<std::int64_t>(kConvolution) * token;
            if (column >= valid) {
                if (channel < kQk) {
                    q[channel + static_cast<std::int64_t>(kQk) * token] =
                        __float2bfloat16(0.0F);
                } else if (channel < 2 * kQk) {
                    k[channel - kQk + static_cast<std::int64_t>(kQk) * token] =
                        __float2bfloat16(0.0F);
                } else {
                    v[channel - 2 * kQk + static_cast<std::int64_t>(kValue) * token] =
                        __float2bfloat16(0.0F);
                }
                continue;
            }
            const __nv_bfloat16 current = input[index];
            records[index] = current;
            float value = w0 * __bfloat162float(s0);
            value = fmaf(w1, __bfloat162float(s1), value);
            value = fmaf(w2, __bfloat162float(s2), value);
            value = fmaf(w3, __bfloat162float(current), value);
            const __nv_bfloat16 convolved =
                __float2bfloat16_rn(value / (1.0F + expf(-value)));
            if (channel < kQk) {
                q[channel + static_cast<std::int64_t>(kQk) * token] = convolved;
            } else if (channel < 2 * kQk) {
                k[channel - kQk + static_cast<std::int64_t>(kQk) * token] = convolved;
            } else {
                v[channel - 2 * kQk + static_cast<std::int64_t>(kValue) * token] = convolved;
            }
            s0 = s1;
            s1 = s2;
            s2 = current;
        }
    }
}

enum class WeightFormats { Bf16, Bf16OrFp8 };

bool fp8_row_weight(const Weight& weight) {
    return weight.qtype == QType::FP8_E4M3FN_ROW_BF16 && weight.layout == QuantLayout::RowScale;
}

void require_weight(const Weight& weight, int rows, int columns, const char* label,
                    WeightFormats formats = WeightFormats::Bf16) {
    const bool bf16 = weight.qtype == QType::BF16 && weight.layout == QuantLayout::Contiguous;
    const bool allowed = bf16 || (formats == WeightFormats::Bf16OrFp8 && fp8_row_weight(weight));
    if (!allowed || weight.qdata == nullptr || weight.n != rows || weight.k != columns) {
        throw std::invalid_argument(label);
    }
}

void validate(const Tensor& input, const FlashNextGdnWeights& weights,
              const Tensor& conv_in, const Tensor& conv_out, const Tensor& recurrent_in,
              const Tensor& recurrent_out, const Tensor& destination) {
    const int tokens = input.ne[1];
    if (tokens <= 0 || input.dtype != DType::BF16 || !input.is_contiguous() ||
        input.ne[0] != kHidden || input.ne[2] != 1 || input.ne[3] != 1 ||
        destination.dtype != DType::BF16 || !destination.is_contiguous() ||
        destination.ne[0] != kHidden || destination.ne[1] != tokens ||
        conv_in.dtype != DType::BF16 || !conv_in.is_contiguous() || conv_in.ne[0] != kConvolution ||
        conv_in.ne[1] != 3 || conv_out.dtype != DType::BF16 || !conv_out.is_contiguous() ||
        conv_out.ne[0] != kConvolution || conv_out.ne[1] != 3 ||
        recurrent_in.dtype != DType::FP32 || !recurrent_in.is_contiguous() ||
        recurrent_in.ne[0] != kDim || recurrent_in.ne[1] != kDim || recurrent_in.ne[2] != kHeads ||
        recurrent_out.dtype != DType::FP32 || !recurrent_out.is_contiguous() ||
        recurrent_out.ne[0] != kDim || recurrent_out.ne[1] != kDim ||
        recurrent_out.ne[2] != kHeads || weights.a_log.dtype != DType::FP32 ||
        weights.a_log.ne[0] != kHeads || weights.dt_bias.dtype != DType::FP32 ||
        weights.dt_bias.ne[0] != kHeads || weights.convolution.dtype != DType::BF16 ||
        weights.convolution.ne[0] != kConvolution || weights.convolution.ne[1] != 4 ||
        weights.convolution.ne[2] != 1 ||
        weights.norm.dtype != DType::BF16 || weights.norm.ne[0] != kDim) {
        throw std::invalid_argument("flash_next_gdn: invalid exact tensor geometry");
    }
    require_weight(weights.a_projection, kHeads, kHidden,
                   "flash_next_gdn: invalid a projection");
    require_weight(weights.b_projection, kHeads, kHidden,
                   "flash_next_gdn: invalid b projection");
    require_weight(weights.query_key_value, kConvolution, kHidden,
                   "flash_next_gdn: invalid qkv projection", WeightFormats::Bf16OrFp8);
    require_weight(weights.output_gate, kValue, kHidden,
                   "flash_next_gdn: invalid output gate projection", WeightFormats::Bf16OrFp8);
    require_weight(weights.output, kHidden, kValue, "flash_next_gdn: invalid output projection",
                   WeightFormats::Bf16OrFp8);
}

} // namespace

std::size_t flash_next_gdn_workspace_capacity_bytes(std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("Flash-Next GDN tokens must be positive"); }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {kHeads, tokens});
    (void)layout.alloc(DType::BF16, {kHeads, tokens});
    (void)layout.alloc(DType::FP32, {kHeads, tokens});
    (void)layout.alloc(DType::FP32, {kHeads, tokens});
    (void)layout.alloc(DType::BF16, {kConvolution, tokens});
    (void)layout.alloc(DType::BF16, {kValue, tokens});
    (void)layout.alloc(DType::BF16, {kQk, tokens});
    (void)layout.alloc(DType::BF16, {kQk, tokens});
    (void)layout.alloc(DType::BF16, {kValue, tokens});
    (void)layout.alloc(DType::BF16, {kValue, tokens});
    (void)layout.alloc(DType::BF16, {kValue, tokens});
    (void)layout.alloc_bytes(gated_delta_net_workspace_capacity_bytes(
        kQkHeads, kHeads, true, tokens, tokens));
    return layout.peak_bytes(1);
}

void flash_next_gdn(const Tensor& input, const FlashNextGdnWeights& weights,
                    const Tensor& convolution_state_in, Tensor& convolution_state_out,
                    const Tensor& recurrent_state_in, Tensor& recurrent_state_out,
                    Tensor& destination, WorkspaceArena& workspace, cudaStream_t stream,
                    Bf16GemmContext* bf16_gemm) {
    NINFER_PERF_SCOPE("gdn.prefill", input.ne[1], 1, 0,
                       flash_next_work::gdn(input.ne[1], 1, false, true,
                                            fp8_row_weight(weights.query_key_value)));

    validate(input, weights, convolution_state_in, convolution_state_out, recurrent_state_in,
             recurrent_state_out, destination);
    const int tokens = input.ne[1];
    auto scope = workspace.scope();
    Tensor a = workspace.alloc(DType::BF16, {kHeads, tokens});
    Tensor b = workspace.alloc(DType::BF16, {kHeads, tokens});
    Tensor g = workspace.alloc(DType::FP32, {kHeads, tokens});
    Tensor beta = workspace.alloc(DType::FP32, {kHeads, tokens});
    if (tokens > 16) {
        linear(input, weights.a_projection, a, stream, bf16_gemm);
        linear(input, weights.b_projection, b, stream, bf16_gemm);
        gdn_gating(a, b, weights.a_log, weights.dt_bias, g, beta, stream);
    } else {
        project_control_gating_kernel<<<dim3(kHeads, static_cast<unsigned int>(tokens)), 256, 0,
                                        stream>>>(
            static_cast<const __nv_bfloat16*>(input.data),
            static_cast<const __nv_bfloat16*>(weights.a_projection.qdata),
            static_cast<const __nv_bfloat16*>(weights.b_projection.qdata),
            static_cast<const float*>(weights.a_log.data),
            static_cast<const float*>(weights.dt_bias.data), static_cast<float*>(g.data),
            static_cast<float*>(beta.data));
        CUDA_CHECK(cudaGetLastError());
    }

    Tensor projected = workspace.alloc(DType::BF16, {kConvolution, tokens});
    Tensor z = workspace.alloc(DType::BF16, {kValue, tokens});
    linear(input, weights.query_key_value, projected, stream, bf16_gemm);
    linear(input, weights.output_gate, z, stream, bf16_gemm);
    Tensor q = workspace.alloc(DType::BF16, {kQk, tokens});
    Tensor k = workspace.alloc(DType::BF16, {kQk, tokens});
    Tensor v = workspace.alloc(DType::BF16, {kValue, tokens});
    Tensor convolution_weight = weights.convolution;
    causal_conv1d_silu_split(projected, convolution_weight, convolution_state_in,
                             convolution_state_out, q, k, v, stream);

    Tensor recurrent = workspace.alloc(DType::BF16, {kValue, tokens});
    Tensor q_heads = q.view({kDim, kQkHeads, tokens});
    Tensor k_heads = k.view({kDim, kQkHeads, tokens});
    Tensor v_heads = v.view({kDim, kHeads, tokens});
    Tensor recurrent_heads = recurrent.view({kDim, kHeads, tokens});
    gated_delta_net(q_heads, k_heads, v_heads, g, beta, 0.08838834764831845F, true,
                    workspace, recurrent_state_in, recurrent_state_out,
                    recurrent_heads, stream);
    Tensor normalized = workspace.alloc(DType::BF16, {kValue, tokens});
    Tensor z_heads = z.view({kDim, kHeads, tokens});
    Tensor normalized_heads = normalized.view({kDim, kHeads, tokens});
    sigmoid_gated_rmsnorm(recurrent_heads, weights.norm, z_heads, 1.0e-6F,
                          normalized_heads, stream);
    linear(normalized, weights.output, destination, stream, bf16_gemm);
}

void flash_next_gdn_batch_update(const Tensor& input, const FlashNextGdnWeights& weights,
                                 Tensor& convolution_states, Tensor& recurrent_states,
                                 const Tensor& source_slots, const Tensor& destination_slots,
                                 Tensor& destination, WorkspaceArena& workspace,
                                 cudaStream_t stream) {
    NINFER_PERF_SCOPE("gdn.update", input.ne[1], input.ne[1], 0,
                       flash_next_work::gdn(input.ne[1], input.ne[1], false, false,
                                            fp8_row_weight(weights.query_key_value)));

    const int batch = input.ne[1];
    if (batch <= 0 || batch > 8 || input.dtype != DType::BF16 || !input.is_contiguous() ||
        input.ne[0] != kHidden || destination.dtype != DType::BF16 ||
        !destination.is_contiguous() || destination.ne[0] != kHidden ||
        destination.ne[1] != batch || convolution_states.dtype != DType::BF16 ||
        !convolution_states.is_contiguous() || convolution_states.ne[0] != kConvolution ||
        convolution_states.ne[1] != 3 || recurrent_states.dtype != DType::FP32 ||
        !recurrent_states.is_contiguous() || recurrent_states.ne[0] != kDim ||
        recurrent_states.ne[1] != kDim || recurrent_states.ne[2] != kHeads ||
        source_slots.dtype != DType::I32 || destination_slots.dtype != DType::I32 ||
        source_slots.ne[0] != batch || destination_slots.ne[0] != batch) {
        throw std::invalid_argument("flash_next_gdn_batch_update: invalid exact geometry");
    }
    validate(input, weights, convolution_states.slice(2, 0, 1).view({kConvolution, 3}),
             convolution_states.slice(2, 0, 1).view({kConvolution, 3}),
             recurrent_states.slice(3, 0, 1).view({kDim, kDim, kHeads}),
             recurrent_states.slice(3, 0, 1).view({kDim, kDim, kHeads}), destination);

    auto scope = workspace.scope();
    Tensor a = workspace.alloc(DType::BF16, {kHeads, batch});
    Tensor b = workspace.alloc(DType::BF16, {kHeads, batch});
    Tensor g = workspace.alloc(DType::FP32, {kHeads, batch});
    Tensor beta = workspace.alloc(DType::FP32, {kHeads, batch});
    project_control_gating_kernel<<<dim3(kHeads, static_cast<unsigned int>(batch)), 256, 0,
                                    stream>>>(
        static_cast<const __nv_bfloat16*>(input.data),
        static_cast<const __nv_bfloat16*>(weights.a_projection.qdata),
        static_cast<const __nv_bfloat16*>(weights.b_projection.qdata),
        static_cast<const float*>(weights.a_log.data),
        static_cast<const float*>(weights.dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data));
    CUDA_CHECK(cudaGetLastError());

    Tensor projected = workspace.alloc(DType::BF16, {kConvolution, batch});
    Tensor z = workspace.alloc(DType::BF16, {kValue, batch});
    linear(input, weights.query_key_value, projected, stream);
    linear(input, weights.output_gate, z, stream);
    Tensor convolved = workspace.alloc(DType::BF16, {kConvolution, batch});
    Tensor convolution_weight = weights.convolution;
    Tensor projected_batch = projected.view({kConvolution, 1, batch});
    Tensor convolved_batch = convolved.view({kConvolution, 1, batch});
    causal_conv1d_silu_snapshot(projected_batch, convolution_weight, convolution_states,
                                Tensor{}, source_slots, destination_slots, convolved_batch,
                                stream);
    Tensor recurrent = workspace.alloc(DType::BF16, {kValue, batch});
    Tensor g_batch = g.view({kHeads, 1, batch});
    Tensor beta_batch = beta.view({kHeads, 1, batch});
    Tensor recurrent_batch = recurrent.view({kDim, kHeads, 1, batch});
    detail::gated_delta_net::launch_recurrent_batch_update_packed_qkv(
        convolved, kQkHeads, kHeads, g_batch, beta_batch, 0.08838834764831845F,
        recurrent_states, source_slots, destination_slots, recurrent_batch, stream);
    Tensor normalized = workspace.alloc(DType::BF16, {kValue, batch});
    Tensor z_heads = z.view({kDim, kHeads, batch});
    Tensor recurrent_heads = recurrent.view({kDim, kHeads, batch});
    Tensor normalized_heads = normalized.view({kDim, kHeads, batch});
    sigmoid_gated_rmsnorm(recurrent_heads, weights.norm, z_heads, 1.0e-6F,
                          normalized_heads, stream);
    linear(normalized, weights.output, destination, stream);
}

void flash_next_gdn_replay_record(const Tensor& input, const FlashNextGdnWeights& weights,
                                  const Tensor& convolution_states,
                                  const Tensor& recurrent_states,
                                  const Tensor& valid_columns, const Tensor& source_slots,
                                  GdnReplayRecordLayer records, Tensor& destination,
                                  WorkspaceArena& workspace, cudaStream_t stream) {
    NINFER_PERF_SCOPE("gdn.record", input.ne[1], records.conv.ne[2], 0,
                       flash_next_work::gdn(input.ne[1], records.conv.ne[2], true, false,
                                            fp8_row_weight(weights.query_key_value)));

    const int width = records.conv.ne[1];
    const int batch = records.conv.ne[2];
    const int tokens = width * batch;
    if (width < 2 || width > 16 || batch <= 0 || batch > 8 || input.ne[1] != tokens ||
        input.dtype != DType::BF16 || !input.is_contiguous() || input.ne[0] != kHidden ||
        destination.dtype != DType::BF16 || !destination.is_contiguous() ||
        destination.ne[0] != kHidden || destination.ne[1] != tokens ||
        convolution_states.dtype != DType::BF16 || !convolution_states.is_contiguous() ||
        convolution_states.ne[0] != kConvolution || convolution_states.ne[1] != 3 ||
        recurrent_states.dtype != DType::FP32 || !recurrent_states.is_contiguous() ||
        recurrent_states.ne[0] != kDim || recurrent_states.ne[1] != kDim ||
        recurrent_states.ne[2] != kHeads || valid_columns.dtype != DType::I32 ||
        valid_columns.ne[0] != batch || source_slots.dtype != DType::I32 ||
        source_slots.ne[0] != batch || records.conv.dtype != DType::BF16 ||
        records.conv.ne[0] != kConvolution || records.key.dtype != DType::BF16 ||
        records.key.ne[0] != kDim || records.key.ne[1] != kQkHeads ||
        records.key.ne[2] != width || records.key.ne[3] != batch ||
        records.value.dtype != DType::BF16 || records.value.ne[0] != kDim ||
        records.value.ne[1] != kHeads || records.value.ne[2] != width ||
        records.value.ne[3] != batch || records.gate.dtype != DType::FP32 ||
        records.gate.ne[0] != 2 || records.gate.ne[1] != kHeads ||
        records.gate.ne[2] != width || records.gate.ne[3] != batch) {
        throw std::invalid_argument("flash_next_gdn_replay_record: invalid exact geometry");
    }
    validate(input, weights,
             convolution_states.slice(2, 0, 1).view({kConvolution, 3}),
             convolution_states.slice(2, 0, 1).view({kConvolution, 3}),
             recurrent_states.slice(3, 0, 1).view({kDim, kDim, kHeads}),
             recurrent_states.slice(3, 0, 1).view({kDim, kDim, kHeads}), destination);

    auto scope = workspace.scope();
    Tensor a = workspace.alloc(DType::BF16, {kHeads, tokens});
    Tensor b = workspace.alloc(DType::BF16, {kHeads, tokens});
    Tensor g = workspace.alloc(DType::FP32, {kHeads, tokens});
    Tensor beta = workspace.alloc(DType::FP32, {kHeads, tokens});
    project_control_gating_kernel<<<dim3(kHeads, static_cast<unsigned int>(tokens)), 256, 0,
                                    stream>>>(
        static_cast<const __nv_bfloat16*>(input.data),
        static_cast<const __nv_bfloat16*>(weights.a_projection.qdata),
        static_cast<const __nv_bfloat16*>(weights.b_projection.qdata),
        static_cast<const float*>(weights.a_log.data),
        static_cast<const float*>(weights.dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data));
    CUDA_CHECK(cudaGetLastError());
    Tensor projected = workspace.alloc(DType::BF16, {kConvolution, tokens});
    Tensor z = workspace.alloc(DType::BF16, {kValue, tokens});
    linear(input, weights.query_key_value, projected, stream);
    linear(input, weights.output_gate, z, stream);
    Tensor q = workspace.alloc(DType::BF16, {kQk, tokens});
    Tensor k = workspace.alloc(DType::BF16, {kQk, tokens});
    Tensor v = workspace.alloc(DType::BF16, {kValue, tokens});
    const dim3 conv_grid(40, static_cast<unsigned int>(batch));
    conv_replay_record_kernel<<<conv_grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data),
        static_cast<const __nv_bfloat16*>(weights.convolution.data),
        static_cast<const __nv_bfloat16*>(convolution_states.data),
        static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<const std::int32_t*>(source_slots.data),
        static_cast<__nv_bfloat16*>(records.conv.data),
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(v.data), width, batch,
        static_cast<std::int64_t>(kConvolution) * 3);
    CUDA_CHECK(cudaGetLastError());
    Tensor recurrent = workspace.alloc(DType::BF16, {kValue, tokens});
    Tensor q_batch = q.view({kDim, kQkHeads, width, batch});
    Tensor k_batch = k.view({kDim, kQkHeads, width, batch});
    Tensor v_batch = v.view({kDim, kHeads, width, batch});
    Tensor g_batch = g.view({kHeads, width, batch});
    Tensor beta_batch = beta.view({kHeads, width, batch});
    Tensor recurrent_batch = recurrent.view({kDim, kHeads, width, batch});
    gated_delta_net_replay_record(q_batch, k_batch, v_batch, g_batch, beta_batch,
                                  0.08838834764831845F, recurrent_states, valid_columns,
                                  source_slots, records.key, records.value, records.gate,
                                  recurrent_batch, stream);
    Tensor normalized = workspace.alloc(DType::BF16, {kValue, tokens});
    Tensor recurrent_heads = recurrent.view({kDim, kHeads, tokens});
    Tensor z_heads = z.view({kDim, kHeads, tokens});
    Tensor normalized_heads = normalized.view({kDim, kHeads, tokens});
    sigmoid_gated_rmsnorm(recurrent_heads, weights.norm, z_heads, 1.0e-6F,
                          normalized_heads, stream);
    linear(normalized, weights.output, destination, stream);
}

} // namespace ninfer::ops
