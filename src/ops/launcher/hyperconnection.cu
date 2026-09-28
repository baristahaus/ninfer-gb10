#include "ninfer/ops/hyperconnection.h"

#include "ops/flash_next_work.h"

#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/bf16/flash_next/bf16_launch.h"
#include "ops/linear/fp8/fp8_flash_next.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr int kStreams = 4;
constexpr int kHidden = 2560;
constexpr int kHyper = kStreams * kHidden;
constexpr int kRank = 320;

__global__ void repeat_kernel(const __nv_bfloat16* input, __nv_bfloat16* hyper,
                              std::int64_t count) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const int d = static_cast<int>(i % kHyper);
        const std::int64_t token = i / kHyper;
        hyper[i] = input[(d % kHidden) + static_cast<std::int64_t>(kHidden) * token];
    }
}

__global__ void add_repeated_kernel(const __nv_bfloat16* embedding, __nv_bfloat16* hyper,
                                    std::int64_t count) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const int d = static_cast<int>(i % kHyper);
        const std::int64_t token = i / kHyper;
        hyper[i] = __float2bfloat16_rn(
            __bfloat162float(hyper[i]) +
            __bfloat162float(embedding[(d % kHidden) +
                                       static_cast<std::int64_t>(kHidden) * token]));
    }
}

__global__ void grouped_rmsnorm_kernel(const __nv_bfloat16* hyper,
                                       const __nv_bfloat16* weight,
                                       __nv_bfloat16* normalized) {
    const int stream = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const std::int64_t base = static_cast<std::int64_t>(kHyper) * token + kHidden * stream;
    float sum = 0.0F;
    for (int d = static_cast<int>(threadIdx.x); d < kHidden; d += static_cast<int>(blockDim.x)) {
        const float x = __bfloat162float(hyper[base + d]);
        sum = fmaf(x, x, sum);
    }
    for (int offset = 16; offset != 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    __shared__ float partial[8];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { partial[warp] = sum; }
    __syncthreads();
    if (warp == 0) {
        float value = lane < 8 ? partial[lane] : 0.0F;
        for (int offset = 16; offset != 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        if (lane == 0) { partial[0] = rsqrtf(value / static_cast<float>(kHidden) + 1.0e-6F); }
    }
    __syncthreads();
    for (int d = static_cast<int>(threadIdx.x); d < kHidden; d += static_cast<int>(blockDim.x)) {
        const float x = __bfloat162float(hyper[base + d]);
        const float scale = 1.0F + __bfloat162float(weight[kHidden * stream + d]);
        normalized[base + d] = __float2bfloat16_rn(x * partial[0] * scale);
    }
}

__global__ void combine_grouped_rmsnorm_kernel(
    __nv_bfloat16* hyper, const __nv_bfloat16* block,
    const __nv_bfloat16* injection, const __nv_bfloat16* weight,
    __nv_bfloat16* normalized) {
    const int stream = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const std::int64_t base = static_cast<std::int64_t>(kHyper) * token + kHidden * stream;
    const float logit = __bfloat162float(injection[stream + kStreams * token]) * 0.25F;
    const float branch_scale = 2.0F / (1.0F + expf(-logit));
    float sum = 0.0F;
    for (int d = static_cast<int>(threadIdx.x); d < kHidden; d += blockDim.x) {
        const __nv_bfloat16 represented = __float2bfloat16_rn(
            __bfloat162float(hyper[base + d]) +
            branch_scale * __bfloat162float(block[d + kHidden * token]));
        hyper[base + d] = represented;
        const float value = __bfloat162float(represented);
        sum = fmaf(value, value, sum);
    }
    for (int offset = 16; offset != 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    __shared__ float partial[8];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { partial[warp] = sum; }
    __syncthreads();
    if (warp == 0) {
        float value = lane < 8 ? partial[lane] : 0.0F;
        for (int offset = 16; offset != 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        if (lane == 0) {
            partial[0] = rsqrtf(value / static_cast<float>(kHidden) + 1.0e-6F);
        }
    }
    __syncthreads();
    for (int d = static_cast<int>(threadIdx.x); d < kHidden; d += blockDim.x) {
        const float value = __bfloat162float(hyper[base + d]);
        const float scale = 1.0F + __bfloat162float(weight[kHidden * stream + d]);
        normalized[base + d] = __float2bfloat16_rn(value * partial[0] * scale);
    }
}

__global__ void scaled_silu_kernel(__nv_bfloat16* values, std::int64_t count) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const float value = __bfloat162float(values[i]) * 0.25F;
        values[i] = __float2bfloat16_rn(value / (1.0F + expf(-value)));
    }
}

__global__ void gate_mix_kernel(const __nv_bfloat16* normalized,
                                const __nv_bfloat16* gate_logits,
                                __nv_bfloat16* block_input, int tokens) {
    const std::int64_t count = static_cast<std::int64_t>(kHidden) * tokens;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const int d = static_cast<int>(i % kHidden);
        const int t = static_cast<int>(i / kHidden);
        float sum = 0.0F;
#pragma unroll
        for (int stream = 0; stream < kStreams; ++stream) {
            const std::int64_t offset = d + static_cast<std::int64_t>(kHidden) *
                (stream + static_cast<std::int64_t>(kStreams) * t);
            const float gate = 1.0F / (1.0F + expf(-__bfloat162float(gate_logits[offset])));
            sum = fmaf(gate, __bfloat162float(normalized[offset]), sum);
        }
        block_input[i] = __float2bfloat16_rn(sum * 0.25F);
    }
}

__global__ void gate_mix_injection_kernel(const __nv_bfloat16* normalized,
                                          const __nv_bfloat16* gate_logits,
                                          const __nv_bfloat16* injection_weight,
                                          __nv_bfloat16* block_input,
                                          __nv_bfloat16* injection) {
    const int token = static_cast<int>(blockIdx.x);
    float injection_sum[kStreams] = {};
    for (int d = static_cast<int>(threadIdx.x); d < kHidden;
         d += static_cast<int>(blockDim.x)) {
        float mixed = 0.0F;
#pragma unroll
        for (int source_stream = 0; source_stream < kStreams; ++source_stream) {
            const int k = source_stream * kHidden + d;
            const std::int64_t offset = k + static_cast<std::int64_t>(kHyper) * token;
            const float value = __bfloat162float(normalized[offset]);
            const float gate =
                1.0F / (1.0F + expf(-__bfloat162float(gate_logits[offset])));
            mixed = fmaf(gate, value, mixed);
#pragma unroll
            for (int destination_stream = 0; destination_stream < kStreams;
                 ++destination_stream) {
                injection_sum[destination_stream] = fmaf(
                    __bfloat162float(injection_weight[
                        static_cast<std::int64_t>(destination_stream) * kHyper + k]),
                    value, injection_sum[destination_stream]);
            }
        }
        block_input[d + static_cast<std::int64_t>(kHidden) * token] =
            __float2bfloat16_rn(mixed * 0.25F);
    }

    __shared__ float partial[8][kStreams];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int stream = 0; stream < kStreams; ++stream) {
        for (int offset = 16; offset != 0; offset >>= 1) {
            injection_sum[stream] +=
                __shfl_down_sync(0xffffffffU, injection_sum[stream], offset);
        }
        if (lane == 0) { partial[warp][stream] = injection_sum[stream]; }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int stream = 0; stream < kStreams; ++stream) {
            float value = lane < 8 ? partial[lane][stream] : 0.0F;
            for (int offset = 16; offset != 0; offset >>= 1) {
                value += __shfl_down_sync(0xffffffffU, value, offset);
            }
            if (lane == 0) {
                injection[stream + static_cast<std::int64_t>(kStreams) * token] =
                    __float2bfloat16_rn(value);
            }
        }
    }
}

__global__ void injection_kernel(const __nv_bfloat16* normalized,
                                 const __nv_bfloat16* weight,
                                 __nv_bfloat16* injection, int tokens) {
    const int stream = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    float sum = 0.0F;
    for (int k = static_cast<int>(threadIdx.x); k < kHyper; k += static_cast<int>(blockDim.x)) {
        sum = fmaf(__bfloat162float(weight[static_cast<std::int64_t>(stream) * kHyper + k]),
                   __bfloat162float(normalized[k + static_cast<std::int64_t>(kHyper) * token]), sum);
    }
    for (int offset = 16; offset != 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    __shared__ float partial[8];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { partial[warp] = sum; }
    __syncthreads();
    if (warp == 0) {
        float value = lane < 8 ? partial[lane] : 0.0F;
        for (int offset = 16; offset != 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        if (lane == 0) { injection[stream + kStreams * token] = __float2bfloat16_rn(value); }
    }
}

__global__ void gate_mix_injection_decode_kernel(
    const __nv_bfloat16* normalized, const __nv_bfloat16* gate_logits,
    const __nv_bfloat16* injection_weight, __nv_bfloat16* block_input,
    __nv_bfloat16* injection) {
    constexpr int kMixBlocks = kHidden / 256;
    const int work = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    if (work < kMixBlocks) {
        const int d = work * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
        float sum = 0.0F;
#pragma unroll
        for (int stream = 0; stream < kStreams; ++stream) {
            const std::int64_t offset = d + static_cast<std::int64_t>(kHidden) *
                (stream + static_cast<std::int64_t>(kStreams) * token);
            const float gate =
                1.0F / (1.0F + expf(-__bfloat162float(gate_logits[offset])));
            sum = fmaf(gate, __bfloat162float(normalized[offset]), sum);
        }
        block_input[d + static_cast<std::int64_t>(kHidden) * token] =
            __float2bfloat16_rn(sum * 0.25F);
        return;
    }

    const int destination_stream = work - kMixBlocks;
    float sum = 0.0F;
    for (int k = static_cast<int>(threadIdx.x); k < kHyper;
         k += static_cast<int>(blockDim.x)) {
        sum = fmaf(
            __bfloat162float(injection_weight[
                static_cast<std::int64_t>(destination_stream) * kHyper + k]),
            __bfloat162float(normalized[k + static_cast<std::int64_t>(kHyper) * token]), sum);
    }
    for (int offset = 16; offset != 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    __shared__ float partial[8];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { partial[warp] = sum; }
    __syncthreads();
    if (warp == 0) {
        float value = lane < 8 ? partial[lane] : 0.0F;
        for (int offset = 16; offset != 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        if (lane == 0) {
            injection[destination_stream + static_cast<std::int64_t>(kStreams) * token] =
                __float2bfloat16_rn(value);
        }
    }
}

__global__ void combine_kernel(__nv_bfloat16* hyper, const __nv_bfloat16* block,
                               const __nv_bfloat16* injection, int tokens) {
    const std::int64_t count = static_cast<std::int64_t>(kHyper) * tokens;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const int d = static_cast<int>(i % kHidden);
        const std::int64_t row = i / kHidden;
        const int stream = static_cast<int>(row % kStreams);
        const int token = static_cast<int>(row / kStreams);
        const float logit = __bfloat162float(injection[stream + kStreams * token]) * 0.25F;
        const float scale = 2.0F / (1.0F + expf(-logit));
        hyper[i] = __float2bfloat16_rn(__bfloat162float(hyper[i]) +
                                       scale * __bfloat162float(block[d + kHidden * token]));
    }
}

int grid_for(std::int64_t count) {
    constexpr int block = 256;
    const std::int64_t blocks = (count + block - 1) / block;
    return static_cast<int>(blocks < 4096 ? blocks : 4096);
}

void validate(const Tensor& hyper, const HyperConnectionWeights& weights, const Tensor& block,
              const Tensor* injection) {
    const int tokens = hyper.ne[1];
    if (hyper.dtype != DType::BF16 || !hyper.is_contiguous() || hyper.ne[0] != kHyper ||
        hyper.ne[2] != 1 || hyper.ne[3] != 1 || tokens <= 0 || hyper.data == nullptr ||
        block.dtype != DType::BF16 || !block.is_contiguous() || block.ne[0] != kHidden ||
        block.ne[1] != tokens || block.ne[2] != 1 || block.ne[3] != 1 || block.data == nullptr ||
        weights.norm.dtype != DType::BF16 || weights.norm.ne[0] != kHyper ||
        !weights.norm.is_contiguous() || weights.down.n != kRank || weights.down.k != kHyper ||
        weights.up.n != kHyper || weights.up.k != kRank) {
        throw std::invalid_argument("hyperconnection_mix: invalid Flash-Next geometry");
    }
    // The low-rank down projection is BF16 or row-scaled FP8; the up projection and the
    // injection rows are read as BF16.
    const bool bf16_down =
        weights.down.qtype == QType::BF16 && weights.down.layout == QuantLayout::Contiguous;
    const bool fp8_down = weights.down.qtype == QType::FP8_E4M3FN_ROW_BF16 &&
                          weights.down.layout == QuantLayout::RowScale;
    if ((!bf16_down && !fp8_down) || weights.up.qtype != QType::BF16 ||
        weights.up.layout != QuantLayout::Contiguous) {
        throw std::invalid_argument("hyperconnection_mix: unsupported weight representation");
    }
    if (injection != nullptr &&
        (injection->dtype != DType::BF16 || !injection->is_contiguous() ||
         injection->ne[0] != kStreams || injection->ne[1] != tokens ||
         weights.injection.n != kStreams || weights.injection.k != kHyper ||
         weights.injection.qtype != QType::BF16 ||
         weights.injection.layout != QuantLayout::Contiguous)) {
        throw std::invalid_argument("hyperconnection_mix: invalid injection geometry");
    }
}

void validate_combine_inputs(const Tensor& hyper, const Tensor& block_output,
                             const Tensor& injection) {
    if (hyper.dtype != DType::BF16 || !hyper.is_contiguous() || hyper.ne[0] != kHyper ||
        block_output.dtype != DType::BF16 || !block_output.is_contiguous() ||
        block_output.ne[0] != kHidden || block_output.ne[1] != hyper.ne[1] ||
        injection.dtype != DType::BF16 || !injection.is_contiguous() ||
        injection.ne[0] != kStreams || injection.ne[1] != hyper.ne[1]) {
        throw std::invalid_argument("hyperconnection_combine: invalid Flash-Next geometry");
    }
}

void finish_mix(const Tensor& normalized, const HyperConnectionWeights& weights,
                Tensor& block_input, Tensor* injection, WorkspaceArena& workspace,
                cudaStream_t stream, Bf16GemmContext* bf16_gemm) {
    const int tokens = normalized.ne[1];
    Tensor low_rank = workspace.alloc(DType::BF16, {kRank, tokens});
    // FP8 runs the fused down+SiLU at every width; BF16 fuses through its small-T family.
    const bool fp8_down        = weights.down.qtype == QType::FP8_E4M3FN_ROW_BF16;
    const bool fused_down_silu = fp8_down || tokens <= 16;
    if (fp8_down) {
        detail::flash_next::launch_fp8_hc_down_silu(normalized, weights.down, low_rank, stream);
    } else if (tokens == 1) {
        detail::flash_next::launch_bf16_hc_down_silu_decode(normalized, weights.down, low_rank, stream);
    } else if (fused_down_silu) {
        detail::flash_next::launch_bf16_hc_down_silu_small_t(normalized, weights.down, low_rank, stream);
    } else {
        linear(normalized, weights.down, low_rank, stream, bf16_gemm);
    }
    constexpr int block = 256;
    if (!fused_down_silu) {
        scaled_silu_kernel<<<grid_for(low_rank.numel()), block, 0, stream>>>(
            static_cast<__nv_bfloat16*>(low_rank.data), low_rank.numel());
    }
    Tensor gate = workspace.alloc(DType::BF16, {kHyper, tokens});
    linear(low_rank, weights.up, gate, stream, bf16_gemm);
    if (injection != nullptr) {
        if (tokens > 16) {
            gate_mix_injection_kernel<<<tokens, block, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(normalized.data),
                static_cast<const __nv_bfloat16*>(gate.data),
                static_cast<const __nv_bfloat16*>(weights.injection.qdata),
                static_cast<__nv_bfloat16*>(block_input.data),
                static_cast<__nv_bfloat16*>(injection->data));
        } else {
            constexpr int kDecodeMixBlocks = kHidden / block;
            gate_mix_injection_decode_kernel<<<
                dim3(kDecodeMixBlocks + kStreams, static_cast<unsigned int>(tokens)),
                block, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(normalized.data),
                static_cast<const __nv_bfloat16*>(gate.data),
                static_cast<const __nv_bfloat16*>(weights.injection.qdata),
                static_cast<__nv_bfloat16*>(block_input.data),
                static_cast<__nv_bfloat16*>(injection->data));
        }
    } else {
        gate_mix_kernel<<<grid_for(block_input.numel()), block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(normalized.data),
            static_cast<const __nv_bfloat16*>(gate.data),
            static_cast<__nv_bfloat16*>(block_input.data), tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void hyperconnection_repeat(const Tensor& input, Tensor& hyper, cudaStream_t stream) {
    NINFER_PERF_SCOPE("hyper.repeat", input.ne[1], 0, 0,
                       flash_next_work::dense(0, input.ne[1], 2 * (2560 + 10240) * input.ne[1]));

    if (input.dtype != DType::BF16 || !input.is_contiguous() || input.ne[0] != kHidden ||
        input.ne[1] <= 0 || input.ne[2] != 1 || input.ne[3] != 1 || input.data == nullptr ||
        hyper.dtype != DType::BF16 || !hyper.is_contiguous() || hyper.ne[0] != kHyper ||
        hyper.ne[1] != input.ne[1] || hyper.ne[2] != 1 || hyper.ne[3] != 1 ||
        hyper.data == nullptr) {
        throw std::invalid_argument("hyperconnection_repeat: invalid Flash-Next geometry");
    }
    repeat_kernel<<<grid_for(hyper.numel()), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data),
        static_cast<__nv_bfloat16*>(hyper.data), hyper.numel());
    CUDA_CHECK(cudaGetLastError());
}

void hyperconnection_add_repeated(const Tensor& embedding, Tensor& hyper,
                                  cudaStream_t stream) {
    NINFER_PERF_SCOPE("hyper.add", embedding.ne[1], 0, 0,
                       flash_next_work::dense(0, embedding.ne[1], 2 * (2560 + 2 * 10240) * embedding.ne[1]));

    if (embedding.dtype != DType::BF16 || !embedding.is_contiguous() ||
        embedding.ne[0] != kHidden || embedding.ne[1] <= 0 || embedding.ne[2] != 1 ||
        embedding.ne[3] != 1 || embedding.data == nullptr || hyper.dtype != DType::BF16 ||
        !hyper.is_contiguous() || hyper.ne[0] != kHyper ||
        hyper.ne[1] != embedding.ne[1] || hyper.ne[2] != 1 || hyper.ne[3] != 1 ||
        hyper.data == nullptr) {
        throw std::invalid_argument("hyperconnection_add_repeated: invalid Flash-Next geometry");
    }
    add_repeated_kernel<<<grid_for(hyper.numel()), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(embedding.data),
        static_cast<__nv_bfloat16*>(hyper.data), hyper.numel());
    CUDA_CHECK(cudaGetLastError());
}

std::size_t hyperconnection_mix_workspace_capacity_bytes(std::int32_t tokens,
                                                          bool with_injection) {
    if (tokens <= 0) { throw std::invalid_argument("HyperConnection token count must be positive"); }
    const std::uint64_t elements = static_cast<std::uint64_t>(tokens) * (kHyper + kRank + kHyper);
    const std::uint64_t bytes = elements * sizeof(__nv_bfloat16);
    if (bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("HyperConnection workspace size overflow");
    }
    (void)with_injection;
    return static_cast<std::size_t>(bytes) + 3 * 256;
}

void hyperconnection_mix(const Tensor& hyper, const HyperConnectionWeights& weights,
                         Tensor& block_input, Tensor* injection, WorkspaceArena& workspace,
                         cudaStream_t stream, Bf16GemmContext* bf16_gemm) {
    NINFER_PERF_SCOPE("hyper.mix", hyper.ne[1], 0, 0,
                       flash_next_work::hyper(hyper.ne[1], injection != nullptr, false,
                                              weights.down.qtype == QType::FP8_E4M3FN_ROW_BF16));

    validate(hyper, weights, block_input, injection);
    const int tokens = hyper.ne[1];
    auto scope = workspace.scope();
    Tensor normalized = workspace.alloc(DType::BF16, {kHyper, tokens});
    grouped_rmsnorm_kernel<<<dim3(kStreams, static_cast<unsigned int>(tokens)), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hyper.data),
        static_cast<const __nv_bfloat16*>(weights.norm.data),
        static_cast<__nv_bfloat16*>(normalized.data));
    finish_mix(normalized, weights, block_input, injection, workspace, stream, bf16_gemm);
}

void hyperconnection_combine_mix(Tensor& hyper, const Tensor& previous_block_output,
                                 const Tensor& previous_injection,
                                 const HyperConnectionWeights& weights,
                                 Tensor& block_input, Tensor* injection,
                                 WorkspaceArena& workspace, cudaStream_t stream,
                                 Bf16GemmContext* bf16_gemm) {
    NINFER_PERF_SCOPE("hyper.combine_mix", hyper.ne[1], 0, 0,
                       flash_next_work::hyper(hyper.ne[1], injection != nullptr, true,
                                              weights.down.qtype == QType::FP8_E4M3FN_ROW_BF16));

    validate(hyper, weights, block_input, injection);
    validate_combine_inputs(hyper, previous_block_output, previous_injection);
    const int tokens = hyper.ne[1];
    auto scope = workspace.scope();
    Tensor normalized = workspace.alloc(DType::BF16, {kHyper, tokens});
    combine_grouped_rmsnorm_kernel<<<
        dim3(kStreams, static_cast<unsigned int>(tokens)), 256, 0, stream>>>(
        static_cast<__nv_bfloat16*>(hyper.data),
        static_cast<const __nv_bfloat16*>(previous_block_output.data),
        static_cast<const __nv_bfloat16*>(previous_injection.data),
        static_cast<const __nv_bfloat16*>(weights.norm.data),
        static_cast<__nv_bfloat16*>(normalized.data));
    finish_mix(normalized, weights, block_input, injection, workspace, stream, bf16_gemm);
}

void hyperconnection_combine(Tensor& hyper, const Tensor& block_output, const Tensor& injection,
                             cudaStream_t stream) {
    NINFER_PERF_SCOPE("hyper.combine", hyper.ne[1], 0, 0,
                       flash_next_work::dense(0, hyper.ne[1], 2 * (2 * 10240 + 2560 + 4) * hyper.ne[1]));

    validate_combine_inputs(hyper, block_output, injection);
    constexpr int block = 256;
    combine_kernel<<<grid_for(hyper.numel()), block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(hyper.data),
        static_cast<const __nv_bfloat16*>(block_output.data),
        static_cast<const __nv_bfloat16*>(injection.data), hyper.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
