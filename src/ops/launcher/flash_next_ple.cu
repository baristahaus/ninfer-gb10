#include "ninfer/ops/flash_next_ple.h"

#include "ops/flash_next_work.h"

#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr int kStreams = 4;
constexpr int kHidden = 2560;
constexpr int kHyper = kStreams * kHidden;
constexpr int kState = 9;

__global__ void dequantize_embedding_kernel(const std::uint8_t* input,
                                            const __nv_bfloat16* scale,
                                            __nv_bfloat16* output, std::int64_t count) {
    const float s = __bfloat162float(*scale);
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        output[i] = __float2bfloat16_rn(detail::decode_nvfp4_e4m3(input[i]) * s);
    }
}

__device__ float reduce_sum(float value, float* partial) {
    for (int offset = 16; offset != 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { partial[warp] = value; }
    __syncthreads();
    if (warp == 0) {
        value = lane < 8 ? partial[lane] : 0.0F;
        for (int offset = 16; offset != 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        if (lane == 0) { partial[0] = value; }
    }
    __syncthreads();
    return partial[0];
}

__global__ void gate_kernel(const __nv_bfloat16* hyper, const __nv_bfloat16* key,
                            const __nv_bfloat16* value, const __nv_bfloat16* query_norm,
                            const __nv_bfloat16* key_norm, __nv_bfloat16* gated) {
    const int stream = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const std::int64_t base = static_cast<std::int64_t>(token) * kHyper + stream * kHidden;
    float query_square = 0.0F;
    float key_square = 0.0F;
    for (int d = static_cast<int>(threadIdx.x); d < kHidden; d += static_cast<int>(blockDim.x)) {
        const float q = __bfloat162float(hyper[base + d]);
        const float k = __bfloat162float(key[base + d]);
        query_square = fmaf(q, q, query_square);
        key_square = fmaf(k, k, key_square);
    }
    __shared__ float partial[8];
    query_square = reduce_sum(query_square, partial);
    const float query_inverse = rsqrtf(query_square / kHidden + 1.0e-6F);
    key_square = reduce_sum(key_square, partial);
    const float key_inverse = rsqrtf(key_square / kHidden + 1.0e-6F);
    float dot = 0.0F;
    for (int d = static_cast<int>(threadIdx.x); d < kHidden; d += static_cast<int>(blockDim.x)) {
        const float q = __bfloat162float(hyper[base + d]) * query_inverse *
                        (1.0F + __bfloat162float(query_norm[stream * kHidden + d]));
        const float k = __bfloat162float(key[base + d]) * key_inverse *
                        (1.0F + __bfloat162float(key_norm[stream * kHidden + d]));
        dot = fmaf(q, k, dot);
    }
    dot = reduce_sum(dot, partial) * rsqrtf(static_cast<float>(kHidden));
    const float transformed = copysignf(sqrtf(fmaxf(fabsf(dot), 1.0e-6F)), dot);
    const float gate = 1.0F / (1.0F + expf(-transformed));
    for (int d = static_cast<int>(threadIdx.x); d < kHidden; d += static_cast<int>(blockDim.x)) {
        gated[base + d] = __float2bfloat16_rn(
            gate * __bfloat162float(value[token * kHidden + d]));
    }
}

__global__ void grouped_norm_kernel(const __nv_bfloat16* input, const __nv_bfloat16* weight,
                                    __nv_bfloat16* output) {
    const int stream = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const std::int64_t base = static_cast<std::int64_t>(token) * kHyper + stream * kHidden;
    float square = 0.0F;
    for (int d = static_cast<int>(threadIdx.x); d < kHidden; d += static_cast<int>(blockDim.x)) {
        const float x = __bfloat162float(input[base + d]);
        square = fmaf(x, x, square);
    }
    __shared__ float partial[8];
    square = reduce_sum(square, partial);
    const float inverse = rsqrtf(square / kHidden + 1.0e-6F);
    for (int d = static_cast<int>(threadIdx.x); d < kHidden; d += static_cast<int>(blockDim.x)) {
        output[base + d] = __float2bfloat16_rn(
            __bfloat162float(input[base + d]) * inverse *
            (1.0F + __bfloat162float(weight[stream * kHidden + d])));
    }
}

// Each thread owns whole channels: it reads the channel's source history before writing the
// destination history, so source_state may alias destination_state.
__global__ void dilated_conv_kernel(const __nv_bfloat16* input, const __nv_bfloat16* weight,
                                    const __nv_bfloat16* source_state,
                                    __nv_bfloat16* destination_state, const __nv_bfloat16* gated,
                                    __nv_bfloat16* destination, int tokens) {
    for (int channel = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
         channel < kHyper; channel += static_cast<int>(blockDim.x) * gridDim.x) {
        float history[kState];
#pragma unroll
        for (int i = 0; i < kState; ++i) {
            history[i] =
                __bfloat162float(source_state[channel + static_cast<std::int64_t>(kHyper) * i]);
        }
        for (int token = 0; token < tokens; ++token) {
            const float current = __bfloat162float(input[channel + static_cast<std::int64_t>(kHyper) * token]);
            float convolution = __bfloat162float(weight[channel]) * history[0];
            convolution = fmaf(__bfloat162float(weight[channel + kHyper]), history[3], convolution);
            convolution = fmaf(__bfloat162float(weight[channel + 2 * kHyper]), history[6], convolution);
            convolution = fmaf(__bfloat162float(weight[channel + 3 * kHyper]), current, convolution);
            const float represented_convolution =
                __bfloat162float(__float2bfloat16_rn(convolution));
            const float silu = __bfloat162float(__float2bfloat16_rn(
                represented_convolution / (1.0F + expf(-represented_convolution))));
            destination[channel + static_cast<std::int64_t>(kHyper) * token] =
                __float2bfloat16_rn(__bfloat162float(gated[channel + static_cast<std::int64_t>(kHyper) * token]) + silu);
#pragma unroll
            for (int i = 0; i < kState - 1; ++i) { history[i] = history[i + 1]; }
            history[kState - 1] = current;
        }
#pragma unroll
        for (int i = 0; i < kState; ++i) {
            destination_state[channel + static_cast<std::int64_t>(kHyper) * i] =
                __float2bfloat16_rn(history[i]);
        }
    }
}

__global__ void dilated_conv_batch_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* states,
    const std::int32_t* valid_columns, const std::int32_t* source_slots,
    const std::int32_t* destination_slots, const __nv_bfloat16* gated,
    __nv_bfloat16* destination, int width, int batch, std::int64_t slot_stride) {
    const int lane = static_cast<int>(blockIdx.y);
    for (int channel = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
         channel < kHyper; channel += static_cast<int>(blockDim.x) * gridDim.x) {
        const __nv_bfloat16* source =
            states + static_cast<std::int64_t>(source_slots[lane]) * slot_stride;
        float history[kState];
#pragma unroll
        for (int i = 0; i < kState; ++i) {
            history[i] = __bfloat162float(source[channel + static_cast<std::int64_t>(kHyper) * i]);
        }
        const int valid = valid_columns == nullptr ? width : valid_columns[lane];
        for (int column = 0; column < width; ++column) {
            const std::int64_t token = static_cast<std::int64_t>(lane) * width + column;
            const std::int64_t index = channel + static_cast<std::int64_t>(kHyper) * token;
            if (column >= valid) {
                destination[index] = __float2bfloat16(0.0F);
                continue;
            }
            const float current = __bfloat162float(input[index]);
            float convolution = __bfloat162float(weight[channel]) * history[0];
            convolution = fmaf(__bfloat162float(weight[channel + kHyper]), history[3], convolution);
            convolution = fmaf(__bfloat162float(weight[channel + 2 * kHyper]), history[6], convolution);
            convolution = fmaf(__bfloat162float(weight[channel + 3 * kHyper]), current, convolution);
            const float represented_convolution =
                __bfloat162float(__float2bfloat16_rn(convolution));
            const float silu = __bfloat162float(__float2bfloat16_rn(
                represented_convolution / (1.0F + expf(-represented_convolution))));
            destination[index] = __float2bfloat16_rn(
                __bfloat162float(gated[index]) + silu);
#pragma unroll
            for (int i = 0; i < kState - 1; ++i) { history[i] = history[i + 1]; }
            history[kState - 1] = current;
        }
        __nv_bfloat16* target =
            states + static_cast<std::int64_t>(destination_slots[lane]) * slot_stride;
#pragma unroll
        for (int i = 0; i < kState; ++i) {
            target[channel + static_cast<std::int64_t>(kHyper) * i] =
                __float2bfloat16_rn(history[i]);
        }
    }
}

__global__ void dilated_conv_record_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, const __nv_bfloat16* states,
    const std::int32_t* valid_columns, const std::int32_t* source_slots,
    __nv_bfloat16* records, const __nv_bfloat16* gated, __nv_bfloat16* destination,
    int width, int batch, std::int64_t slot_stride) {
    const int lane = static_cast<int>(blockIdx.y);
    for (int channel = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
         channel < kHyper; channel += static_cast<int>(blockDim.x) * gridDim.x) {
        const __nv_bfloat16* source =
            states + static_cast<std::int64_t>(source_slots[lane]) * slot_stride;
        float history[kState];
#pragma unroll
        for (int i = 0; i < kState; ++i) {
            history[i] = __bfloat162float(source[channel + static_cast<std::int64_t>(kHyper) * i]);
        }
        const int valid = valid_columns == nullptr ? width : valid_columns[lane];
        for (int column = 0; column < width; ++column) {
            const std::int64_t token = static_cast<std::int64_t>(lane) * width + column;
            const std::int64_t index = channel + static_cast<std::int64_t>(kHyper) * token;
            if (column >= valid) {
                destination[index] = __float2bfloat16(0.0F);
                continue;
            }
            const __nv_bfloat16 current_bits = input[index];
            records[index] = current_bits;
            const float current = __bfloat162float(current_bits);
            float convolution = __bfloat162float(weight[channel]) * history[0];
            convolution = fmaf(__bfloat162float(weight[channel + kHyper]), history[3], convolution);
            convolution = fmaf(__bfloat162float(weight[channel + 2 * kHyper]), history[6], convolution);
            convolution = fmaf(__bfloat162float(weight[channel + 3 * kHyper]), current, convolution);
            const float represented_convolution =
                __bfloat162float(__float2bfloat16_rn(convolution));
            const float silu = __bfloat162float(__float2bfloat16_rn(
                represented_convolution / (1.0F + expf(-represented_convolution))));
            destination[index] = __float2bfloat16_rn(
                __bfloat162float(gated[index]) + silu);
#pragma unroll
            for (int i = 0; i < kState - 1; ++i) { history[i] = history[i + 1]; }
            history[kState - 1] = current;
        }
    }
}

struct PleFoldKernelRows {
    FlashNextPleFoldRow rows[8];
};

// device_rows, when non-null, replaces rows: four I32 words per row
// {source_state_slot, destination_state_slot, commit_columns, 0} read when the work runs.
__global__ void ple_fold_kernel(const __nv_bfloat16* records, __nv_bfloat16* states,
                                PleFoldKernelRows rows, const std::int32_t* device_rows,
                                int row_count, int width, std::int64_t slot_stride) {
    const int lane = static_cast<int>(blockIdx.y);
    const FlashNextPleFoldRow row =
        device_rows != nullptr
            ? FlashNextPleFoldRow{device_rows[4 * lane], device_rows[4 * lane + 1],
                                  device_rows[4 * lane + 2]}
            : rows.rows[lane];
    for (int channel = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x; channel < kHyper;
         channel += static_cast<int>(blockDim.x) * gridDim.x) {
        if (lane >= row_count || row.commit_columns == 0) { continue; }
        const __nv_bfloat16* source =
            states + static_cast<std::int64_t>(row.source_state_slot) * slot_stride;
        __nv_bfloat16 history[kState];
#pragma unroll
        for (int i = 0; i < kState; ++i) {
            history[i] = source[channel + static_cast<std::int64_t>(kHyper) * i];
        }
        for (int column = 0; column < row.commit_columns; ++column) {
#pragma unroll
            for (int i = 0; i < kState - 1; ++i) { history[i] = history[i + 1]; }
            history[kState - 1] = records[channel + static_cast<std::int64_t>(kHyper) *
                                               (lane * width + column)];
        }
        __nv_bfloat16* target =
            states + static_cast<std::int64_t>(row.destination_state_slot) * slot_stride;
#pragma unroll
        for (int i = 0; i < kState; ++i) {
            target[channel + static_cast<std::int64_t>(kHyper) * i] = history[i];
        }
    }
}

int grid_for(std::int64_t count) {
    const std::int64_t blocks = (count + 255) / 256;
    return static_cast<int>(blocks < 4096 ? blocks : 4096);
}

void validate(const Tensor& hyper, const Tensor& gathered, const FlashNextPleWeights& weights,
              const Tensor& state, const Tensor& destination) {
    const int tokens = hyper.ne[1];
    if (tokens <= 0 || hyper.dtype != DType::BF16 || !hyper.is_contiguous() ||
        hyper.ne[0] != kHyper || gathered.dtype != DType::FP8_E4M3FN ||
        !gathered.is_contiguous() || gathered.ne[0] != kHidden || gathered.ne[1] != tokens ||
        destination.dtype != DType::BF16 || !destination.is_contiguous() ||
        destination.ne[0] != kHyper || destination.ne[1] != tokens || state.dtype != DType::BF16 ||
        !state.is_contiguous() || state.ne[0] != kHyper || state.ne[1] != kState ||
        weights.key_projection.n != kHyper || weights.key_projection.k != kHidden ||
        weights.value_projection.n != kHidden || weights.value_projection.k != kHidden ||
        weights.key_norm.dtype != DType::BF16 || weights.key_norm.ne[0] != kHyper ||
        weights.query_norm.dtype != DType::BF16 || weights.query_norm.ne[0] != kHyper ||
        weights.convolution_norm.dtype != DType::BF16 || weights.convolution_norm.ne[0] != kHyper ||
        weights.convolution.dtype != DType::BF16 || weights.convolution.ne[0] != kHyper ||
        weights.convolution.ne[1] != 4 || weights.convolution.ne[2] != 1 ||
        weights.embedding_scale.dtype != DType::BF16 ||
        weights.embedding_scale.numel() != 1) {
        throw std::invalid_argument("flash_next_ple: invalid exact geometry");
    }
}

} // namespace

std::size_t flash_next_ple_workspace_capacity_bytes(std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("Flash-Next PLE token count must be positive"); }
    const std::uint64_t elements = static_cast<std::uint64_t>(tokens) *
                                   (kHidden + kHyper + kHidden + kHyper + kHyper);
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(__nv_bfloat16)) {
        throw std::overflow_error("Flash-Next PLE workspace size overflow");
    }
    return static_cast<std::size_t>(elements * sizeof(__nv_bfloat16)) + 5 * 256;
}

void flash_next_ple(const Tensor& hyper, const Tensor& gathered_fp8,
                    const FlashNextPleWeights& weights, const Tensor& source_state,
                    Tensor& destination_state, Tensor& destination, WorkspaceArena& workspace,
                    cudaStream_t stream, Bf16GemmContext* bf16_gemm) {
    NINFER_PERF_SCOPE("ple.prefill", hyper.ne[1], 1, 0, flash_next_work::ple(hyper.ne[1]));

    validate(hyper, gathered_fp8, weights, source_state, destination);
    validate(hyper, gathered_fp8, weights, destination_state, destination);
    if (source_state.data != destination_state.data) {
        const auto* source_begin = static_cast<const std::byte*>(source_state.data);
        const auto* target_begin = static_cast<const std::byte*>(destination_state.data);
        if (source_begin < target_begin + destination_state.bytes() &&
            target_begin < source_begin + source_state.bytes()) {
            throw std::invalid_argument("flash_next_ple: source and destination states overlap");
        }
    }
    const int tokens = hyper.ne[1];
    auto scope = workspace.scope();
    Tensor embedding = workspace.alloc(DType::BF16, {kHidden, tokens});
    dequantize_embedding_kernel<<<grid_for(embedding.numel()), 256, 0, stream>>>(
        static_cast<const std::uint8_t*>(gathered_fp8.data),
        static_cast<const __nv_bfloat16*>(weights.embedding_scale.data),
        static_cast<__nv_bfloat16*>(embedding.data), embedding.numel());
    Tensor key = workspace.alloc(DType::BF16, {kHyper, tokens});
    Tensor value = workspace.alloc(DType::BF16, {kHidden, tokens});
    linear(embedding, weights.key_projection, key, stream, bf16_gemm);
    linear(embedding, weights.value_projection, value, stream, bf16_gemm);
    Tensor gated = workspace.alloc(DType::BF16, {kHyper, tokens});
    gate_kernel<<<dim3(kStreams, static_cast<unsigned int>(tokens)), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hyper.data), static_cast<const __nv_bfloat16*>(key.data),
        static_cast<const __nv_bfloat16*>(value.data),
        static_cast<const __nv_bfloat16*>(weights.query_norm.data),
        static_cast<const __nv_bfloat16*>(weights.key_norm.data),
        static_cast<__nv_bfloat16*>(gated.data));
    Tensor normalized = workspace.alloc(DType::BF16, {kHyper, tokens});
    grouped_norm_kernel<<<dim3(kStreams, static_cast<unsigned int>(tokens)), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gated.data),
        static_cast<const __nv_bfloat16*>(weights.convolution_norm.data),
        static_cast<__nv_bfloat16*>(normalized.data));
    dilated_conv_kernel<<<40, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(normalized.data),
        static_cast<const __nv_bfloat16*>(weights.convolution.data),
        static_cast<const __nv_bfloat16*>(source_state.data),
        static_cast<__nv_bfloat16*>(destination_state.data),
        static_cast<const __nv_bfloat16*>(gated.data),
        static_cast<__nv_bfloat16*>(destination.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_ple_batch_update(const Tensor& hyper, const Tensor& gathered_fp8,
                                 const FlashNextPleWeights& weights, Tensor& states,
                                 const Tensor& valid_columns, const Tensor& source_slots,
                                 const Tensor& destination_slots, std::int32_t width,
                                 std::int32_t batch, Tensor& destination,
                                 WorkspaceArena& workspace, cudaStream_t stream,
                                 Bf16GemmContext* bf16_gemm) {
    NINFER_PERF_SCOPE("ple.update", hyper.ne[1], batch, 0, flash_next_work::ple(hyper.ne[1]));

    if (width <= 0 || batch <= 0 || batch > 8 || hyper.ne[1] != width * batch ||
        states.dtype != DType::BF16 || !states.is_contiguous() || states.ne[0] != kHyper ||
        states.ne[1] != kState || source_slots.dtype != DType::I32 ||
        destination_slots.dtype != DType::I32 || source_slots.ne[0] != batch ||
        destination_slots.ne[0] != batch ||
        (valid_columns.data != nullptr &&
         (valid_columns.dtype != DType::I32 || valid_columns.ne[0] != batch))) {
        throw std::invalid_argument("flash_next_ple_batch_update: invalid state geometry");
    }
    validate(hyper, gathered_fp8, weights,
             states.slice(2, 0, 1).view({kHyper, kState}), destination);
    const int tokens = width * batch;
    auto scope = workspace.scope();
    Tensor embedding = workspace.alloc(DType::BF16, {kHidden, tokens});
    dequantize_embedding_kernel<<<grid_for(embedding.numel()), 256, 0, stream>>>(
        static_cast<const std::uint8_t*>(gathered_fp8.data),
        static_cast<const __nv_bfloat16*>(weights.embedding_scale.data),
        static_cast<__nv_bfloat16*>(embedding.data), embedding.numel());
    Tensor key = workspace.alloc(DType::BF16, {kHyper, tokens});
    Tensor value = workspace.alloc(DType::BF16, {kHidden, tokens});
    linear(embedding, weights.key_projection, key, stream, bf16_gemm);
    linear(embedding, weights.value_projection, value, stream, bf16_gemm);
    Tensor gated = workspace.alloc(DType::BF16, {kHyper, tokens});
    gate_kernel<<<dim3(kStreams, static_cast<unsigned int>(tokens)), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hyper.data), static_cast<const __nv_bfloat16*>(key.data),
        static_cast<const __nv_bfloat16*>(value.data),
        static_cast<const __nv_bfloat16*>(weights.query_norm.data),
        static_cast<const __nv_bfloat16*>(weights.key_norm.data),
        static_cast<__nv_bfloat16*>(gated.data));
    Tensor normalized = workspace.alloc(DType::BF16, {kHyper, tokens});
    grouped_norm_kernel<<<dim3(kStreams, static_cast<unsigned int>(tokens)), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gated.data),
        static_cast<const __nv_bfloat16*>(weights.convolution_norm.data),
        static_cast<__nv_bfloat16*>(normalized.data));
    const dim3 grid(40, static_cast<unsigned int>(batch));
    dilated_conv_batch_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(normalized.data),
        static_cast<const __nv_bfloat16*>(weights.convolution.data),
        static_cast<__nv_bfloat16*>(states.data),
        valid_columns.data == nullptr ? nullptr : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<const std::int32_t*>(source_slots.data),
        static_cast<const std::int32_t*>(destination_slots.data),
        static_cast<const __nv_bfloat16*>(gated.data),
        static_cast<__nv_bfloat16*>(destination.data), width, batch,
        static_cast<std::int64_t>(kHyper) * kState);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_ple_replay_record(const Tensor& hyper, const Tensor& gathered_fp8,
                                  const FlashNextPleWeights& weights, const Tensor& states,
                                  const Tensor& valid_columns, const Tensor& source_slots,
                                  std::int32_t width, std::int32_t batch, Tensor& records,
                                  Tensor& destination, WorkspaceArena& workspace,
                                  cudaStream_t stream, Bf16GemmContext* bf16_gemm) {
    NINFER_PERF_SCOPE("ple.record", hyper.ne[1], batch, 0, flash_next_work::ple(hyper.ne[1]));

    if (width <= 0 || batch <= 0 || batch > 8 || hyper.ne[1] != width * batch ||
        states.dtype != DType::BF16 || !states.is_contiguous() || states.ne[0] != kHyper ||
        states.ne[1] != kState || source_slots.dtype != DType::I32 ||
        source_slots.ne[0] != batch || valid_columns.dtype != DType::I32 ||
        valid_columns.ne[0] != batch || records.dtype != DType::BF16 ||
        !records.is_contiguous() || records.ne[0] != kHyper ||
        records.ne[1] != width || records.ne[2] != batch) {
        throw std::invalid_argument("flash_next_ple_replay_record: invalid state geometry");
    }
    validate(hyper, gathered_fp8, weights,
             states.slice(2, 0, 1).view({kHyper, kState}), destination);
    const int tokens = width * batch;
    auto scope = workspace.scope();
    Tensor embedding = workspace.alloc(DType::BF16, {kHidden, tokens});
    dequantize_embedding_kernel<<<grid_for(embedding.numel()), 256, 0, stream>>>(
        static_cast<const std::uint8_t*>(gathered_fp8.data),
        static_cast<const __nv_bfloat16*>(weights.embedding_scale.data),
        static_cast<__nv_bfloat16*>(embedding.data), embedding.numel());
    Tensor key = workspace.alloc(DType::BF16, {kHyper, tokens});
    Tensor value = workspace.alloc(DType::BF16, {kHidden, tokens});
    // Replay must produce the same represented transition as sequential ordinary decode.
    // Multi-column BF16 GEMMs may select a different reduction kernel than M=1 and perturb
    // a verified token before the remaining layers amplify the difference.  Preserve the
    // ordinary one-column projection route; gate, convolution, and recording remain batched.
    for (int token = 0; token < tokens; ++token) {
        Tensor embedding_column = embedding.slice(1, token, 1);
        Tensor key_column = key.slice(1, token, 1);
        Tensor value_column = value.slice(1, token, 1);
        linear(embedding_column, weights.key_projection, key_column, stream, bf16_gemm);
        linear(embedding_column, weights.value_projection, value_column, stream, bf16_gemm);
    }
    Tensor gated = workspace.alloc(DType::BF16, {kHyper, tokens});
    gate_kernel<<<dim3(kStreams, static_cast<unsigned int>(tokens)), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hyper.data), static_cast<const __nv_bfloat16*>(key.data),
        static_cast<const __nv_bfloat16*>(value.data),
        static_cast<const __nv_bfloat16*>(weights.query_norm.data),
        static_cast<const __nv_bfloat16*>(weights.key_norm.data),
        static_cast<__nv_bfloat16*>(gated.data));
    Tensor normalized = workspace.alloc(DType::BF16, {kHyper, tokens});
    grouped_norm_kernel<<<dim3(kStreams, static_cast<unsigned int>(tokens)), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gated.data),
        static_cast<const __nv_bfloat16*>(weights.convolution_norm.data),
        static_cast<__nv_bfloat16*>(normalized.data));
    const dim3 grid(40, static_cast<unsigned int>(batch));
    dilated_conv_record_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(normalized.data),
        static_cast<const __nv_bfloat16*>(weights.convolution.data),
        static_cast<const __nv_bfloat16*>(states.data),
        static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<const std::int32_t*>(source_slots.data),
        static_cast<__nv_bfloat16*>(records.data),
        static_cast<const __nv_bfloat16*>(gated.data),
        static_cast<__nv_bfloat16*>(destination.data), width, batch,
        static_cast<std::int64_t>(kHyper) * kState);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_ple_replay_fold(const Tensor& records, Tensor& states,
                                std::span<const FlashNextPleFoldRow> rows,
                                cudaStream_t stream) {
    const int width = records.ne[1];
    if (rows.empty() || rows.size() > 8 || width <= 0 || records.dtype != DType::BF16 ||
        !records.is_contiguous() || records.ne[0] != kHyper ||
        records.ne[2] < static_cast<std::int32_t>(rows.size()) || states.dtype != DType::BF16 ||
        !states.is_contiguous() || states.ne[0] != kHyper || states.ne[1] != kState) {
        throw std::invalid_argument("flash_next_ple_replay_fold: invalid geometry");
    }
    PleFoldKernelRows packed{};
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].source_state_slot < 0 || rows[i].source_state_slot >= states.ne[2] ||
            rows[i].destination_state_slot < 0 || rows[i].destination_state_slot >= states.ne[2] ||
            rows[i].commit_columns < 0 || rows[i].commit_columns > width) {
            throw std::invalid_argument("flash_next_ple_replay_fold: invalid row");
        }
        packed.rows[i] = rows[i];
    }
    const dim3 grid(40, static_cast<unsigned int>(rows.size()));
    ple_fold_kernel<<<grid, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(records.data),
                                              static_cast<__nv_bfloat16*>(states.data), packed,
                                              nullptr, static_cast<int>(rows.size()), width,
                                              static_cast<std::int64_t>(kHyper) * kState);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_ple_replay_fold_device(const Tensor& records, Tensor& states,
                                       const Tensor& device_rows, std::int32_t row_count,
                                       cudaStream_t stream) {
    const int width = records.ne[1];
    if (row_count <= 0 || row_count > 8 || width <= 0 || records.dtype != DType::BF16 ||
        !records.is_contiguous() || records.ne[0] != kHyper || records.ne[2] < row_count ||
        states.dtype != DType::BF16 || !states.is_contiguous() || states.ne[0] != kHyper ||
        states.ne[1] != kState || device_rows.data == nullptr || device_rows.dtype != DType::I32 ||
        !device_rows.is_contiguous() ||
        device_rows.numel() < static_cast<std::int64_t>(4) * row_count) {
        throw std::invalid_argument("flash_next_ple_replay_fold: invalid device geometry");
    }
    const dim3 grid(40, static_cast<unsigned int>(row_count));
    ple_fold_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(records.data), static_cast<__nv_bfloat16*>(states.data),
        PleFoldKernelRows{}, static_cast<const std::int32_t*>(device_rows.data), row_count, width,
        static_cast<std::int64_t>(kHyper) * kState);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
