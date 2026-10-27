#include "ops/linear/fp8/fp8_flash_next.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_flash_next_route.cuh"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/launcher/hyperconnection_math.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::flash_next {
namespace {

using HcDownGeometry    = Fp8Geometry<320, 10240>;
using HcUpGeometry      = Fp8Geometry<10240, 320>;
using HcDownGemv        = Fp8A16GemvSchedule<4, 1, 16, 4, Fp8CodeCache::Streaming, 4, 4>;
using QueryGateGeometry = Fp8Geometry<12288, 2560>;
using QueryGateGemv     = Fp8A16GemvSchedule<8, 2, 16, 4, Fp8CodeCache::Default, 1, 2>;

// The BF16 route rounds the projection to BF16 before the 0.25-scaled SiLU; this epilogue keeps
// that represented boundary.
struct Fp8HcDownSiluEpilogue {
    __device__ __forceinline__ float apply(int, int, float value) const {
        const float represented = __bfloat162float(__float2bfloat16_rn(value)) * 0.25F;
        return represented / (1.0F + expf(-represented));
    }
};

// Packed rows alternate one 256-wide query head and its 256-wide gate head.
struct Fp8QueryGateOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* gate;

    __device__ __forceinline__ void store(int parent_row, int, float value) const {
        constexpr int kHeadWidth       = 256;
        constexpr int kPackedHeadWidth = 2 * kHeadWidth;
        const int head                 = parent_row / kPackedHeadWidth;
        const int within_head          = parent_row - head * kPackedHeadWidth;
        const __nv_bfloat16 stored     = __float2bfloat16_rn(value);
        if (within_head < kHeadWidth) {
            query[head * kHeadWidth + within_head] = stored;
        } else {
            gate[head * kHeadWidth + within_head - kHeadWidth] = stored;
        }
    }
};

// The HyperConnection up projection with the gate mix in its epilogue: the stream-quad row
// policy hands each warp the four stream logits of a position, so the gate logits never round
// trip through memory. The first row CTA of each token tile also finishes the injection from the
// normalization's partials.
struct Fp8NoOutput {};

struct Fp8HcUpMixEpilogue {
    const __nv_bfloat16* normalized;
    __nv_bfloat16* block_input;
    const float* injection_partials;
    __nv_bfloat16* injection;

    __device__ __forceinline__ void mix_streams(int position, int token,
                                                const float* logits) const {
        hyperconnection::gate_mix_position(logits, normalized, block_input, position, token);
    }

    __device__ __forceinline__ void finish_cta(int token_begin, int live_columns, int tid) const {
        if (injection == nullptr) return;
        for (int item = tid; item < hyperconnection::kStreams * live_columns;
             item += static_cast<int>(blockDim.x)) {
            hyperconnection::finish_injection(injection_partials, injection,
                                              item % hyperconnection::kStreams,
                                              token_begin + item / hyperconnection::kStreams);
        }
    }
};

template <class Geometry>
void validate_input(const Tensor& x, const Weight& weight, const char* label) {
    if (x.dtype != DType::BF16 || !x.is_contiguous() || x.ne[0] != Geometry::kInputRows ||
        x.ne[1] <= 0 || weight.n != Geometry::kOutputRows || weight.k != Geometry::kInputRows) {
        throw std::invalid_argument(std::string(label) + ": invalid exact problem");
    }
    (void)validate_fp8_weight(weight, label);
}

} // namespace

void launch_fp8_hc_down_silu(const Tensor& x, const Weight& weight, Tensor& out,
                             cudaStream_t stream) {
    validate_input<HcDownGeometry>(x, weight, "FP8 HyperConnection down-SiLU");
    if (out.dtype != DType::BF16 || !out.is_contiguous() ||
        out.numel() != static_cast<std::int64_t>(HcDownGeometry::kOutputRows) * x.ne[1]) {
        throw std::invalid_argument("FP8 HyperConnection down-SiLU: invalid output");
    }
    launch_fp8_dense_a16<HcDownGeometry, HcDownGemv>(
        x, weight,
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), HcDownGeometry::kOutputRows},
        Fp8HcDownSiluEpilogue{}, stream);
}

void launch_fp8_hc_up_mix(const Tensor& low_rank, const Weight& weight, const Tensor& normalized,
                          const Tensor* injection_partials, Tensor& block_input, Tensor* injection,
                          cudaStream_t stream) {
    validate_input<HcUpGeometry>(low_rank, weight, "FP8 HyperConnection up-mix");
    const int tokens = low_rank.ne[1];
    if (tokens < 2 || tokens > 64 || normalized.dtype != DType::BF16 ||
        !normalized.is_contiguous() ||
        normalized.numel() != static_cast<std::int64_t>(hyperconnection::kHyper) * tokens ||
        block_input.dtype != DType::BF16 || !block_input.is_contiguous() ||
        block_input.numel() != static_cast<std::int64_t>(hyperconnection::kHidden) * tokens ||
        (injection != nullptr) != (injection_partials != nullptr) ||
        (injection != nullptr &&
         (injection->dtype != DType::BF16 || !injection->is_contiguous() ||
          injection->numel() != static_cast<std::int64_t>(hyperconnection::kStreams) * tokens ||
          injection_partials->dtype != DType::FP32 || !injection_partials->is_contiguous() ||
          injection_partials->numel() !=
              static_cast<std::int64_t>(hyperconnection::kInjectionPartialsPerToken) * tokens))) {
        throw std::invalid_argument("FP8 HyperConnection up-mix: invalid operands");
    }
    constexpr int K       = HcUpGeometry::kInputRows;
    constexpr int kWide   = sliced_k_warps<K, 8>();
    constexpr int kNarrow = sliced_k_warps<K, 4>();
    using Rows            = Fp8StreamQuadRows<hyperconnection::kHidden>;
    const Fp8HcUpMixEpilogue epilogue{
        static_cast<const __nv_bfloat16*>(normalized.data),
        static_cast<__nv_bfloat16*>(block_input.data),
        injection_partials != nullptr ? static_cast<const float*>(injection_partials->data)
                                      : nullptr,
        injection != nullptr ? static_cast<__nv_bfloat16*>(injection->data) : nullptr};
    const auto operands = fp8_a16_operands(low_rank, weight);
    // The same sliced-K instances as the unfused up projection at these widths.
    if (tokens <= 8) {
        return launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<8, kWide, 2>, K>>(
            operands, Fp8NoOutput{}, epilogue, stream, Rows{});
    }
    if (tokens <= 16) {
        return launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<16, kWide, 2>, K>>(
            operands, Fp8NoOutput{}, epilogue, stream, Rows{});
    }
    if (tokens <= 32) {
        return launch_fp8_a16_sliced_k_mma<
            Fp8ScheduleInstance<Fp8SlicedInstance<16, kNarrow, 2>, K>>(operands, Fp8NoOutput{},
                                                                       epilogue, stream, Rows{});
    }
    launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<32, kNarrow, 1>, K>>(
        operands, Fp8NoOutput{}, epilogue, stream, Rows{});
}

void launch_fp8_query_gate_decode(const Tensor& x, const Weight& weight, Tensor& query,
                                  Tensor& gate, cudaStream_t stream) {
    validate_input<QueryGateGeometry>(x, weight, "FP8 query/gate decode");
    if (x.ne[1] != 1 || query.dtype != DType::BF16 || gate.dtype != DType::BF16 ||
        !query.is_contiguous() || !gate.is_contiguous() || query.numel() != 6144 ||
        gate.numel() != 6144) {
        throw std::invalid_argument("FP8 query/gate decode: invalid outputs");
    }
    launch_fp8_a16_gemv<Fp8ScheduleInstance<QueryGateGemv, QueryGateGeometry::kInputRows>>(
        fp8_a16_operands(x, weight),
        Fp8QueryGateOutput{static_cast<__nv_bfloat16*>(query.data),
                           static_cast<__nv_bfloat16*>(gate.data)},
        LinearIdentityEpilogue{}, stream);
}

} // namespace ninfer::ops::detail::flash_next
