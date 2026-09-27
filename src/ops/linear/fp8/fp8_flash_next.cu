#include "ops/linear/fp8/fp8_flash_next.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/fp8/fp8_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::flash_next {
namespace {

using HcDownGeometry    = Fp8Geometry<320, 10240>;
using HcDownSchedule    = Fp8GemvSchedule<4, 1, 16, 4, Fp8CodeCache::Streaming, 4, 4>;
using QueryGateGeometry = Fp8Geometry<12288, 2560>;
using QueryGateSchedule = Fp8GemvSchedule<8, 2, 16, 4, Fp8CodeCache::Default, 1, 2>;

// The BF16 route rounds the projection to BF16 before the 0.25-scaled SiLU; this epilogue keeps
// that represented boundary.
struct Fp8HcDownSiluEpilogue {
    __device__ __forceinline__ float apply(std::int32_t, std::int32_t, float value) const {
        const float represented = __bfloat162float(__float2bfloat16_rn(value)) * 0.25F;
        return represented / (1.0F + expf(-represented));
    }
};

// Packed rows alternate one 256-wide query head and its 256-wide gate head.
struct Fp8QueryGateOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* gate;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t,
                                          float value) const {
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

template <class Geometry>
void validate_decode(const Tensor& x, const Weight& weight, const char* label) {
    if (x.dtype != DType::BF16 || !x.is_contiguous() || x.ne[0] != Geometry::kInputRows ||
        x.ne[1] != 1 || weight.n != Geometry::kOutputRows || weight.k != Geometry::kInputRows) {
        throw std::invalid_argument(std::string(label) + ": invalid exact problem");
    }
    (void)validate_fp8_weight(weight, label);
}

} // namespace

void launch_fp8_hc_down_silu_decode(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream) {
    validate_decode<HcDownGeometry>(x, weight, "FP8 HyperConnection decode down-SiLU");
    if (out.dtype != DType::BF16 || !out.is_contiguous() ||
        out.numel() != HcDownGeometry::kOutputRows) {
        throw std::invalid_argument("FP8 HyperConnection decode down-SiLU: invalid output");
    }
    const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                     HcDownGeometry::kOutputRows};
    fp8_gemv_kernel<HcDownGeometry, HcDownSchedule, Fp8ContiguousOutput, Fp8GemvIdentityRows, false,
                    Fp8HcDownSiluEpilogue>
        <<<HcDownGeometry::kOutputRows / HcDownSchedule::kRowsPerCta, HcDownSchedule::kThreads, 0,
           stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                     static_cast<const std::uint8_t*>(weight.qdata),
                     static_cast<const __nv_bfloat16*>(weight.scales), output, {}, {});
    CUDA_CHECK(cudaGetLastError());
}

void launch_fp8_query_gate_decode(const Tensor& x, const Weight& weight, Tensor& query,
                                  Tensor& gate, cudaStream_t stream) {
    validate_decode<QueryGateGeometry>(x, weight, "FP8 query/gate decode");
    if (query.dtype != DType::BF16 || gate.dtype != DType::BF16 || !query.is_contiguous() ||
        !gate.is_contiguous() || query.numel() != 6144 || gate.numel() != 6144) {
        throw std::invalid_argument("FP8 query/gate decode: invalid outputs");
    }
    const Fp8QueryGateOutput output{static_cast<__nv_bfloat16*>(query.data),
                                    static_cast<__nv_bfloat16*>(gate.data)};
    fp8_gemv_kernel<QueryGateGeometry, QueryGateSchedule, Fp8QueryGateOutput>
        <<<QueryGateGeometry::kOutputRows / QueryGateSchedule::kRowsPerCta,
           QueryGateSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::flash_next
