#include "ops/linear/fp8/fp8_flash_next.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_flash_next_route.cuh"
#include "ops/linear/fp8/fp8_format.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::flash_next {
namespace {

using HcDownGeometry    = Fp8Geometry<320, 10240>;
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
