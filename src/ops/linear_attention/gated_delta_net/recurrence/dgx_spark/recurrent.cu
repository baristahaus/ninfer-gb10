#include "ops/linear_attention/gated_delta_net/recurrence/launch.h"
#include "ops/linear_attention/gated_delta_net/recurrence/recurrent_io.cuh"
#include "ops/linear_attention/gated_delta_net/kernels/recurrent_subgroup.cuh"
#include "core/device.h"

namespace ninfer::ops::detail::gated_delta_net {
namespace {

template <bool Normalize, bool Prefetch>
__global__ __launch_bounds__(128, 2) void direct_kernel(DirectAccess access) {
    extern __shared__ __align__(16) char backing[];
    subgroup::run<16, Normalize, subgroup::Mode::Update, Prefetch>(
        access, access.coordinates(), access.width,
        reinterpret_cast<subgroup::Window<subgroup::kWindowSize>*>(backing));
}

template <bool Normalize>
__global__ __launch_bounds__(128, 2) void batch_kernel(BatchUpdateAccess access) {
    extern __shared__ __align__(16) char backing[];
    subgroup::run<16, Normalize, subgroup::Mode::Update, true>(
        access, access.coordinates(), 1, reinterpret_cast<subgroup::Window<1>*>(backing));
}

template <int QkHeads, int ValueHeads, bool Normalize>
__global__ __launch_bounds__(128, 2) void packed_batch_kernel(PackedBatchUpdateAccess<QkHeads,
                                                                                 ValueHeads>
                                                                  access) {
    extern __shared__ __align__(16) char backing[];
    subgroup::run<16, Normalize, subgroup::Mode::Update, true>(
        access, access.coordinates(), 1, reinterpret_cast<subgroup::Window<1>*>(backing));
}

template <bool Normalize>
void launch_direct(const DirectAccess& access, cudaStream_t stream) {
    const dim3 grid(access.heads.H_v, 1, kStateDim / kBlockDv), block(32, kNumWarps);
    if (access.width <= subgroup::kWindowSize) {
        direct_kernel<Normalize, true>
            <<<grid, block, sizeof(subgroup::Window<subgroup::kWindowSize>), stream>>>(access);
    } else {
        direct_kernel<Normalize, false><<<grid, block, 0, stream>>>(access);
    }
}

} // namespace

void launch_recurrent_inout(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk,
                            const Tensor& state_in, Tensor& state_out, Tensor& out,
                            cudaStream_t stream) {
    const DirectAccess access{static_cast<const __nv_bfloat16*>(q.data),
                              static_cast<const __nv_bfloat16*>(k.data),
                              static_cast<const __nv_bfloat16*>(v.data),
                              static_cast<const float*>(g.data),
                              static_cast<const float*>(beta.data),
                              static_cast<const float*>(state_in.data),
                              static_cast<float*>(state_out.data),
                              static_cast<__nv_bfloat16*>(out.data),
                              HeadMap::of(q.ne[1], v.ne[1]),
                              q.ne[2],
                              scale};
    if (normalize_qk)
        launch_direct<true>(access, stream);
    else
        launch_direct<false>(access, stream);
    CUDA_CHECK(cudaGetLastError());
}

void launch_recurrent_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   bool normalize_qk, Tensor& states, const Tensor& source_slots,
                                   const Tensor& destination_slots, Tensor& out,
                                   cudaStream_t stream) {
    const BatchUpdateAccess access{static_cast<const __nv_bfloat16*>(q.data),
                                   static_cast<const __nv_bfloat16*>(k.data),
                                   static_cast<const __nv_bfloat16*>(v.data),
                                   static_cast<const float*>(g.data),
                                   static_cast<const float*>(beta.data),
                                   static_cast<const float*>(states.data),
                                   static_cast<float*>(states.data),
                                   static_cast<const std::int32_t*>(source_slots.data),
                                   static_cast<const std::int32_t*>(destination_slots.data),
                                   static_cast<__nv_bfloat16*>(out.data),
                                   HeadMap::of(q.ne[1], v.ne[1]),
                                   states.nb[3] / std::int64_t(sizeof(float)),
                                   scale};
    const dim3 grid(v.ne[1], q.ne[3], kStateDim / kBlockDv), block(32, kNumWarps);
    if (normalize_qk) {
        batch_kernel<true><<<grid, block, sizeof(subgroup::Window<1>), stream>>>(access);
    } else {
        batch_kernel<false><<<grid, block, sizeof(subgroup::Window<1>), stream>>>(access);
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_recurrent_batch_update_packed_qkv(
    const Tensor& packed_qkv, std::int32_t qk_heads, std::int32_t value_heads,
    const Tensor& g, const Tensor& beta, float scale, Tensor& ssm_states,
    const Tensor& source_state_slots, const Tensor& destination_state_slots,
    Tensor& out, cudaStream_t stream) {
    if (qk_heads != 16 || value_heads != 48 || packed_qkv.ne[0] != 10240 ||
        packed_qkv.ne[1] <= 0 || packed_qkv.ne[1] > 8 || !packed_qkv.is_contiguous()) {
        throw std::invalid_argument("packed GDN batch update: unsupported exact geometry");
    }
    using Access = PackedBatchUpdateAccess<16, 48>;
    const Access access{static_cast<const __nv_bfloat16*>(packed_qkv.data),
                        static_cast<const float*>(g.data),
                        static_cast<const float*>(beta.data),
                        static_cast<const float*>(ssm_states.data),
                        static_cast<float*>(ssm_states.data),
                        static_cast<const std::int32_t*>(source_state_slots.data),
                        static_cast<const std::int32_t*>(destination_state_slots.data),
                        static_cast<__nv_bfloat16*>(out.data),
                        HeadMap::of(16, 48),
                        ssm_states.nb[3] / std::int64_t(sizeof(float)),
                        scale};
    const dim3 grid(48, static_cast<unsigned>(packed_qkv.ne[1]), kStateDim / kBlockDv);
    const dim3 block(32, kNumWarps);
    packed_batch_kernel<16, 48, true>
        <<<grid, block, sizeof(subgroup::Window<1>), stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::gated_delta_net
