#include "ops/launcher/rmsnorm_pack_tail.h"

#include "core/device.h"
#include "ops/kernel/rmsnorm_pack_tail.cuh"

namespace ninfer::ops::detail {

void rmsnorm_pack_tail_launch(const Tensor& input, const Tensor& weight, Tensor& output,
                              cudaStream_t stream) {
    const dim3 grid(input.ne[1] - 1, input.ne[2]);
    // 1280 pairs per row for D=2560 and 2560 for D=5120; both keep five pairs per thread.
    if (input.ne[0] == 2560) {
        constexpr int threads = 256;
        rmsnorm_pack_tail_kernel<threads, 2560>
            <<<grid, threads, 0, stream>>>(static_cast<const __nv_bfloat162*>(input.data),
                                           static_cast<const __nv_bfloat162*>(weight.data),
                                           static_cast<__nv_bfloat162*>(output.data), input.ne[1]);
    } else {
        constexpr int threads = 512;
        rmsnorm_pack_tail_kernel<threads, 5120>
            <<<grid, threads, 0, stream>>>(static_cast<const __nv_bfloat162*>(input.data),
                                           static_cast<const __nv_bfloat162*>(weight.data),
                                           static_cast<__nv_bfloat162*>(output.data), input.ne[1]);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
