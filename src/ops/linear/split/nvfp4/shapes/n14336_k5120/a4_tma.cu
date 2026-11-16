#include "ops/linear/split/nvfp4/shapes/n14336_k5120/launch.h"

#include "ops/linear/kernels/nvfp4/nvfp4_a4_tma_shared.h"

namespace ninfer::ops::detail::linear_split::nvfp4_n14336_k5120 {

void launch_a4_tma(const Nvfp4A4Operands& p, std::span<const Tensor> outputs, cudaStream_t stream) {
    launch_nvfp4_a4_tma_segmented_6144_1024(
        p,
        Nvfp4Segmented6144x1024{static_cast<__nv_bfloat16*>(outputs[0].data),
                                static_cast<__nv_bfloat16*>(outputs[1].data),
                                static_cast<__nv_bfloat16*>(outputs[2].data),
                                static_cast<__nv_bfloat16*>(outputs[3].data)},
        stream);
}

} // namespace ninfer::ops::detail::linear_split::nvfp4_n14336_k5120
