#include "ops/linear_attention/gated_delta_net/input_conv/nvfp4/tma_launch.h"

#include "ops/linear/kernels/nvfp4/nvfp4_a4_tma_shared.h"

namespace ninfer::ops::detail {
void launch_nvfp4_a4_tma_gdn(const Nvfp4A4Operands& p, __nv_bfloat16* qkv, __nv_bfloat16* z,
                             cudaStream_t stream) {
    launch_nvfp4_a4_tma_segmented_10240_6144(p, Nvfp4Segmented10240x6144{qkv, z}, stream);
}
} // namespace ninfer::ops::detail
