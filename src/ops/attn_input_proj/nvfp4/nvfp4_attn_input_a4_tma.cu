#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_a4_tma_launch.h"

#include "ops/linear/kernels/nvfp4/nvfp4_a4_tma_shared.h"

namespace ninfer::ops::detail {
void launch_nvfp4_a4_tma_attention(const Nvfp4A4Operands& p, __nv_bfloat16* query,
                                   __nv_bfloat16* gate, __nv_bfloat16* key, __nv_bfloat16* value,
                                   cudaStream_t stream) {
    launch_nvfp4_a4_tma_segmented_6144_1024(p, Nvfp4Segmented6144x1024{query, key, gate, value},
                                            stream);
}
} // namespace ninfer::ops::detail
