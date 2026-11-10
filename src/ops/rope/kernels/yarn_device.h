#pragma once

// The YaRN inputs of one rope launch, resolved on the host and consumed by the device kernels.
//
// Kept in its own header, free of CUDA types, so that both the host-side dispatch declaration and
// the CUDA kernel header can carry it: src/ops/rope/apply/rope.cpp is compiled by the host
// compiler and cannot include a .cuh.

#include <cstdint>

namespace ninfer::ops::detail {

// `inv_frequency` is the 32-pair NTK-by-parts table and `scale` the reference's attention
// temperature, both already computed exactly as the reference does; the kernel only selects and
// applies them.
//
// `enabled == false` is the whole disabled path: every text kernel then reads the compiled-in
// table and applies no scale, which is bit-identical to the engine before YaRN existed. The table
// is only meaningful for rotary_dim 64 (32 pairs), which the host enforces, so `pair_count` also
// guards the generic kernel against indexing past the table on any other rotary width.
struct RopeYarnDevice {
    float inv_frequency[32] = {};
    float scale             = 1.0F;
    std::int32_t pair_count = 0;
    bool enabled            = false;
};

} // namespace ninfer::ops::detail
