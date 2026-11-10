#pragma once
#include "ops/linear/kernels/nvfp4/nvfp4_launch.h"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_plan.h"

namespace ninfer::ops::detail {
struct Nvfp4LinearShape {
    std::int32_t n, k;
    Nvfp4Launch a16;
    void (*a4)(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace, cudaStream_t);
    bool (*uses_a4)(std::int32_t min_tokens, std::int32_t max_tokens);
};

extern const Nvfp4LinearShape kNvfp4N12288K4096;
extern const Nvfp4LinearShape kNvfp4N4096K6144;
extern const Nvfp4LinearShape kNvfp4N14336K5120;
extern const Nvfp4LinearShape kNvfp4N16384K5120;
extern const Nvfp4LinearShape kNvfp4N34816K5120;
extern const Nvfp4LinearShape kNvfp4N5120K6144;
extern const Nvfp4LinearShape kNvfp4N5120K17408;
// The sharded 27B lane's MLP shapes: the fused gate_up half and one gate/up role.
extern const Nvfp4LinearShape kNvfp4N17408K5120;
extern const Nvfp4LinearShape kNvfp4N8704K5120;
// mlp/down at width 2: the Columns shard over 17408 halves K, so the rank presents 5120x8704 where
// width 1 presents 5120x17408.
extern const Nvfp4LinearShape kNvfp4N5120K8704;
} // namespace ninfer::ops::detail
