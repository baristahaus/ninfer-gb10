#pragma once
#include "ops/linear/kernels/fp8/fp8_launch.h"
#include "ops/linear/kernels/fp8/fp8_a8_plan.h"

namespace ninfer::ops::detail {
struct Fp8LinearShape {
    std::int32_t n, k;
    Fp8Launch a16;
    void (*a8)(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace, cudaStream_t);
    bool (*uses_a8)(std::int32_t min_tokens, std::int32_t max_tokens);
    std::size_t (*partial_capacity_bytes)(std::int32_t max_tokens) = nullptr;
};

extern const Fp8LinearShape kFp8N14336K5120;
extern const Fp8LinearShape kFp8N16384K5120;
extern const Fp8LinearShape kFp8N34816K5120;
extern const Fp8LinearShape kFp8N5120K6144;
extern const Fp8LinearShape kFp8N5120K17408;
extern const Fp8LinearShape kFp8N248320K5120;
extern const Fp8LinearShape kFp8N10240K2560;
extern const Fp8LinearShape kFp8N6144K2560;
extern const Fp8LinearShape kFp8N2560K6144;
extern const Fp8LinearShape kFp8N320K10240;
extern const Fp8LinearShape kFp8N248320K2560;
extern const Fp8LinearShape kFp8ProjN12288K2560;
extern const Fp8LinearShape kFp8ProjN10240K320;
extern const Fp8LinearShape kFp8ProjN2560K640;

// The sharded shapes of the 27B's FP8 parameters at tensor-parallel width 2, one per fused op view.
// A Rows shard halves the output rows of the fused object: 14336 -> 7168 (the attention qkv),
// 16384 -> 8192 (the GDN qkvz) and 248320 -> 124160 (the output head). Each inherits the schedule
// set and token bands of the shape it halves, because halving an axis changes the tile count and
// not the tile shapes.
extern const Fp8LinearShape kFp8N7168K5120;
extern const Fp8LinearShape kFp8N8192K5120;
extern const Fp8LinearShape kFp8N124160K5120;

// The ROLE-SPLIT shapes: the same weights as the fused objects above, viewed one role at a time.
// At tensor-parallel width 2 the sharded parts of a fused object are non-contiguous, so
// `projection_region_count` exceeds its threshold and `execution/profile.cpp` describes each role
// separately instead of the whole fused projection:
//   attention  role per parameter: query 3072, key 512, gate 3072, value 512  -> N3072K5120, N512K5120
//   GDN        role 0 = query+key+value fused: 1024+1024+3072 = 5120        -> N5120K5120
//              role 1 = z alone: 3072                                       -> N3072K5120
// Width 1 never reaches these: the four parameters bind one contiguous region of the fused object
// there, the count stays at its threshold, and the whole-projection branch runs.
extern const Fp8LinearShape kFp8N5120K5120;
extern const Fp8LinearShape kFp8N3072K5120;
extern const Fp8LinearShape kFp8N512K5120;
// The dense MLP role split (execution/profile.cpp:26-34): gate and up described individually when
// their bindings are non-contiguous, which is the width-2 case. Each rank half is 8704 rows.
extern const Fp8LinearShape kFp8N8704K5120;
} // namespace ninfer::ops::detail
