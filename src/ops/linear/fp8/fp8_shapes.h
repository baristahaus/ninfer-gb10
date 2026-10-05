#pragma once
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

namespace ninfer::ops::detail {
struct Fp8LinearShape {
    std::int32_t n, k;
    Fp8Launch a16;
    void (*a8)(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace, cudaStream_t);
    bool (*uses_a8)(std::int32_t min_tokens, std::int32_t max_tokens);
};

extern const Fp8LinearShape kFp8N14336K5120;
extern const Fp8LinearShape kFp8N16384K5120;
extern const Fp8LinearShape kFp8N34816K5120;
extern const Fp8LinearShape kFp8N5120K6144;
extern const Fp8LinearShape kFp8N5120K17408;
extern const Fp8LinearShape kFp8N248320K5120;
extern const Fp8LinearShape kFp8N12288K2560;
extern const Fp8LinearShape kFp8N10240K2560;
extern const Fp8LinearShape kFp8N6144K2560;
extern const Fp8LinearShape kFp8N2560K6144;
extern const Fp8LinearShape kFp8N512K2560;
extern const Fp8LinearShape kFp8N640K2560;
extern const Fp8LinearShape kFp8N248320K2560;
} // namespace ninfer::ops::detail
