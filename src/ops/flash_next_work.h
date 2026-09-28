#pragma once

#include "core/performance.h"

#ifdef NINFER_PERFORMANCE_TRACE
#    include <algorithm>

namespace ninfer::ops::flash_next_work {
using performance::Work;
using U = std::uint64_t;

// These models count the dominant exact projection formulas and public input/output traffic.
// They intentionally exclude private scratch, launch cost, transcendentals, and padding in MMA
// tiles. T is the launch envelope (including padded columns), not committed output tokens.
constexpr Work dense_stored(U parameters, U weight_bytes, U tokens, U io_bytes) {
    const U bytes = weight_bytes + io_bytes;
    return {bytes, bytes, 2 * parameters * tokens, 0, 0};
}

constexpr Work dense(U parameters, U tokens, U io_bytes) {
    return dense_stored(parameters, 2 * parameters, tokens, io_bytes);
}

// Stored bytes of one [rows,columns] projection: BF16 words, or E4M3 codes plus one BF16
// multiplier per row.
constexpr U projection_bytes(U rows, U columns, bool fp8) {
    return fp8 ? rows * columns + 2 * rows : 2 * rows * columns;
}

constexpr Work moe(U t, bool nvfp4, bool fp8_shared_gate_up, bool fp8_shared_down) {
    constexpr U expert        = 3 * 2560 * 640;
    constexpr U shared_router = expert + 2560 * (512 + 1);
    // Unique experts are data dependent: all tokens may share ten, or use disjoint top-10s.
    // NVFP4 has one code nibble and one E4M3 scale per 16 values, plus four FP32
    // gate-up/down weight/input divisors per expert. BF16 has no such scale planes. The
    // shared expert is BF16 or row-scaled FP8; the router and shared gate stay BF16.
    const U expert_bytes = nvfp4 ? expert / 2 + expert / 16 + 16 : 2 * expert;
    const U shared_bytes = projection_bytes(1280, 2560, fp8_shared_gate_up) +
                           projection_bytes(2560, 640, fp8_shared_down) + 2 * 2560 * (512 + 1);
    const U common       = shared_bytes + 2 * 2560 * t * 2;
    const U routed_flops = 2 * expert * 10 * t;
    return {common + 10 * expert_bytes, common + std::min<U>(512, 10 * t) * expert_bytes,
            2 * shared_router * t + (nvfp4 ? 0 : routed_flops), nvfp4 ? routed_flops : 0, 0};
}

constexpr Work gdn(U t, U batch, bool record, bool prefill = false, bool fp8_projections = false) {
    constexpr U parameters = 2560 * (2 * 48 + 10240 + 6144) + 2560 * 6144;
    const U weight_bytes   = 2 * 2560 * 2 * 48 + projection_bytes(10240, 2560, fp8_projections) +
                           projection_bytes(6144, 2560, fp8_projections) +
                           projection_bytes(2560, 6144, fp8_projections);
    // One state read, and one write only for state-update paths. Record paths publish
    // convolution/key/value/gate records instead. Sequential recurrence has at least two
    // matrix-vector products and one outer-product update (6 FLOPs per state element).
    constexpr U state = 48 * 128 * 128;
    const U records   = record ? t * (2 * (10240 + 128 * 16 + 128 * 48) + 4 * 2 * 48) : 0;
    auto work = dense_stored(parameters, weight_bytes, t,
                             4 * 2560 * t + 4 * state * batch * (record ? 1 : 2) + records);
    // Chunked prefill uses a different arithmetic decomposition, including tensor cores;
    // do not charge the sequential formula to FP32 CUDA cores for that route.
    work.fp32_flops = prefill ? 0 : 6 * state * t;
    return work;
}

constexpr Work qsa(U t, bool reuse, bool fp8_query_gate_output = false) {
    // Attention and selection depend on device-resident positions/valid extents and chosen
    // indices. Do not substitute a graph's maximum context for those values. This is the
    // projection floor only; the report identifies it as such and retains context metadata.
    const U parameters = 2560 * (12288 + 512 + 512 + 128 + (reuse ? 0 : 512) + 6144);
    const U weight_bytes = projection_bytes(12288, 2560, fp8_query_gate_output) +
                           projection_bytes(2560, 6144, fp8_query_gate_output) +
                           2 * 2560 * (512 + 512 + 128 + (reuse ? 0 : 512));
    return dense_stored(parameters, weight_bytes, t, 4 * 2560 * t);
}

constexpr Work hyper(U t, bool injection, bool combine, bool fp8_down, bool fp8_up) {
    const U parameters   = 2 * 10240 * 320 + (injection ? 4 * 10240 : 0);
    const U weight_bytes = projection_bytes(320, 10240, fp8_down) +
                           projection_bytes(10240, 320, fp8_up) + (injection ? 2 * 4 * 10240 : 0);
    return dense_stored(
        parameters, weight_bytes, t,
        2 * t * (10240 + 2560 + (injection ? 4 : 0) + (combine ? 10240 + 2560 + 4 : 0)));
}

constexpr Work ple(U t) { return dense(2560 * (10240 + 2560), t, t * (2 * 10240 * 2 + 2560)); }
} // namespace ninfer::ops::flash_next_work
#endif
