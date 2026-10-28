#include "ops/linear/exl3/exl3_dense_dispatch.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

// The registered problems: [N, K] pairs of the 27B dense-route projections.
constexpr int kProblems[][2] = {
    {1024, 5120},   // attention key and value projections
    {2048, 5120},   // GDN query and key projections (fused-mixer slices)
    {5120, 6144},   // attention output / GDN out projections
    {6144, 5120},   // attention query and gate projections, GDN value and z
    {10240, 5120},  // GDN fused qkv projection
    {17408, 5120},  // MLP gate and up projections
    {5120, 17408},  // MLP down projection
};

constexpr std::int32_t kMaxTokens = 8;  // the decode shapes the route registers

} // namespace

bool exl3_dense_registered(std::int32_t n, std::int32_t k) {
    if (n <= 0 || k <= 0) { return false; }
    for (const auto& problem : kProblems) {
        if (problem[0] == n && problem[1] == k) { return true; }
    }
    return false;
}

std::size_t exl3_dense_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t max_tokens) {
    if (max_tokens <= 0 || max_tokens > kMaxTokens) {
        throw std::invalid_argument("exl3 dense workspace: token interval outside 1..8");
    }
    if (!exl3_dense_registered(n, k)) {
        throw std::invalid_argument("exl3 dense workspace: unregistered problem " +
                                    std::to_string(n) + "x" + std::to_string(k));
    }
    // Stage 1 stages one FP32 partial z per (k chunk, output row, token); stage 2 reduces the
    // chunks and applies the output transform.
    const std::int64_t chunks = (static_cast<std::int64_t>(k) + 511) / 512;
    return static_cast<std::size_t>(chunks) * static_cast<std::size_t>(n) *
           static_cast<std::size_t>(max_tokens) * sizeof(float);
}

std::uint32_t exl3_dense_k2(const Weight& w) {
    if (w.exl3_k2 != 0) { return w.exl3_k2; }
    if (w.exl3_trellis_offset != 0 || w.exl3_suh_offset <= w.exl3_trellis_offset) {
        throw std::invalid_argument("exl3 dense k2: payload planes are not dense layout");
    }
    const std::size_t extent = w.exl3_suh_offset - w.exl3_trellis_offset;
    const auto total = static_cast<std::size_t>(w.n) * static_cast<std::size_t>(w.k);
    if (extent * 16 % total != 0) {
        throw std::invalid_argument("exl3 dense k2: trellis extent matches no bitrate");
    }
    const auto derived = extent * 16 / total;
    if (derived < 2 || derived > 16) {
        throw std::invalid_argument("exl3 dense k2: bitrate outside the admitted range");
    }
    return static_cast<std::uint32_t>(derived);
}

Weight exl3_dense_slice(const Weight& parent, std::int32_t row_begin, std::int32_t rows) {
    if (parent.qtype != QType::EXL3 || parent.layout != QuantLayout::Exl3Dense ||
        parent.exl3_experts != 1) {
        throw std::invalid_argument("exl3 dense slice: parent is not a dense EXL3 object");
    }
    if (row_begin < 0 || rows <= 0 || row_begin % 16 != 0 || row_begin + rows > parent.n) {
        throw std::invalid_argument("exl3 dense slice: rows outside the parent");
    }
    const std::uint32_t k2 = exl3_dense_k2(parent);
    Weight slice           = parent;  // payload, k, codebook, suh offset are the parent's
    slice.n = slice.shape[0] = slice.padded_shape[0] = rows;
    slice.exl3_k2         = k2;
    slice.exl3_row_stride = static_cast<std::uint32_t>(parent.n);
    // Row band bytes inside the parent trellis: (row_begin/16) tiles of 16*k2 bytes each.
    slice.exl3_trellis_offset += static_cast<std::uint64_t>(row_begin) * k2;
    slice.exl3_svh_offset += static_cast<std::uint64_t>(row_begin) * 2;
    return slice;
}

void exl3_dense_dispatch(const Tensor& x, const Weight& w, Tensor& out, MemoryArena* workspace,
                         cudaStream_t stream) {
    if (w.layout != QuantLayout::Exl3Dense) {
        throw std::invalid_argument("exl3 dense linear: weight is not an exl3_dense_v1 object");
    }
    if (w.exl3_codebook == 0) {
        throw std::invalid_argument("exl3 dense linear: 3INST banks have no qualified route");
    }
    if (w.exl3_experts != 1) {
        throw std::invalid_argument("exl3 dense linear: dense objects carry exactly one linear");
    }
    if (!exl3_dense_registered(w.n, w.k)) {
        throw std::invalid_argument("exl3 dense linear: unregistered problem " +
                                    std::to_string(w.n) + "x" + std::to_string(w.k));
    }
    const std::int32_t tokens = out.ne[1];
    if (tokens <= 0 || tokens > kMaxTokens) {
        throw std::invalid_argument("exl3 dense linear: registers decode shapes T in 1..8, got " +
                                    std::to_string(tokens));
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("exl3 dense linear requires caller workspace");
    }
    auto scope = workspace->scope();
    auto scratch = workspace->alloc_bytes(exl3_dense_workspace_bytes(w.n, w.k, tokens));
    launch_exl3_dense(x, w, out, scratch.data, stream);
}

} // namespace ninfer::ops::detail
