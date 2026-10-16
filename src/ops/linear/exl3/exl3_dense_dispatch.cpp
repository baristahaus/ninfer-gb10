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
    {5120, 6144},   // attention output / GDN out projections
    {6144, 5120},   // attention query and gate projections
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

void exl3_dense_dispatch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena* workspace,
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
