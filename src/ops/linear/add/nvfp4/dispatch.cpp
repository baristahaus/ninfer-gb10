#include "core/weight.h"
#include "ops/linear/add/nvfp4/dispatch.h"

#include <stdexcept>
#include <cstdio>
#include <string>

namespace ninfer::ops::detail {
namespace {
// The refused weight's full descriptor. n and k alone do not identify it: a fused consumer or a
// repack gather can present a shape no artifact binding declares, and `ndim`/`shape[]` plus the
// qdata address are what tell those apart against a binding dump.
std::string describe_weight(const Weight& w) {
    std::string out = "N" + std::to_string(w.n) + "K" + std::to_string(w.k) + " [ndim=" +
                      std::to_string(w.ndim) + " shape=";
    for (std::uint32_t axis = 0; axis < w.ndim && axis < 4; ++axis) {
        out += (axis ? "," : "") + std::to_string(w.shape[axis]);
    }
    out += " qtype=" + std::to_string(static_cast<int>(w.qtype)) + " layout=" +
           std::to_string(static_cast<int>(w.layout)) + " payload_bytes=" +
           std::to_string(w.payload_bytes) + " qdata=";
    char text[32];
    std::snprintf(text, sizeof(text), "%p", static_cast<const void*>(w.qdata));
    return out + text + "]";
}


Nvfp4LinearAddPlan select_plan(std::int32_t rows, std::int32_t k, LinearPolicy policy,
                               std::int32_t tokens) {
    if (tokens > 0 && rows == 5120) {
        if (k == 6144) return select_nvfp4_add_n5120_k6144(policy, tokens);
        if (k == 17408) return select_nvfp4_add_n5120_k17408(policy, tokens);
        if (k == 8704) return select_nvfp4_add_n5120_k8704(policy, tokens);
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported shape " +
                                describe_weight(Weight{.n = rows, .k = k}));
}
} // namespace

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t rows, std::int32_t k,
                                                      LinearPolicy policy, std::int32_t first,
                                                      std::int32_t last) {
    if (first <= 0 || last < first)
        throw std::invalid_argument("nvfp4 linear_add: invalid token interval");
    (void)select_plan(rows, k, policy, first);
    const auto plan = select_plan(rows, k, policy, last);
    return plan.a16 ? 0 : nvfp4_a4_workspace_capacity_bytes(last, k);
}

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                               MemoryArena& arena, cudaStream_t stream) {
    const auto plan = select_plan(w.n, w.k, policy, x.ne[1]);
    if (plan.a16) return plan.a16(x, w, out, stream);
    auto scope         = arena.scope();
    const auto scratch = allocate_nvfp4_a4_workspace(arena, x.ne[1], w.k);
    launch_nvfp4_a4_quantize(x, w, scratch, plan.scale_layout, stream);
    plan.quantized(x, w, out, scratch, stream);
}
} // namespace ninfer::ops::detail
