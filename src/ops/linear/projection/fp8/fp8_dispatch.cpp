#include "ops/linear/projection/fp8/fp8_dispatch.h"
#include "ops/common/host/operands.h"
#include "ops/linear/projection/fp8/fp8_shapes.h"
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {
const std::array kShapes{&kFp8N14336K5120, &kFp8N16384K5120, &kFp8N34816K5120,
                         &kFp8N5120K6144,  &kFp8N5120K17408, &kFp8N248320K5120,
                         &kFp8N10240K2560, &kFp8N6144K2560,  &kFp8N2560K6144,
                         &kFp8N320K10240,  &kFp8N248320K2560, &kFp8ProjN12288K2560,
                         &kFp8ProjN10240K320, &kFp8ProjN2560K640,
                         // The role-split and fused-half shapes a width-2 rank presents. Appended,
                         // never reordered: the lookup is linear first-match, so the fourteen above
                         // resolve exactly as they did.
                         &kFp8N5120K5120,  &kFp8N3072K5120,  &kFp8N512K5120, &kFp8N8704K5120,
                         &kFp8N7168K5120,  &kFp8N8192K5120,  &kFp8N124160K5120,
                         // The column-sharded outputs: the two 5120x6144 projections at half K,
                         // and mlp/down at half K. Appended for the same reason.
                         &kFp8N5120K3072,  &kFp8N5120K8704,
                         // The width-1 presentations: the GDN input projection's two parents, and
                         // the attention key/value roles.
                         &kFp8N10240K5120, &kFp8N6144K5120, &kFp8N1024K5120};

// A refused shape's FULL descriptor, not just n and k. n and k do not identify the weight: a fused
// consumer and a repack gather can both present a shape that no binding in the artifact declares,
// and the descriptor is what tells those apart. `ndim` and `shape[]` are the object's own geometry,
// so a square `shape` behind a non-square object — or the reverse — is visible here; `qdata`
// identifies which binding it came from when correlated against a binding dump. Printed on the
// throw, so no environment variable has to be known in advance.
std::string describe(const Weight& weight) {
    std::string out = "N" + std::to_string(weight.n) + "K" + std::to_string(weight.k) + " [ndim=" +
                      std::to_string(weight.ndim) + " shape=";
    for (std::uint32_t axis = 0; axis < weight.ndim && axis < 4; ++axis) {
        out += (axis ? "," : "") + std::to_string(weight.shape[axis]);
    }
    out += " padded=";
    for (std::uint32_t axis = 0; axis < weight.ndim && axis < 4; ++axis) {
        out += (axis ? "," : "") + std::to_string(weight.padded_shape[axis]);
    }
    out += " qtype=" + std::to_string(static_cast<int>(weight.qtype)) + " layout=" +
           std::to_string(static_cast<int>(weight.layout)) + " group=" +
           std::to_string(weight.group) + " payload_bytes=" +
           std::to_string(weight.payload_bytes) + " qdata=";
    const auto* address = static_cast<const void*>(weight.qdata);
    char text[32];
    std::snprintf(text, sizeof(text), "%p", address);
    return out + text + "]";
}

const Fp8LinearShape& resolve_shape(const Weight& weight, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("fp8 linear: unsupported policy");
    for (const auto* shape : kShapes)
        if (shape->n == weight.n && shape->k == weight.k) return *shape;
    // Name the shape AND the descriptor: "unsupported shape" alone does not say which weight was
    // refused, and on the two-GPU lane the caller is looking at a shape it did not write by hand.
    throw std::invalid_argument("fp8 linear: unsupported shape " + describe(weight));
}
} // namespace

std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k, LinearPolicy policy,
                                                std::int32_t min_tokens, std::int32_t max_tokens,
                                                std::string_view tag) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("fp8 linear workspace: invalid token interval");
    // The capacity query has no Weight — it runs during planning, before materialization — so the
    // descriptor is empty by construction and the only thing that can name the request is the
    // caller's tag. It is printed with the shape on the refusal.
    const Weight weight{.n = n, .k = k};
    const auto& shape = [&]() -> const Fp8LinearShape& {
        for (const auto* candidate : kShapes)
            if (candidate->n == n && candidate->k == k) return *candidate;
        throw std::invalid_argument("fp8 linear workspace: unsupported shape N" + std::to_string(n) +
                                    "K" + std::to_string(k) + " for " +
                                    (tag.empty() ? std::string("<untagged>") : std::string(tag)));
    }();
    return allows_a8(policy) && shape.uses_a8(min_tokens, max_tokens)
               ? fp8_a8_workspace_capacity_bytes(
                     max_tokens, k,
                     shape.partial_capacity_bytes ? shape.partial_capacity_bytes(max_tokens) : 0)
               : 0;
}

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  MemoryArena* workspace, cudaStream_t stream) {
    common::validate_fp8_weight(weight, "fp8 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("fp8 linear: T must be positive");
    const auto& shape = resolve_shape(weight, policy);
    if (!allows_a8(policy) || !shape.uses_a8(x.ne[1], x.ne[1]))
        return shape.a16(x, weight, out, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("fp8 A8 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_fp8_a8_workspace(
        *workspace, x.ne[1], weight.k,
        shape.partial_capacity_bytes ? shape.partial_capacity_bytes(x.ne[1]) : 0);
    shape.a8(x, weight, out, scratch, stream);
}
} // namespace ninfer::ops::detail
