#pragma once

#include <stdexcept>

namespace ninfer::ops::detail {

template <int QHeadsValue, int KVHeadsValue>
struct AttentionHeadMapping {
    static_assert(QHeadsValue > 0 && KVHeadsValue > 0);
    static_assert(QHeadsValue % KVHeadsValue == 0);

    static constexpr int QHeads    = QHeadsValue;
    static constexpr int KVHeads   = KVHeadsValue;
    static constexpr int GroupSize = QHeads / KVHeads;
};

inline constexpr int kCausalHeadDim = 256;

template <int HeadDim, int QueryHeads, int KVHeads>
struct CausalGeometry : AttentionHeadMapping<QueryHeads, KVHeads> {
    static_assert(HeadDim > 0 && HeadDim % 64 == 0);
    static constexpr int kHeadDim = HeadDim;
};

using CausalD256H24Kv4 = CausalGeometry<256, 24, 4>;
using CausalD256H16Kv2 = CausalGeometry<256, 16, 2>;
// The 27B's per-rank shape at tensor-parallel width 2: 24 query heads and 4 KV heads split evenly
// by rows, giving 12 and 2. Its group size is 6, the same as 24/4, so every schedule and every
// key-row tiling the 24/4 instance already runs applies unchanged — only the strides, the grid
// extents and the KV-head count in the cache layout differ.
using CausalD256H12Kv2 = CausalGeometry<256, 12, 2>;

// The registered causal head geometries. One table feeds both the plan builders' acceptance test
// and the launch dispatch, so a head count cannot be accepted by a plan and then dispatched to the
// wrong instance. `kv_heads` is what the plan builders derive their split budgets from; stating it
// here rather than in a `heads == 24 ? 4 : 2` ternary is what keeps a third geometry from being
// silently mis-tuned.
struct CausalHeadCounts {
    int query_heads;
    int kv_heads;
};

inline constexpr CausalHeadCounts kCausalHeadCounts[] = {
    {24, 4}, // D256H24Kv4 — Qwen3.5/3.6/3.8 27B, single device
    {16, 2}, // D256H16Kv2 — the smaller dense members
    {12, 2}, // D256H12Kv2 — the 27B at tensor-parallel width 2
};

// True and the KV head count when `query_heads` is a registered geometry.
[[nodiscard]] constexpr bool causal_kv_heads(int query_heads, int* kv_heads) noexcept {
    for (const auto& geometry : kCausalHeadCounts) {
        if (geometry.query_heads == query_heads) {
            *kv_heads = geometry.kv_heads;
            return true;
        }
    }
    return false;
}

// Dispatches `call` on the geometry registered for `query_heads`. Host-side only: every call site
// is launch code. An unregistered head count throws rather than falling through to another
// instance, which is how a 12-head rank used to reach the 16/2 kernels and fail inside them.
template <class Call>
inline void with_causal_geometry(int query_heads, Call&& call) {
    switch (query_heads) {
    case 24: call.template operator()<CausalD256H24Kv4>(); return;
    case 16: call.template operator()<CausalD256H16Kv2>(); return;
    case 12: call.template operator()<CausalD256H12Kv2>(); return;
    default:
        throw std::invalid_argument("causal attention: unregistered query head count " +
                                    std::to_string(query_heads));
    }
}

} // namespace ninfer::ops::detail
